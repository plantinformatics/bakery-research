from langchain_core.documents.base import Document
import os, glob, hashlib, json, argparse
from typing import Any, List, Literal, Tuple
import rdflib
import asyncio
from rdflib.namespace import RDF
from rdflib import RDF, Namespace
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import LLMGraphTransformer, Neo4jGraph, Neo4jVector
import csv
import time
import logging
from pathlib import Path
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import yaml
from dotenv import load_dotenv

# Enviroment imports (keys ect)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Model Specific config
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_EXTRACTION_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Chunking config
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Relationship config
MAX_CHARACTERS = 30000
MAX_TRIPLES = 50
QUERY_VECTOR_MAX_CHUNKS = 1
QUERY_FULL_TEXT_MAX_CHUNKS = 1
QUERY_MAX_CHUNKS = 16

# Retry strategy (for LLM calls)
RetryStrategy = Literal["exponential", "linear"]
GLOBAL_MAX_RETRIES = 5


def get_retry_wait_s(attempts: int, strategy: RetryStrategy = "exponential") -> float:
    if strategy not in ("exponential", "linear"):
        raise ValueError(f"Unknown retry strategy: {strategy}")

    wait_min = 2 ** (attempts - 1) if strategy == "exponential" else 5 * attempts
    if strategy == "exponential" and attempts > 5:
        logging.warning(
            "Exponential retry attempt %s exceeds 5; next wait is %.0f minutes ",
            attempts,
            wait_min,
        )
    return wait_min * 60.0


# Logging goes to console and and respective log files
def setup_logging(log_file: str) -> None:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )


# def parse_schema_for_llm(schema_path: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
#         # Parse SHACL Turtle to extract allowed Nodes and Relationships.
#         g = rdflib.Graph()
#         g.parse(schema_path, format="turtle")
#         SH = Namespace("http://www.w3.org/ns/shacl#")
#         allowed_nodes = set()
#         allowed_rels_tuples = [] # (Source, Rel, Target)
#         for shape in g.subjects(RDF.type, SH.NodeShape):
#             target_class = g.value(shape, SH.targetClass)
#             if not target_class:
#                 continue
#             source_node = target_class.split('#')[-1]
#             allowed_nodes.add(source_node)
#             for prop in g.objects(shape, SH.property):
#                 path = g.value(prop, SH.path)
#                 class_constraint = g.value(prop, SH["class"])
#                 # If sh:class exists, it's a relationship to another node.
#                 if path and class_constraint:
#                     rel_name = path.split('#')[-1].upper().replace("-", "_")
#                     target_node = class_constraint.split('#')[-1]
#                     allowed_nodes.add(target_node)
#                     allowed_rels_tuples.append((source_node, rel_name, target_node))
#         return list(allowed_nodes), allowed_rels_tuples


def parse_schema_for_llm(
    schema_path: str,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    with open(schema_path, encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}
    allowed_nodes = list(schema.get("entities", {}))
    allowed_relationships = []
    for relationship_name, spec in schema.get("relationships", {}).items():
        source_nodes = (
            spec["from"] if isinstance(spec["from"], list) else [spec["from"]]
        )
        target_nodes = spec["to"] if isinstance(spec["to"], list) else [spec["to"]]
        for source_node in source_nodes:
            for target_node in target_nodes:
                allowed_relationships.append(
                    (source_node, relationship_name, target_node)
                )
    return allowed_nodes, allowed_relationships


class PlantBioRAG:
    def __init__(
        self, md_dir: str, add_dir: str, pretzel_functions_dir: str, schema_path: str
    ):
        self.md_dir = md_dir
        self.add_dir = add_dir
        self.pretzel_functions_dir = pretzel_functions_dir
        self.schema_path = schema_path
        self.emb = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        self.extraction_llm = ChatGoogleGenerativeAI(
            model=GEMINI_EXTRACTION_MODEL, temperature=0
        )
        self.graph = Neo4jGraph()
        self.allowed_nodes, self.allowed_rels = parse_schema_for_llm(schema_path)
        self.vs = Neo4jVector(
            embedding=self.emb,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
            index_name="vector",
        )
        self.vs_pretzel_functions = Neo4jVector(
            embedding=self.emb,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            node_label="PretzelFunction",
            text_node_property="text",
            embedding_node_property="embedding",
            index_name="pretzel_functions_vector",
        )
        headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.failed_docs_file = Path(r"failed_docs.jsonl")

    # Part 1: Build
    # 1. Load markdown files from a folder or a single file and split to Documents.
    def load_and_split(self, path) -> List[Document]:
        all_chunks = []
        if os.path.isfile(path):
            paths = [path]
        else:  # route if directory is supplied
            paths = glob.glob(os.path.join(path, "**", "*.md"), recursive=True)
        if not paths:
            logging.warning("No markdown files found at %s", path)
            return all_chunks
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Split by header first
            md_docs = self.md_splitter.split_text(content)
            # Split to chunks
            chunks = self.text_splitter.split_documents(md_docs)
            # Add metadata for file tracking
            for i, chunk in enumerate[Document](chunks):
                source_path = os.path.basename(path)
                # Added fixed chunking IDs so they don't change between runs
                chunk_id = hashlib.sha256(
                    f"{source_path}::{i}::{chunk.page_content}".encode("utf-8")
                ).hexdigest()
                chunk.metadata["source_path"] = source_path
                chunk.metadata["chunk_id"] = chunk_id
                chunk.id = chunk_id
            all_chunks.extend(chunks)
        return all_chunks

    # 2. Upsert chunks and vectors to a Neo4j graph.
    def upsert_chunks_and_vectors(self, docs: List[Document]):
        # Create Chunk nodes with text and embedding via Neo4jVector
        # Make sure uniqueness constraint for chunk_id
        self.graph.query(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )

        max_retries = GLOBAL_MAX_RETRIES
        batch_size = 20
        logging.info(f"len(docs): {len(docs)}")
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self.vs.add_documents(batch_docs)
                    logging.info(f"Embedded batch {i} to {i + len(batch_docs) - 1}")
                    time.sleep(0.5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = get_retry_wait_s(attempt + 1)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} after {wait_s:.1f}s due to: {e}"
                        )
                        time.sleep(wait_s)
                    else:
                        raise

    # 3. Creat vector index and full-text index.
    def create_indexes(self):
        self.graph.query(
            """
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 3072,
              `vector.similarity_function`: 'cosine'
            }}
        """
        )

        # Full-text over chunk text
        self.graph.query(
            "CREATE FULLTEXT INDEX idx_chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
        )
        # Full-text over entity names for allowed node labels
        labels = "|".join(self.allowed_nodes) if self.allowed_nodes else "Entity"
        self.graph.query(
            f"CREATE FULLTEXT INDEX idx_node_name IF NOT EXISTS FOR (n:{labels}) ON EACH [n.name]"
        )

    def _save_failed_chunks(self, chunks: List[Document]):
        self.failed_docs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.failed_docs_file, "a", encoding="utf-8") as f:
            json.dump(
                [
                    {"page_content": c.page_content, "metadata": c.metadata}
                    for c in chunks
                ],
                f,
                ensure_ascii=False,
                default=str,
            )
            f.write("\n")

    # 4. Extract nodes and relationships from document text.
    def extract_graph_and_link_mentions(self, chunks: List[Document]):
        original_count = len(chunks)
        # Filtering for chunks with no content because of overlap
        chunks = [c for c in chunks if len(c.page_content) >= CHUNK_OVERLAP]
        if not chunks:
            logging.info("No chunk to process.")
            return
        if len(chunks) < original_count:
            logging.info(
                f"Filtered out {original_count - len(chunks)} short chunk(s); "
                f"{len(chunks)} remaining."
            )
        xformer = LLMGraphTransformer(
            llm=self.extraction_llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_rels,
        )

        max_retries = GLOBAL_MAX_RETRIES
        batch_size = 8
        logging.info(f"len(chunks): {len(chunks)}")

        input_tokens_sum = 0
        output_tokens_sum = 0
        total_tokens_sum = 0
        for i in range(0, len(chunks), batch_size):
            logging.info(f"index i: {i}")
            batch = chunks[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    logging.info(
                        f"Processing batch starting at index {i}, attempt {attempt + 1}/{max_retries}"
                    )

                    handler = UsageMetadataCallbackHandler()
                    gdocs = asyncio.run(
                        xformer.aconvert_to_graph_documents(
                            batch, config={"callbacks": [handler]}
                        )
                    )

                    logging.info(f"Batch {i} token usage: {handler.usage_metadata}")

                    input_tokens_sum += sum(
                        v.get("input_tokens", 0)
                        for v in handler.usage_metadata.values()
                    )
                    output_tokens_sum += sum(
                        v.get("output_tokens", 0)
                        for v in handler.usage_metadata.values()
                    )
                    total_tokens_sum += sum(
                        v.get("total_tokens", 0)
                        for v in handler.usage_metadata.values()
                    )

                    if not gdocs:
                        logging.info(
                            f"No graph documents returned for batch {i} to {i + len(batch) - 1}"
                        )
                        break

                    logging.info("convert_to_graph_documents")

                    self.graph.add_graph_documents(gdocs)

                    logging.info("add_graph_documents")

                    # Group mentions by node.type to batch Cypher queries
                    mentions_by_type = {}
                    for gd in gdocs:
                        chunk_id = gd.source.metadata.get("chunk_id")
                        if not chunk_id:
                            continue

                        for node in gd.nodes:
                            mentions_by_type.setdefault(node.type, []).append(
                                {"chunk_id": chunk_id, "node_id": node.id}
                            )

                    # Execute one batched query per node type using UNWIND
                    for node_type, params in mentions_by_type.items():
                        cypher = f"""
                        UNWIND $params AS row
                        MATCH (c:Chunk {{chunk_id: row.chunk_id}})
                        MERGE (n:{node_type} {{id: row.node_id}})
                        MERGE (c)-[:MENTIONS]->(n)
                        """
                        self.graph.query(cypher, params={"params": params})

                    logging.info(f"Completed batch {i} to {i + len(batch) - 1}")
                    time.sleep(0.5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = get_retry_wait_s(attempt + 1)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for batch {i} to {i + len(batch) - 1} "
                            f"after {wait_s:.1f}s due to: {e}"
                        )
                        time.sleep(wait_s)
                    else:
                        logging.exception(
                            f"Failed batch {i} to {i + len(batch) - 1} after {max_retries} attempts: {e}"
                        )
                        self._save_failed_chunks(batch)

        logging.info(f"Input tokens sum: {input_tokens_sum}")
        logging.info(f"Output tokens sum: {output_tokens_sum}")
        logging.info(f"Total tokens sum: {total_tokens_sum}")

    def build(self):
        try:
            setup_logging(r"build.log")
            logging.info("Build started.")

            docs = self.load_and_split(self.md_dir)
            if not docs:
                logging.warning("No documents to process at", self.md_dir)
                return
            logging.info("Completed load_and_split")

            self.upsert_chunks_and_vectors(docs)
            logging.info("Completed upsert_chunks_and_vectors")

            self.create_indexes()
            logging.info("Completed create_indexes")

            self.extract_graph_and_link_mentions(docs)
            logging.info("Build completed.")
        except Exception as e:
            logging.exception(e)

    def upsert_chunks_without_vectors(self, docs: List[Document]):
        rows = [
            {
                "chunk_id": d.metadata["chunk_id"],
                "source_path": d.metadata["source_path"],
                "text": d.page_content,
            }
            for d in docs
        ]
        self.graph.query(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.source_path = row.source_path,
                c.text = row.text
            """,
            params={"rows": rows},
        )

    # Add documents
    def add(self, extract_nodes):
        try:
            setup_logging(r"add.log")
            logging.info("Add started.")

            docs = self.load_and_split(self.add_dir)
            if not docs:
                logging.warning("No documents to process at", self.add_dir)
                return
            logging.info("Completed load_and_split")

            if extract_nodes:
                self.upsert_chunks_and_vectors(docs)
                logging.info("Completed upsert_chunks_and_vectors")

                self.extract_graph_and_link_mentions(docs)
                logging.info("Add completed.")
            else:
                self.upsert_chunks_without_vectors(docs)
        except Exception as e:
            logging.exception(e)

    def upsert_pretzel_functions_and_vectors(self, docs: List[Document]):
        self.graph.query(
            """
            CREATE CONSTRAINT pretzel_function_id_unique IF NOT EXISTS
            FOR (p:PretzelFunction)
            REQUIRE p.id IS UNIQUE"""
        )

        self.graph.query(
            """
            CREATE VECTOR INDEX pretzel_functions_vector IF NOT EXISTS
            FOR (p:PretzelFunction) ON (p.embedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: 3072,
            `vector.similarity_function`: 'cosine'
            }}"""
        )

        self.graph.query(
            """
            CREATE FULLTEXT INDEX idx_pretzel_function_text IF NOT EXISTS
            FOR (p:PretzelFunction) ON EACH [p.text]"""
        )

        max_retries = GLOBAL_MAX_RETRIES
        batch_size = 4
        logging.info(f"len(docs): {len(docs)}")
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self.vs_pretzel_functions.add_documents(batch_docs)
                    logging.info(f"Embedded batch {i} to {i + len(batch_docs) - 1}")
                    time.sleep(0.5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = get_retry_wait_s(attempt + 1)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} after {wait_s:.1f}s due to: {e}"
                        )
                        time.sleep(wait_s)
                    else:
                        raise

    def add_pretzel_functions(self):
        try:
            setup_logging(r"addpretzelfunctions.log")
            logging.info("Add pretzel functions started.")

            docs = self.load_and_split(self.pretzel_functions_dir)
            if not docs:
                logging.warning("No documents to process; exiting.")
                return
            logging.info("Completed load_and_split")

            self.upsert_pretzel_functions_and_vectors(docs)
            logging.info("Completed upsert_chunks_and_vectors")
        except Exception as e:
            logging.exception(e)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build or incrementally add to the literature and pretzel documentation graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  build        full graph from literature\n"
            "  add          add docs with node extraction\n"
            "  add_sup      add docs without node extraction\n"
            "  add_pretzel  add pretzel function docs"
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["build", "add", "add_sup", "add_pretzel"],
        help="Operation to run. See modes below.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--dir",
        default=None,
        help=(
            "Input markdown directory. Defaults: md_dir (build), "
            "add_dir (add, add_sup), add_pretzel_functions (add_pretzel)."
        ),
    )
    input_group.add_argument(
        "--file",
        default=None,
        help="Process a single markdown file instead of a directory.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to schema YAML. Defaults to schema.yaml in the same folder.",
    )
    args = parser.parse_args()

    if args.file is not None and not os.path.isfile(args.file):
        parser.error(f"File not found: {args.file}")

    md_dir = str(here / "md_dir")
    add_dir = str(here / "add_dir")
    pretzel_functions_dir = str(here / "add_pretzel_functions")
    schema = args.schema if args.schema is not None else str(here / "schema.yaml")

    input_path = args.file if args.file is not None else args.dir
    if input_path is not None:
        if args.mode == "build":
            md_dir = input_path
        elif args.mode in ("add", "add_sup"):
            add_dir = input_path
        elif args.mode == "add_pretzel":
            pretzel_functions_dir = input_path

    rag = PlantBioRAG(md_dir, add_dir, pretzel_functions_dir, schema)

    if args.mode == "build":
        rag.build()
    elif args.mode == "add":
        rag.add(extract_nodes=True)
    elif args.mode == "add_sup":
        rag.add(extract_nodes=False)
    elif args.mode == "add_pretzel":
        rag.add_pretzel_functions()


if __name__ == "__main__":
    main()
