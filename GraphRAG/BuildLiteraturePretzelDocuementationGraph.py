import os, glob, uuid, json, argparse
from typing import List, Tuple, Dict
import rdflib
from rdflib.namespace import RDF
from rdflib import RDF, Namespace
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import csv
import time
import logging
from pathlib import Path
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = ""
os.environ["NEO4J_URI"] = ""
os.environ["NEO4J_USERNAME"] = ""
os.environ["NEO4J_PASSWORD"] = ""

GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_EXTRACTION_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
MAX_CHARACTERS = 30000
MAX_TRIPLES = 50
QUERY_VECTOR_MAX_CHUNKS = 1
QUERY_FULL_TEXT_MAX_CHUNKS = 1
QUERY_MAX_CHUNKS = 16

def parse_schema_for_llm(schema_path: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        # Parse SHACL Turtle to extract allowed Nodes and Relationships. 
        g = rdflib.Graph()
        g.parse(schema_path, format="turtle")        
        SH = Namespace("http://www.w3.org/ns/shacl#")        
        allowed_nodes = set()
        allowed_rels_tuples = [] # (Source, Rel, Target)
        for shape in g.subjects(RDF.type, SH.NodeShape):
            target_class = g.value(shape, SH.targetClass)
            if not target_class:
                continue            
            source_node = target_class.split('#')[-1]
            allowed_nodes.add(source_node)
            for prop in g.objects(shape, SH.property):
                path = g.value(prop, SH.path)
                class_constraint = g.value(prop, SH["class"])                
                # If sh:class exists, it's a relationship to another node. 
                if path and class_constraint:
                    rel_name = path.split('#')[-1].upper().replace("-", "_")
                    target_node = class_constraint.split('#')[-1]
                    allowed_nodes.add(target_node)
                    allowed_rels_tuples.append((source_node, rel_name, target_node))
        return list(allowed_nodes), allowed_rels_tuples


class PlantBioRAG:
    def __init__(self, md_dir: str, add_dir: str, pretzel_functions_dir: str, schema_path: str):
        self.md_dir = md_dir
        self.add_dir = add_dir
        self.pretzel_functions_dir = pretzel_functions_dir
        self.schema_path = schema_path
        self.emb = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        self.extraction_llm = ChatGoogleGenerativeAI(model=GEMINI_EXTRACTION_MODEL, temperature=0)
        self.graph = Neo4jGraph()
        self.allowed_nodes, self.allowed_rels = parse_schema_for_llm(schema_path)
        self.vs = Neo4jVector(embedding=self.emb, url=os.getenv("NEO4J_URI"),
                              username=os.getenv("NEO4J_USERNAME"),
                              password=os.getenv("NEO4J_PASSWORD"),
                              node_label="Chunk", text_node_property="text",
                              embedding_node_property="embedding", 
                              index_name="vector")
        self.vs_pretzel_functions = Neo4jVector(embedding=self.emb, url=os.getenv("NEO4J_URI"),
                              username=os.getenv("NEO4J_USERNAME"),
                              password=os.getenv("NEO4J_PASSWORD"),
                              node_label="PretzelFunction", text_node_property="text",
                              embedding_node_property="embedding", 
                              index_name="pretzel_functions_vector")
        headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        self.failed_docs_file = Path(r"failed_docs.jsonl")

    # Part 1: Build 
    # 1. Load markdown files from folder and split to Documents. 
    def load_and_split(self, dir) -> List[Document]:    
        all_chunks = []
        for path in glob.glob(os.path.join(dir, "**", "*.md"), recursive=True):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Split by header first
            md_docs = self.md_splitter.split_text(content)
            # Split to chunks
            chunks = self.text_splitter.split_documents(md_docs)
            # Add metadata for file tracking
            for i, chunk in enumerate(chunks):
                chunk.metadata["source_path"] = os.path.basename(path)
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            all_chunks.extend(chunks)
        return all_chunks    
    
    def get_retry_wait_s(attempt: int) -> float:
        retry_waits = [60, 120, 300, 600, 1200]
        return retry_waits[min(attempt, len(retry_waits) - 1)]

    # 2. Upsert chunks and vectors to a Neo4j graph. 
    def upsert_chunks_and_vectors(self, docs: List[Document]):
        # Create Chunk nodes with text and embedding via Neo4jVector
        # Make sure uniqueness constraint for chunk_id
        self.graph.query("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        
        retry_waits = [60, 2*60, 3*60, 5*60, 10*60, 20*60, 30*60, 45*60, 60*60, 120*60, 240*60]
        max_retries = len(retry_waits)
        batch_size = 20
        logging.info(f"len(docs): {len(docs)}")
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    self.vs.add_documents(batch_docs)
                    logging.info(f"Embedded batch {i} to {i + len(batch_docs) - 1}")
                    time.sleep(0.5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = retry_waits[attempt]
                        logging.warning(f"Retry {attempt + 1}/{max_retries} after {wait_s:.1f}s due to: {e}")
                        time.sleep(wait_s)
                    else:
                        raise

    # 3. Creat vector index and full-text index. 
    def create_indexes(self):
        self.graph.query("""
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 3072,
              `vector.similarity_function`: 'cosine'
            }}
        """)

        # Full-text over chunk text
        self.graph.query("CREATE FULLTEXT INDEX idx_chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
        # Full-text over entity names for allowed node labels
        labels = "|".join(self.allowed_nodes) if self.allowed_nodes else "Entity"
        self.graph.query(f"CREATE FULLTEXT INDEX idx_node_name IF NOT EXISTS FOR (n:{labels}) ON EACH [n.name]")

    def _save_failed_docs(self, docs: List[Document]):
        self.failed_docs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.failed_docs_file, "a", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                f,
                ensure_ascii=False,
                default=str
            )
            f.write("\n")

    def _chunk_has_content(self, doc: Document) -> bool:
        if len(doc.page_content) < 200:
            return False
        return True
    
    # 4. Extract nodes and relationships from document text. 
    def extract_graph_and_link_mentions(self, docs: List[Document]):
        xformer = LLMGraphTransformer(
            llm=self.extraction_llm,
            allowed_nodes=self.allowed_nodes or None,
            allowed_relationships=self.allowed_rels
        )

        docs = [d for d in docs if self._chunk_has_content(d)]
        if not docs:
            logging.info("No chunk to process.")
            return
        
        retry_waits = [60, 2*60, 3*60, 5*60, 10*60, 20*60, 30*60, 45*60, 60*60, 120*60, 240*60]
        max_retries = len(retry_waits)
        batch_size = 4
        logging.info(f"len(docs): {len(docs)}")

        input_tokens_sum = 0
        output_tokens_sum = 0
        total_tokens_sum = 0
        for i in range(0, len(docs), batch_size):
            logging.info(f"index i: {i}")
            batch = docs[i : i+batch_size]
            for attempt in range(max_retries):
                try:
                    logging.info(f"Processing batch starting at index {i}, attempt {attempt + 1}/{max_retries}")

                    handler = UsageMetadataCallbackHandler()
                    gdocs = xformer.convert_to_graph_documents(batch, config={"callbacks": [handler]})

                    logging.info(f"Batch {i} token usage: {handler.usage_metadata}")
                    
                    input_tokens_sum += sum(v.get("input_tokens", 0) for v in handler.usage_metadata.values())
                    output_tokens_sum += sum(v.get("output_tokens", 0) for v in handler.usage_metadata.values())
                    total_tokens_sum += sum(v.get("total_tokens", 0) for v in handler.usage_metadata.values())

                    if not gdocs:
                        logging.info(f"No graph documents returned for batch {i} to {i + len(batch) - 1}")
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
                            mentions_by_type.setdefault(node.type, []).append({
                                "chunk_id": chunk_id,
                                "node_id": node.id
                            })

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
                        wait_s = retry_waits[attempt]
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for batch {i} to {i + len(batch) - 1} "
                            f"after {wait_s:.1f}s due to: {e}"
                        )
                        time.sleep(wait_s)
                    else:
                        logging.exception(
                            f"Failed batch {i} to {i + len(batch) - 1} after {max_retries} attempts: {e}"
                        )
                        self._save_failed_docs(batch)

        logging.info(f"Input tokens sum: {input_tokens_sum}")
        logging.info(f"Output tokens sum: {output_tokens_sum}")
        logging.info(f"Total tokens sum: {total_tokens_sum}")
    
    def build(self):
        try:
            logging.basicConfig(filename=r"build.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filemode="a")
            logging.info("Build started.")

            docs = self.load_and_split(self.md_dir)
            logging.info("Completed load_and_split")

            self.upsert_chunks_and_vectors(docs)
            logging.info("Completed upsert_chunks_and_vectors")

            self.create_indexes()
            logging.info("Completed create_indexes")

            self.extract_graph_and_link_mentions(docs)
            logging.info("Build completed.")
        except Exception as e:
            print(e)

    def upsert_chunks_without_vectors(self, docs: List[Document]):
        rows = [
            {
                "chunk_id": d.metadata["chunk_id"],
                "source_path": d.metadata["source_path"],
                "text": d.page_content,
            }
            for d in docs
        ]
        self.graph.query("""
            UNWIND $rows AS row
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.source_path = row.source_path,
                c.text = row.text
        """, params={"rows": rows})

    # Add documents 
    def add(self, extract_nodes):
        try:
            logging.basicConfig(filename=r"add.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filemode="a")
            logging.info("Add started.")

            docs = self.load_and_split(self.add_dir)
            logging.info("Completed load_and_split")

            if extract_nodes:
                self.upsert_chunks_and_vectors(docs)
                logging.info("Completed upsert_chunks_and_vectors")

                self.extract_graph_and_link_mentions(docs)
                logging.info("Add completed.")
            else:
                self.upsert_chunks_without_vectors(docs)
        except Exception as e:
            print(e)
    
    def upsert_pretzel_functions_and_vectors(self, docs: List[Document]):
        self.graph.query("""
            CREATE CONSTRAINT pretzel_function_id_unique IF NOT EXISTS
            FOR (p:PretzelFunction)
            REQUIRE p.id IS UNIQUE""")

        self.graph.query("""
            CREATE VECTOR INDEX pretzel_functions_vector IF NOT EXISTS
            FOR (p:PretzelFunction) ON (p.embedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: 3072,
            `vector.similarity_function`: 'cosine'
            }}""")

        self.graph.query("""
            CREATE FULLTEXT INDEX idx_pretzel_function_text IF NOT EXISTS
            FOR (p:PretzelFunction) ON EACH [p.text]""")

        retry_waits = [60, 2*60, 3*60, 5*60, 10*60, 20*60, 30*60, 45*60, 60*60, 120*60, 240*60]
        max_retries = len(retry_waits)
        batch_size = 4
        logging.info(f"len(docs): {len(docs)}")
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    self.vs_pretzel_functions.add_documents(batch_docs)
                    logging.info(f"Embedded batch {i} to {i + len(batch_docs) - 1}")
                    time.sleep(0.5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = retry_waits[attempt]
                        logging.warning(f"Retry {attempt + 1}/{max_retries} after {wait_s:.1f}s due to: {e}")
                        time.sleep(wait_s)
                    else:
                        raise

    def add_pretzel_functions(self):
        try:
            logging.basicConfig(filename=r"addpretzelfunctions.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filemode="a")
            logging.info("Add pretzel functions started.")

            docs = self.load_and_split(self.pretzel_functions_dir)
            logging.info("Completed load_and_split")

            self.upsert_pretzel_functions_and_vectors(docs)
            logging.info("Completed upsert_chunks_and_vectors")
        except Exception as e:
            print(e)


def main():
    mode = "build"
    md_dir = r""
    add_dir = r""
    pretzel_functions_dir = r""
    extract_nodes = False
    schema = r"shapes.ttl"

    rag = PlantBioRAG(md_dir, add_dir, pretzel_functions_dir, schema)
    
    if mode == "build":
        rag.build()
    elif mode == "add":
        rag.add(extract_nodes)
    elif mode == "add_pretzel_functions":
        rag.add_pretzel_functions()

if __name__ == "__main__":
    main()