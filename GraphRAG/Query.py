# RAG pipeline for plant biology papers using Neo4j + Gemini + LangChain
# Requires env: GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

import os
import argparse
from typing import List, Dict, Optional, Tuple, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import requests
import re
import json
import warnings
import logging
import time
from langchain_core.prompts import PromptTemplate
from neo4j.graph import Node, Relationship, Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.simplefilter("ignore", DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
MAX_CHARACTERS = 600000
MAX_TRIPLES = 50
QUERY_VECTOR_MAX_CHUNKS = 40
QUERY_FULL_TEXT_MAX_CHUNKS = 40
QUERY_MAX_CHUNKS = 80
MAX_METADATA_CHUNKS = 80
RERANK_MAX_TEXT_CHARS = 2000

# Accession API config
ACCESSION_API_URL = ""
ACCESSION_API_TOKEN = "research_accessions"
ACCESSION_API_TIMEOUT = 120

METADATA_MAX_CHARACTERS = 300000

global_instruction_and_information = """
You are an expert of a plant biology organisation. 
Background information: 
1. Pretzel is an open-sourced web-based online framework for the real-time interactive display integration of genetic and genomic datasets. It is built on Ember.js (front end), Loopback.js (back end) and D3.js (visualisation).
2. When user mentions Pretzel in their questions, this knowledge graph is the knowledge base of Pretzel. 
3. BlastDb tag means that there is a blast databases available to enable searching by sequence using a tool called BLAST. 
4. To be able to align two genome assemblies, the same kind of marker needs to be defined against them. 
5. Genetic maps are often referred to by the parents used, for example WAWHT2046 x AvocetS where WAWHT20246 and AvocetS are the parents (the order is not important). The parents of a genetic map are recorded in the Parent names field.
6. Genetic maps can be aligned to alignments if they have the same Marker type. 

A Genome dataset defines linear sequences representing chromosomes. A Genome dataset enables:
- If Blast is enabled (indicated by the Blastdb tag), the location of a given nucleotide sequence (in FASTA format) can be searched and located
- Within the chromosomes, genomic features can be defined in Annotation datasets, for example for genes, markers, and other features such as repeats

The Annotation dataset defines the genomic features within a given Genome dataset. Annotation datasets enable:
- Genomic features defined in a genome using an Annotation dataset can be searched by their ID
- Two chromosomes of different genomes can be aligned in Pretzel if markers of the same type are defined against them in an Annotation dataset
- If two chromosomes are aligned via a common feature or marker type, then a position on one chromosome can be projected into the other using the relative locations of the markers defined in both
- Combining the above, the relative location of features (markers, genes) can be found in relation to other features of interest, or locations identified by Blast-ing user-defined sequences

The Genetic Map dataset defines a linear order of markers organised by linkage group (or chromosome). Genetic Map datasets enable:
- When 2 Genetic Maps have been generated using the same marker type, they can be aligned
- If the markers defined in a Genetic Map are defined in an Annotation dataset associated with a Genome dataset, the Genetic Map can be aligned to the Genome
- Intervals in the Genetic Map can be projected into the Genome sequence using the relative position of common markers
- If the order of markers in a Genetic Map are inverted relative to the Genome orientation, the orientation can be flipped in Pretzel

The VCF dataset defines a genotype matrix of allele states for a set of accessions (samples) at a set of markers. VCF datasets include markers for which positions are defined against a given Genome, which defines the reference allele in the VCF file. For the location of the markers to be searchable, an Annotation dataset for the markers needs to be available in Pretzel. VCF datasets enable:
- The genotype calls (alleles) for samples defined in the file can be visualised at a given interval of the genome it is defined against
- For a given haplotype (pattern of alleles) manually input by the user, the number of samples in the VCF file matching that haplotype can be identified and their genotype data visualised
- Once genotype data is loaded into the Pretzel view, users can order the samples (accessions) based on their haplotype (allele pattern) by defining a haplotype manually
- Combining with other datasets, various combinations are possible, such as: 1) Visualising genotype data for a set of accessions around a gene or marker defined in an Annotation dataset; 2) Visualising genotype data for a set of accessions around a location in a Genome found by searching nucleotide sequence by Blast.
- More complex combinations of steps can be achieved, such as viewing the haplotypes among a set of accessions in the region of a Genome corresponding to a region defined in a Genetic Map by projecting the Genetic Map to the genome as described above

A QTL dataset defines single positions or intervals within a Genome or Genetic Map associated with traits. QTL datasets enable:
- By combining a QTL dataset defined in one Genetic Map to another QTL dataset in another Genetic Map using the same marker type, the location of the QTLs can be compared
- If an Annotation dataset exists against a Genome defining the location of the markers in a given Genetic Map, then QTLs defined in that Genetic Map can be projected to the Genome
- As described above, a QTL defined in either a Genome or Genetic Map can be projected to another Genome or Genetic Map
- Thus, the genes underlying a QTL can be identified by projecting a QTL into a Genome where an Annotation dataset defines the genes in the sequence
- In this way, combining all the above, genes underlying QTLs for a given trait can be found 
"""

class PlantBioRAG:
    def __init__(self):
        self.emb = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        self.graph = Neo4jGraph()
        self.vs = Neo4jVector(embedding=self.emb, url=os.getenv("NEO4J_URI"),
                              username=os.getenv("NEO4J_USERNAME"),
                              password=os.getenv("NEO4J_PASSWORD"),
                              node_label="Chunk", text_node_property="text",
                              embedding_node_property="embedding", 
                              index_name="vector")
    
    # 1. Literature Graph RAG 
    # 1. Vector indexing 
    def _vector_chunks(self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS) -> Dict[str, float]:
        out = {}
        # 1 means the vectors are identical (most similar). 0 means the vectors are diametrically opposite (most dissimilar).
        for doc, score in self.vs.similarity_search_with_score(q, k=k):
            # if score < 0.25:
            #     continue
            cid = doc.metadata.get("chunk_id")
            if cid: out[cid] = max(out.get(cid, 0), score)
        return out    

    def escape_lucene_plain_text(self, q: str) -> str:
        _LUCENE_SPECIAL_CHARS = re.compile(r'([+\-!(){}\[\]^"~*?:\\\/&|])')
        _LUCENE_BOOLEAN_WORDS = re.compile(r'\b(AND|OR|NOT)\b')
        # Treat user input as plain text for a Lucene-backed Neo4j fulltext query: Escapes Lucene query parser metacharacters; Lowercases uppercase Boolean operators so they are searched as words; Normalises whitespace.
        if not q:
            return ""
        q = re.sub(r"\s+", " ", q).strip()
        q = _LUCENE_BOOLEAN_WORDS.sub(lambda m: m.group(1).lower(), q)
        q = _LUCENE_SPECIAL_CHARS.sub(r"\\\1", q)
        return q    

    # Run vector + full-text retrieval concurrently
    def _hybrid_scores_concurrent(self, q: str, vector_fn, fulltext_fn, k: int) -> Dict[str, float]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(vector_fn, q, k)
            fulltext_future = executor.submit(fulltext_fn, q, k)
            vector_scores = vector_future.result()
            fulltext_scores = fulltext_future.result()
        return self._rrf_fusion(vector_scores, fulltext_scores)

    # Run expanded-query searches concurrently
    def _multi_query_hybrid_scores_concurrent(self, expanded_queries: list[str], vector_fn, fulltext_fn, k: int) -> Dict[str, float]:
        all_fused: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(expanded_queries)))) as executor:
            future_to_query = {
                executor.submit(self._hybrid_scores_concurrent, eq, vector_fn, fulltext_fn, k): eq
                for eq in expanded_queries
            }
            for future in as_completed(future_to_query):
                fused = future.result()
                for key, score in fused.items():
                    all_fused[key] = all_fused.get(key, 0.0) + score
        return all_fused

    # One literature search branch for one expanded query
    def _search_literature_one_query(self, expanded_query: str, k: int, max_chars_for_query: int) -> dict:
        fused = self._hybrid_scores_concurrent(expanded_query, self._vector_chunks, self._fulltext_chunks, k)
        seed_cids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]
        seeded_chunks, expanded_chunks, triples = self._expand_hops(seed_cids)
        all_chunks_deduplicated = self._dedupe_chunks(seeded_chunks + expanded_chunks)
        context_chunks, total_characters = [], 0
        for chunk in all_chunks_deduplicated:
            text = chunk.get("text", "")
            if not text:
                continue
            if total_characters + len(text) > max_chars_for_query:
                break
            context_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "source_path": chunk.get("source_path", ""),
                "text": text
            })
            total_characters += len(text)
        return {
            "expanded_query": expanded_query,
            "context_chunks": context_chunks,
            "triples": triples[:MAX_TRIPLES]
        }

    # Run all expanded literature searches concurrently
    def _get_literature_context_concurrent(self, expanded_queries: list[str], k: int, max_context_chars: int) -> str:
        if not expanded_queries:
            return ""
        max_chars_for_query = int(max_context_chars / max(1, len(expanded_queries)))
        results_by_query: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(expanded_queries))) as executor:
            future_to_query = {
                executor.submit(self._search_literature_one_query, eq, k, max_chars_for_query): eq
                for eq in expanded_queries
            }
            for future in as_completed(future_to_query):
                eq = future_to_query[future]
                results_by_query[eq] = future.result()
        # Preserve expanded_queries order in final prompt
        added_chunk_keys = set()
        parts = []
        for eq in expanded_queries:
            result = results_by_query.get(eq)
            if not result:
                continue
            context_chunks = []
            for chunk in result["context_chunks"]:
                cid = chunk.get("chunk_id")
                source_path = chunk.get("source_path", "")
                text = chunk.get("text", "")
                key = cid if cid else (source_path, hash(text))
                if key in added_chunk_keys:
                    continue
                added_chunk_keys.add(key)
                context_chunks.append(f"[Source: {source_path}] {text}\n")
            triple_summ = "\n".join(result["triples"])
            parts.append(f"""
For sub-question
{eq}

### Context Chunks are:
{os.linesep.join(context_chunks)}

### Entity Relationships are:
{triple_summ}
""")
        return "\n".join(parts)

    # 2. Full-text indexing 
    def _fulltext_chunks(self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS) -> Dict[str, float]:
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query("""
            CALL db.index.fulltext.queryNodes('idx_chunk_text', $q) YIELD node, score
            RETURN node.chunk_id AS cid, score ORDER BY score DESC LIMIT $k
        """, params={"q": cleaned_q, "k": k})
        return {r["cid"]: r["score"] for r in res if r.get("cid")}
    
    # Use Reciprocal Rank Fusion (RRF) instead of min-max normalized weights
    def _rrf_fusion(self, vector_scores: Dict[str, float], ft_scores: Dict[str, float], k_penalty=60) -> Dict[str, float]:
        rrf_scores = {}
        for rankings in [vector_scores, ft_scores]:
            # Sort by score descending to get rank
            sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            for rank, (cid, _) in enumerate(sorted_items):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_penalty + rank + 1))
        return rrf_scores
    
    # Deduplicate chunks before reranking and prompt assembly. 
    def _dedupe_chunks(self, chunks: List[dict]) -> List[dict]:
        seen = set()
        deduped = []
        for c in chunks:
            key = (c.get("chunk_id") or (c.get("source_path", ""), c.get("text", "")[:200]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped
    
    # Helper for embedding rerank
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / norm) if norm else 0.0

    # 4. Expand from seed chunks to nodes and chunks. 
    def _expand_hops(self, cids: List[str]):
        # Fetch actual Triples (Node-Rel-Node) 
        query = """
        MATCH (c1:Chunk) WHERE c1.chunk_id IN $cids
        OPTIONAL MATCH (c1)-[:MENTIONS]->(n)
        OPTIONAL MATCH (n)-[r]-(m) WHERE type(r) <> 'MENTIONS' 
        AND n.id IS NOT NULL AND m.id IS NOT NULL  
        OPTIONAL MATCH (m)<-[:MENTIONS]-(c2:Chunk) WHERE NOT c2.chunk_id IN $cids
        RETURN
            collect(DISTINCT c1 {.chunk_id, .source_path, .text}) AS seedchunks,
            collect(DISTINCT c2 {.chunk_id, .source_path, .text}) AS expandedchunks,
            collect(DISTINCT
            CASE
                WHEN r IS NULL THEN NULL
                WHEN c2.source_path IS NULL OR c2.source_path = c1.source_path THEN
                    '[Source: ' + coalesce(c1.source_path, '') + '] ' + n.id + ' -[' + type(r) + ']-> ' + m.id
                ELSE
                    '[Source: ' + coalesce(c1.source_path, '') + '; ' + coalesce(c2.source_path, '') + '] ' + n.id + ' -[' + type(r) + ']-> ' + m.id
            END
        ) AS triples
        """
        res = self.graph.query(query, params={"cids": cids})
        if not res or not res[0]["seedchunks"]:
            return [], [], []
        # Filter out "None -[None]-> None" strings
        triples = [t for t in res[0]["triples"] if t is not None] 
        return res[0]["seedchunks"], res[0]["expandedchunks"], triples
    
    def _llm_invoke(self, prompt: Any) -> str:
        resp = self.llm.invoke(prompt)
        return (getattr(resp, "text", None) or getattr(resp, "content", "") or str(resp)).strip()
    
    # Question analysis and retrieval query expansion 
    def expand_question_and_queries(self, q: str) -> tuple[str, list[str], bool, str, str]:
        prompt = f"""
        You are a professional plant biology RAG expert.
        Given the user question in @@@@, do two tasks:
        1. for RAG retrieval, analyse user question and output step-by-step instructions. 
           - Do not add information not present in the user question.
           - Keep it within 100 words.
        2. Produce retrieval-optimised standalone atomic questions for searching scientific papers, Neo4j graph data, embedded vectors, and keyword indexes.
           - Preserve all exact biological entities from the user question.
           - If a short symbol or name appears, include likely textual variants that may appear in scientific papers.
           - Return maximum 3 questions.
           - Do not force 3 questions if fewer are sufficient.
        3. Determine whether the provided user's question is asking to search for, find, or check accessions in the Australian Grains Genebank (AGG).

        Return output as JSON only, with exactly these keys:
        {{
            "expanded_question": "...",
            "expanded_queries": [
                "...",
                "..."
            ],
            "is_agg_accession_query": true or false,
            "accession_question": "shortened accession search question", 
            "species": "wheat" | "barley" | "oat" | "oats" | "maize" | "corn" | "chickpea" | "chick pea" | "lentil" | "lentils" | "canola" | "rapeseed" | "rye" | "sorghum" | "pea" | "peas" | "faba" | "faba bean" | "mungbean" | "soy" | "soybean" | etc., or empty string if not specified or inferable"
        }}

        Rules:
        1. "is_agg_accession_query" must be true if the user is asking about searching, finding, checking, listing, matching, or identifying accessions in AGG.
        2. "is_agg_accession_query" must be false if the question is not about AGG accession search.
        3. "species": the species if explicitly stated or clearly inferable from context; empty string if cannot be determined..
        4. "accession_question" must be short, contain type information (wheat, barley, chick pea, oat, etc. if available), and focused on "Are these [species] accessions in AGG".
        4. Do NOT include explanations, extra commentary, or metadata.
        5. If "is_agg_accession_query" is false, return an empty string for "accession_question".
        6. Example 1:
        User question: "Is the wheat variety Wyalkatchem available in the Australian Grains Genebank?"
        Output:
        {{
            "is_agg_accession_query": true,
            "accession_question": "Are these [species] accessions in AGG?",
            "species": "wheat"
        }}

        Example 2: 
        User question: "What evidence is there to suggest the wheat variety Wyalkatchem carries the 2NS introgression?"
        Output:
        {{
            "is_agg_accession_query": false,
            "accession_question": "",
            "species": "wheat"
        }}
        @@@@
        {q}
        @@@@
        """
        resp = self._llm_invoke(prompt)
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        expanded_question = str(data.get("expanded_question", q))
        expanded_queries = data.get("expanded_queries", [])
        expanded_question = expanded_question.replace("/", " ").replace(":", " ").replace("\n", " ")
        expanded_queries = [
            str(s).replace("/", " ").replace(":", " ").replace("\n", " ")
            for s in expanded_queries
            if s
        ]
        # Always include original user question first for exact-match retrieval
        expanded_queries.insert(0, q)
        # De-duplicate while preserving order
        expanded_queries = list(dict.fromkeys(expanded_queries))
        is_agg_accession_query = bool(data.get("is_agg_accession_query", False))
        accession_question = str(data.get("accession_question", "")).strip()
        species = str(data.get("species", "")).strip()
        return expanded_question, expanded_queries, is_agg_accession_query, accession_question, species

    def _extract_accessions(self, answer_text: str, species: str) -> List[str]:
        prompt = f"""You are a plant biology expert. From the text below, extract all plant variety names, cultivar names, accession names, and accession numbers (e.g. AGG-prefixed IDs, variety names like "Milan", "Kachu", etc.). Return ONLY a JSON array of strings. If none found, return [].

        Species is {species}. 

        If species is barley and check each string in the output list that starts with AGG, followed by numbers, and does not end with BARL, add BARL to this string in the output list. 
        If a string in the output list ends with BARL, then keep BARL. 
        e.g. AGG495287 should become AGG 495287 BARL. 

        If species is wheat and check each string in the output list that starts with AGG, followed by numbers, and does not end with WHEA, add WHEA to this string in the output list. 
        If a string in the output list ends with WHEA, then keep WHEA. 
        e.g. AGG 41804 should become AGG 41804 WHEA. 

        If species is chickpea and check each string in the output list that starts with AGG, followed by numbers, and does not end with CHIC, add CHIC to this string in the output list. 
        If a string in the output list ends with CHIC, then keep CHIC. 
        e.g. AGG 41804 should become AGG 41804 CHIC. 

        If species is field pea and check each string in the output list that starts with AGG, followed by numbers, and does not end with PEAS, add PEAS to this string in the output list. 
        If a string in the output list ends with PEAS, then keep PEAS. 
        e.g. AGG 41804 should become AGG 41804 PEAS. 

        If species is lentil and check each string in the output list that starts with AGG, followed by numbers, and does not end with LENS, add LENS to this string in the output list. 
        If a string in the output list ends with LENS, then keep LENS. 
        e.g. AGG 41804 should become AGG 41804 LENS. 

        If species is lupin and check each string in the output list that starts with AGG, followed by numbers, and does not end with LUPN, add LUPN to this string in the output list. 
        If a string in the output list ends with LUPN, then keep LUPN. 
        e.g. AGG 41804 should become AGG 41804 LUPN. 

Text:
{answer_text}
"""
        raw = self._llm_invoke(prompt)
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
        try:
            accessions = json.loads(raw)
            return [str(a) for a in accessions if a]
        except Exception:
            # Fallback: extract quoted strings
            return re.findall(r'"([^"]+)"', raw)
        
    # Call the accession API with extracted accession names
    def _call_accession_api(self, question: str, accessions: List[str]) -> Optional[dict]:
        payload = {
            "token": ACCESSION_API_TOKEN,
            "question": question,
            "accessions": accessions
        }
        try:
            response = requests.post(
                ACCESSION_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=ACCESSION_API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Accession API Error: The request timed out.")
        except requests.exceptions.ConnectionError:
            logger.error("Accession API Error: Failed to connect to the server.")
        except requests.exceptions.HTTPError as err:
            logger.error("Accession API HTTP Error: %s", err)
        except Exception as e:
            logger.exception("Accession API unexpected error: %s", e)
        return None
    
    # Use LLM to present accession API response clearly to the user
    def _present_accession_results(self, original_question: str, api_response: dict) -> str:
        prompt = f"""You are a plant biology expert. 
        For user question, clearly and concisely present the AGG accession API results to the user. 
For each accession queried, summarise whether it was found in the AGG and include its accession number(s), name(s), and institute if available. 
Use a structured and readable format in response. 
Barley and wheat Australian Grains Genebank (AGG) Accession_Number typically has this format. eg. AGG 495017 BARL, AGG 495017 WHEA 
AWCC genebank is also part of AGG and has format as AUS+number. eg. AUS123456 
Use entire AGG Accession_Number in the response. 
If api results cannot answer part of user question, eg. Visualise in Pretzel, skip this part and do not answer. 
Never make up answers. 


A user asked: "{original_question}". 

The AGG accession API returned the following results:
{api_response}
"""
        resp = self._llm_invoke(prompt)
        return resp
    

    # 2. Metadata Graph RAG 
    def _vector_chunks_metadata(self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS) -> Dict[str, float]:
        # Vector search in metadata_graph via metadata_vector_index using Gemini embeddings. 
        q_emb = self.emb.embed_query(q)
        res = self.graph.query(
            "CALL db.index.vector.queryNodes('metadata_vector_index', $k, $emb) "
            "YIELD node, score RETURN elementId(node) AS nid, score",
            params={"k": k, "emb": q_emb}
        )
        return {r["nid"]: r["score"] for r in res}

    def _fulltext_chunks_metadata(self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS) -> Dict[str, float]:
        # Fulltext search in metadata_graph via metadata_fulltext_index. 
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            "CALL db.index.fulltext.queryNodes('metadata_fulltext_index', $q) YIELD node, score "
            "RETURN elementId(node) AS nid, score ORDER BY score DESC LIMIT $k",
            params={"q": cleaned_q, "k": k}
        )
        return {r["nid"]: r["score"] for r in res}
    
    def _expand_one_hop(self, nids: List[str]):
        # Expand MetadataGraph seed nodes by 1 hop following any relationship, both directions.
        # Returns nodes with all properties plus chunk_id for dedupe/fetch.
        query = """
        MATCH (n1:MetadataGraph) WHERE elementId(n1) IN $nids
        OPTIONAL MATCH (n1)-[r]-(n2)
        RETURN
            collect(DISTINCT n1 {
                .*,
                chunk_id: elementId(n1),
                source_path: 'Metadata Graph',
                labels: labels(n1)
            }) AS seedchunks,

            collect(DISTINCT n2 {
                .*,
                chunk_id: elementId(n2),
                source_path: 'Metadata Graph',
                labels: labels(n2)
            }) AS expandedchunks,

            collect(DISTINCT
                CASE
                    WHEN r IS NULL OR n2 IS NULL THEN NULL
                    ELSE
                        '[Source: Metadata Graph] ' +
                        coalesce(n1.displayName, n1.shortName, n1.id, n1.projectName,
                                n1.accessionName, n1.curatorName, elementId(n1)) +
                        ' -[' + type(r) + ']- ' +
                        coalesce(n2.displayName, n2.shortName, n2.id, n2.projectName,
                                n2.accessionName, n2.curatorName, elementId(n2))
                END
            ) AS triples
        """
        res = self.graph.query(query, params={"nids": nids})
        if not res or not res[0]["seedchunks"]:
            return [], [], []
        seedchunks = [
            c for c in res[0]["seedchunks"]
            if c and c.get("chunk_id")
        ]
        expandedchunks = [
            c for c in res[0]["expandedchunks"]
            if c and c.get("chunk_id")
        ]
        triples = [
            t for t in res[0]["triples"]
            if t is not None
        ]
        return seedchunks, expandedchunks, triples

    def _search_metadata_hybrid(self, expanded_queries: list[str], max_chars: int = METADATA_MAX_CHARACTERS) -> str:
        # Hybrid search for metadata_graph.
        # Run expanded-query metadata hybrid searches concurrently. 
        # Each query also runs vector + full-text concurrently. 
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            self._vector_chunks_metadata,
            self._fulltext_chunks_metadata,
            QUERY_VECTOR_MAX_CHUNKS
        )
        # Sort by fused score descending
        top_nids = sorted(all_fused, key=lambda x: all_fused[x], reverse=True)[:MAX_METADATA_CHUNKS]
        if not top_nids:
            return ""
        seeded_chunks, expanded_chunks, triples = self._expand_one_hop(top_nids)
        all_chunks_deduplicated = self._dedupe_chunks(seeded_chunks + expanded_chunks)
        deduped_nids = [
            c.get("chunk_id")
            for c in all_chunks_deduplicated
            if c.get("chunk_id")
        ]
        fetch_nids = deduped_nids or top_nids
        res = self.graph.query(
            """
            UNWIND range(0, size($nids) - 1) AS i
            WITH $nids[i] AS nid, i
            MATCH (n:MetadataGraph)
            WHERE elementId(n) = nid
            RETURN i, n {.*} AS props
            ORDER BY i
            """,
            params={"nids": fetch_nids}
        )
        # Limit by max_chars. 
        context_parts, total_chars = [], 0        
        for t in triples[:MAX_TRIPLES]:
            text = json.dumps({"relationship": t}, ensure_ascii=False)
            if total_chars + len(text) > max_chars:
                return "\n".join(context_parts)
            context_parts.append(text)
            total_chars += len(text)
        for r in res:
            props = {k: v for k, v in r["props"].items()
                     if k != "embedding" and v is not None}
            text = json.dumps(props)
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(text)
            total_chars += len(text)
        return "\n".join(context_parts)
    
    def _get_metadata_context(self, query: str, expanded_queries: list[str]) -> str:
        context_parts = []
        result = self._search_metadata_hybrid(expanded_queries)
        if result:
            context_parts.append(f"### Metadata Graph (Hybrid Search):\n{result}")
        return "\n\n".join(context_parts)

    def _vector_chunks_pretzel(self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS) -> Dict[str, float]:
        # Vector search in pretzel_graph using Gemini embeddings. 
        q_emb = self.emb.embed_query(q)
        res = self.graph.query(
            "CALL db.index.vector.queryNodes('pretzel_functions_vector', $k, $emb) "
            "YIELD node, score RETURN elementId(node) AS nid, score",
            params={"k": k, "emb": q_emb}
        )
        return {r["nid"]: r["score"] for r in res}

    def _fulltext_chunks_pretzel(self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS) -> Dict[str, float]:
        # Fulltext search in pretzel_graph. 
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            "CALL db.index.fulltext.queryNodes('idx_pretzel_function_text', $q) YIELD node, score "
            "RETURN elementId(node) AS nid, score ORDER BY score DESC LIMIT $k",
            params={"q": cleaned_q, "k": k}
        )
        return {r["nid"]: r["score"] for r in res}
    
    def _get_pretzel_context(self, expanded_queries: list[str], max_chars: int = METADATA_MAX_CHARACTERS) -> str:
        # Hybrid search for pretzel_graph.
        # Run expanded-query Pretzel hybrid searches concurrently
        # Each query also runs vector + full-text concurrently
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            self._vector_chunks_pretzel,
            self._fulltext_chunks_pretzel,
            QUERY_VECTOR_MAX_CHUNKS
        )
        # Sort by fused score descending
        top_nids = sorted(all_fused, key=lambda x: all_fused[x], reverse=True)[:50]
        if not top_nids:
            return ""
        # Fetch full node properties 
        res = self.graph.query(
            """UNWIND $nids AS nid
            MATCH (p:PretzelFunction) WHERE elementId(p) = nid 
            RETURN p {.*} AS props""",
            params={"nids": top_nids}
        )
        # Limit by total_characters = 10000
        context_parts, total_chars = [], 0
        exclude_keys = {"embedding", "id", "chunk_id"}
        for r in res:
            props = {k: v for k, v in r["props"].items()
                     if k not in exclude_keys and v is not None}
            text = json.dumps(props)
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(text)
            total_chars += len(text)
        return "\n".join(context_parts)
    

    # Main query 
    def query(self, q: str, k: int = QUERY_MAX_CHUNKS, max_context_chars: int = MAX_CHARACTERS) -> tuple[str, str, dict]:
        logger.info(f"Start query.")
        start_time = time.perf_counter()        
        try:
            expanded_question, expanded_queries, is_agg_accession_query, accession_question, species = self.expand_question_and_queries(q)
            logger.info(f"expanded_question: {expanded_question}")
            logger.info(f"expanded_queries: {expanded_queries}")
            logger.info(f"is_agg_accession_query: {is_agg_accession_query}")
            logger.info(f"accession_question: {accession_question}")
            logger.info(f"species: {species}")
            logger.info(f"Start query.")
        except Exception as e:
            logger.warning("Question and query expansion failed: %s", e)
            expanded_question = q
            expanded_queries = [q]
            is_agg_accession_query = False
            accession_question = ""
            species = ""
        end_time = time.perf_counter()
        logger.info(f"1. Analysis of question: {end_time - start_time:0.1f} sec.")

        prompt = global_instruction_and_information + f"""\n\n\nYou are a plant biology RAG expert. 
        read the provided context. 
        read user question in @@@@. 

        if a piece of provided context is contradictory or irrelevant to the user question, ignore it. 
        if a piece of provided context directly supports answer to user question, keep it. 

        concisely and directly answer user question in @@@@ based ONLY on the provided Context Chunks, Entity Relationships (eg. [Source: ...md] Marker-Trait Associations -[MARKER]-> Significant Markers), and Context from Metadata Graph, and Context from Pretzel documentation. 

        Cite sources after facts by appending [Source: ]. 
        If file name is like Surname_Year.pdf.md, use Surname Year only and do not include pdf.md. eg. [Source: Wallwork 2022] 
        If file name is like title.pdf.md, use complete file name ending with .pdf.md]. eg. [Source: An_island_of_receptor-like_genes_at_the_Rrs13_locu.pdf.md] 
        If source is from an Entity Relationship, use relevant [Source: Surname Year] or [Source: File Name.pdf.md]. Do not cite [Source: Entity Relationship]. Never cite [Source: Entity Relationship].
        If the source is Metadata Graph, use [Source: Metadata Graph]. 
        Double check citing source. 
        If source is Pretzel documentation, cite [Source: Pretzel Documentation]. 

        Do not confuse Entity Relationships with Metadata Graph. 
        Do not cite [Source: Background Information] or instructions. Never cite [Source: Background Information]. 

        Do not make up content in answer. 
        If unsure or evidence is missing, say "No information available". 

        Do not infer beyond the retrieved context. 
        Prefer concise and direct answers. 
        
        If useful, structure answer as:
        1. Answer
        2. Evidence
        3. Limitations / missing information

        Never assume genomic coordinates, chromosome assignments, or marker locations are transferable between assemblies. 
        Before reporting that a marker is located in the requested assembly, verify that the marker is explicitly annotated in that exact assembly in the retrieved context. 
        Chromosome-level evidence from literature, trait associations, or another assembly does not prove the marker has a position in the requested assembly. 
        If the marker is annotated only in another assembly, label that assembly as the source assembly and say the requested assembly coordinate is not available in the retrieved context. 
        """

        # Run literature, metadata, and Pretzel context retrieval concurrently. 
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            literature_future = executor.submit(self._get_literature_context_concurrent, expanded_queries, k, max_context_chars)
            metadata_future = executor.submit(self._get_metadata_context, q, expanded_queries)
            pretzel_future = None
            if "pretzel" in q.lower():
                pretzel_future = executor.submit(self._get_pretzel_context, expanded_queries)
            literature_context = literature_future.result()
            metadata_context = metadata_future.result()
            pretzel_context = pretzel_future.result() if pretzel_future else ""
        if literature_context:
            prompt += literature_context
        if metadata_context:
            prompt += f"""\n\n\n
### [Source: Metadata Graph]:
{metadata_context}"""
        if pretzel_context:
            prompt += f"""\n\n\n
### [Source: Pretzel Documentation]:
{pretzel_context}"""
        end_time = time.perf_counter()
        logger.info(
            "2. Concurrent retrieval: literature + metadata + Pretzel context: %.1f sec.",
            end_time - start_time
        )
        
        # 3: Answer user question based on all retrived context. 
        prompt += f"""



            Analysis of user question: 
            {expanded_question}

            User Question:
            @@@@
            {q}
            @@@@
            Answer:"""
        
        start_time = time.perf_counter()
        resp = self.llm.invoke(prompt)
        answer = (getattr(resp, "text", None) or getattr(resp, "content", "") or str(resp)).strip()
        usage_metadata = getattr(resp, "usage_metadata", {}) or {}
        end_time = time.perf_counter()
        logger.info(f"4. Call LLM to answer: {end_time - start_time:0.1f} sec.")

        if is_agg_accession_query and not species:
            species = input("Please specify the species (e.g. wheat, barley, oat, chickpea): ").strip()
            accession_question = f"Are these {species} accessions in AGG?"
            logger.info("Updated accession_question: %s", accession_question)
        if is_agg_accession_query:
            logger.info("[AGG Accession Query Detected] Extracting accessions from RAG answer...")
            start_time = time.perf_counter()
            accessions = self._extract_accessions(f"User question:\n{q}\n\nRAG answer:\n{answer}", species)
            end_time = time.perf_counter()
            logger.info(f"6. Extract accessions: {end_time - start_time:0.1f} sec.")
            logger.info("[Extracted Accessions]: %s", accessions)
            if accessions:
                start_time = time.perf_counter()
                api_response = self._call_accession_api(accession_question, accessions)
                end_time = time.perf_counter()
                logger.info(f"7. Call accession API: {end_time - start_time:0.1f} sec.")
                logger.info(f"api_response: {api_response}")
                if api_response:
                    logger.info("[Accession API response received] Presenting to user via LLM...")
                    logger.debug("Accession API response: %s", api_response)
                    start_time = time.perf_counter()
                    accession_summary = self._present_accession_results(q, api_response)
                    end_time = time.perf_counter()
                    logger.info(f"8. Summarise and present accession results: {end_time - start_time:0.1f} sec.")
                    # Append accession lookup results to the main answer
                    answer = answer + "\n\n---\n\n**Australian Grains Genebank (AGG) Accession Lookup:**\n" + accession_summary
                else:
                    answer += "\n\n[AGG accession lookup failed — API unavailable or returned no data.]"
            else:
                answer += "\n\n[No accession names could be extracted from the RAG answer to query the AGG API.]"
        return expanded_question, answer, usage_metadata

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plant Biology RAG Pipeline")
    parser.add_argument("query", type=str, help="The question you want to ask the RAG pipeline")
    args = parser.parse_args()

    rag = PlantBioRAG()

    expanded_question, answer, tokens = rag.query(args.query)
    logger.info("Final Answer:\n%s", answer)
    logger.info("Token Usage: %s", str(tokens))

if __name__ == "__main__":
    main()