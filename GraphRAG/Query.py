# RAG pipeline to query plant biology papers using Neo4j, Gemini, LangChain. 
# To run in docker: docker run -it --rm --network host -e GOOGLE_API_KEY="Key ..." -e NEO4J_URI="" -e NEO4J_USERNAME="" -e NEO4J_PASSWORD="" graph-rag "User question ..."

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
ACCESSION_API_TOKEN = ""
ACCESSION_API_TIMEOUT = 120

METADATA_MAX_CHARACTERS = 300000
MIN_CYPHER_CHARACTERS = 6000

break_down_question_instruction = """
        You are an expert in language and understanding semantic meaning for a plant biology organisation.
        You read input from person and your job is to expand the user question into stand alone atomic questions. 
        If there will be more than 3 stand alone atomic questions, condense them and respond with maximum 3 stand alone atomic questions. 
        Work on the input question case by case. 
        There is no need to break down into 3 stand alone atomic questions every time. 
        You must return JSON using the template below.
        Example:
        Input: How do I find genes near a marker of interest I have DNA sequence for?
        Output:
        {{
            "questions": [
                "How do I map a DNA sequence to a specific genomic location?",
                "How do I search for genes located near a specific genomic marker?"
            ]
        }}
    """

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

generate_cypher_query_prompt_template_string = global_instruction_and_information + """
Instructions: 
For each user question in &&&&, generate a Cypher statement to query a graph database that can run on Neo4j Desktop on Windows 10, to get the as much data as possible for LLM to answer a user question. 
Do not retrieve embedding. 

Graph Database Schema:
Use only the provided node labels and properties, and relationships in the schema in ````.
Do not use any other nodes or relationships that are not provided in the schema. 
````
{schema}
````

Terminology mapping:
This section is helpful to map terminology between the user question and the graph database schema. 
Read ontology definitions and comments in %%%%. 
%%%%
{terminology}
%%%%

User questions:  
&&&&
{user_questions}
&&&&

Format instructions:
Do not include any explanations or apologies in your responses. 
Do not include any questions that might ask anything else than for you to construct a Cypher statement. 
Respond in following json format: 
[{{
"sub_question_id": "1", 
"sub_question": "sub question", 
"generated_cypher_query": "Generated cypher query. Do not generate unnecessarily complicated queries. Strictly do not include new line \\n or any other irrelevant characters. Before responding, check the generated cypher query and make sure it is valid."
}}, 
...
]
"""

schema_string = """
'Node properties:\nDataset {id: STRING, embedding: LIST, displayName: STRING, shortName: STRING, publication: STRING, crop: STRING, species: STRING, type: STRING, dataSource: STRING, tags: STRING, panbarlexName: STRING, comments: STRING, categories: STRING, alignmentType: STRING, genomeId: STRING, markerType: STRING, platform: STRING, licensingOfOriginalData: STRING, parentNames: STRING, populationType: STRING, definedIn: STRING}\nCurator {curatorName: STRING, embedding: LIST, contact: STRING}\nDatasetAccession {accessionName: STRING, embedding: LIST, pedigree: STRING, growthHabit: STRING, origin: STRING, features: STRING, marketClass: STRING, yearOfRelease: STRING}\nProject {projectId: STRING, embedding: LIST, projectName: STRING, projectDescription: STRING}\nMetadataGraph {curatorName: STRING, embedding: LIST, contact: STRING, accessionName: STRING, pedigree: STRING, growthHabit: STRING, origin: STRING, features: STRING, marketClass: STRING, yearOfRelease: STRING, projectId: STRING, projectName: STRING, projectDescription: STRING, id: STRING, displayName: STRING, shortName: STRING, publication: STRING, crop: STRING, species: STRING, type: STRING, dataSource: STRING, tags: STRING, panbarlexName: STRING, comments: STRING, categories: STRING, alignmentType: STRING, genomeId: STRING, markerType: STRING, platform: STRING, licensingOfOriginalData: STRING, parentNames: STRING, populationType: STRING, definedIn: STRING}\nGenomeDataset {id: STRING, embedding: LIST, displayName: STRING, shortName: STRING, publication: STRING, crop: STRING, species: STRING, type: STRING, dataSource: STRING, tags: STRING, panbarlexName: STRING, comments: STRING, categories: STRING}\nAlignmentDataset {id: STRING, embedding: LIST, displayName: STRING, publication: STRING, categories: STRING, crop: STRING, species: STRING, type: STRING, dataSource: STRING, alignmentType: STRING, genomeId: STRING, shortName: STRING, markerType: STRING, comments: STRING}\nVCFDataset {id: STRING, embedding: LIST, displayName: STRING, shortName: STRING, comments: STRING, crop: STRING, species: STRING, type: STRING, dataSource: STRING, tags: STRING, markerType: STRING, platform: STRING, categories: STRING, licensingOfOriginalData: STRING}\nGeneticMapdataset {id: STRING, embedding: LIST, displayName: STRING, publication: STRING, categories: STRING, crop: STRING, species: STRING, type: STRING, markerType: STRING, parentNames: STRING, shortName: STRING, comments: STRING, populationType: STRING}\nQTL {id: STRING, embedding: LIST, displayName: STRING, shortName: STRING, comments: STRING, publication: STRING, categories: STRING, crop: STRING, species: STRING, type: STRING, tags: STRING, definedIn: STRING}\n
Relationship properties:\n\nThe relationships:\n(:Dataset)-[:hasCurator]->(:Curator)\n(:Dataset)-[:hasCurator]->(:MetadataGraph)\n(:Dataset)-[:hasDatasetAccession]->(:DatasetAccession)\n(:Dataset)-[:hasDatasetAccession]->(:MetadataGraph)\n(:Dataset)-[:ALIGNED_TO_GENOME]->(:Dataset)\n(:Dataset)-[:ALIGNED_TO_GENOME]->(:MetadataGraph)\n(:Dataset)-[:ALIGNED_TO_GENOME]->(:GenomeDataset)\n(:Dataset)-[:hasProject]->(:Project)\n(:Dataset)-[:hasProject]->(:MetadataGraph)\n(:Dataset)-[:ALIGNED_TO]->(:Dataset)\n(:Dataset)-[:ALIGNED_TO]->(:MetadataGraph)\n(:Dataset)-[:ALIGNED_TO]->(:AlignmentDataset)\n(:Dataset)-[:DEFINED_IN]->(:Dataset)\n(:Dataset)-[:DEFINED_IN]->(:MetadataGraph)\n(:Dataset)-[:DEFINED_IN]->(:GeneticMapdataset)\n(:Dataset)-[:DEFINED_IN]->(:GenomeDataset)\n(:MetadataGraph)-[:hasCurator]->(:Curator)\n(:MetadataGraph)-[:hasCurator]->(:MetadataGraph)\n(:MetadataGraph)-[:hasDatasetAccession]->(:DatasetAccession)\n(:MetadataGraph)-[:hasDatasetAccession]->(:MetadataGraph)\n(:MetadataGraph)-[:ALIGNED_TO_GENOME]->(:Dataset)\n(:MetadataGraph)-[:ALIGNED_TO_GENOME]->(:MetadataGraph)\n(:MetadataGraph)-[:ALIGNED_TO_GENOME]->(:GenomeDataset)\n(:MetadataGraph)-[:hasProject]->(:Project)\n(:MetadataGraph)-[:hasProject]->(:MetadataGraph)\n(:MetadataGraph)-[:ALIGNED_TO]->(:Dataset)\n(:MetadataGraph)-[:ALIGNED_TO]->(:MetadataGraph)\n(:MetadataGraph)-[:ALIGNED_TO]->(:AlignmentDataset)\n(:MetadataGraph)-[:DEFINED_IN]->(:Dataset)\n(:MetadataGraph)-[:DEFINED_IN]->(:MetadataGraph)\n(:MetadataGraph)-[:DEFINED_IN]->(:GeneticMapdataset)\n(:MetadataGraph)-[:DEFINED_IN]->(:GenomeDataset)\n(:GenomeDataset)-[:hasCurator]->(:Curator)\n(:GenomeDataset)-[:hasCurator]->(:MetadataGraph)\n(:GenomeDataset)-[:hasDatasetAccession]->(:DatasetAccession)\n(:GenomeDataset)-[:hasDatasetAccession]->(:MetadataGraph)\n(:AlignmentDataset)-[:hasCurator]->(:Curator)\n(:AlignmentDataset)-[:hasCurator]->(:MetadataGraph)\n(:AlignmentDataset)-[:ALIGNED_TO_GENOME]->(:Dataset)\n(:AlignmentDataset)-[:ALIGNED_TO_GENOME]->(:MetadataGraph)\n(:AlignmentDataset)-[:ALIGNED_TO_GENOME]->(:GenomeDataset)\n(:VCFDataset)-[:hasCurator]->(:Curator)\n(:VCFDataset)-[:hasCurator]->(:MetadataGraph)\n(:VCFDataset)-[:hasProject]->(:Project)\n(:VCFDataset)-[:hasProject]->(:MetadataGraph)\n(:GeneticMapdataset)-[:hasCurator]->(:Curator)\n(:GeneticMapdataset)-[:hasCurator]->(:MetadataGraph)\n(:GeneticMapdataset)-[:ALIGNED_TO]->(:Dataset)\n(:GeneticMapdataset)-[:ALIGNED_TO]->(:MetadataGraph)\n(:GeneticMapdataset)-[:ALIGNED_TO]->(:AlignmentDataset)\n(:QTL)-[:hasCurator]->(:Curator)\n(:QTL)-[:hasCurator]->(:MetadataGraph)\n(:QTL)-[:DEFINED_IN]->(:Dataset)\n(:QTL)-[:DEFINED_IN]->(:MetadataGraph)\n(:QTL)-[:DEFINED_IN]->(:GeneticMapdataset)\n(:QTL)-[:DEFINED_IN]->(:GenomeDataset)'
"""

terminology_string = """
@prefix pretzel: <https://agg.plantinformatics.io/ont#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .

# Ontology Metadata
pretzel: a owl:Ontology ;
    dcterms:title "Pretzel Dataset Metadata Ontology"@en ;
    dcterms:description "Ontology for Pretzel platform dataset metadata, covering genomes, alignments, genetic maps, VCF data, and associated context."@en ;
    dcterms:created "2025-10-28T16:23:59.793518"^^xsd:dateTime ;
    dcterms:creator "Pretzel Project"@en ;
    owl:versionInfo "20251028" .

##############################################
# Classes
##############################################

pretzel:Dataset a owl:Class ;
    rdfs:label "Dataset"@en ;
    rdfs:comment "A dataset in the Pretzel platform"@en .

pretzel:AlignmentDataset a owl:Class ;
    rdfs:subClassOf pretzel:Dataset ;
    rdfs:label "Alignment Dataset"@en ;
    rdfs:comment "Dataset of type Alignment"@en .

pretzel:GeneticMapdataset a owl:Class ;
    rdfs:subClassOf pretzel:Dataset ;
    rdfs:label "Genetic Map Dataset"@en ;
    rdfs:comment "Dataset of type Genetic Map"@en .

pretzel:GenomeDataset a owl:Class ;
    rdfs:subClassOf pretzel:Dataset ;
    rdfs:label "Genome Dataset"@en ;
    rdfs:comment "Dataset of type Genome"@en .

pretzel:VCFDataset a owl:Class ;
    rdfs:subClassOf pretzel:Dataset ;
    rdfs:label "VCF Dataset"@en ;
    rdfs:comment "Dataset of type VCF"@en .

pretzel:Curator a owl:Class ;
    rdfs:label "Curator"@en ;
    rdfs:comment "Context information: Curator"@en .

pretzel:DatasetAccession a owl:Class ;
    rdfs:label "DatasetAccession"@en ;
    rdfs:comment "Context information: DatasetAccession"@en .

pretzel:Project a owl:Class ;
    rdfs:label "Project"@en ;
    rdfs:comment "Context information: Project"@en .

##############################################
# Object Properties
##############################################

pretzel:hasCurator a owl:ObjectProperty ;
    rdfs:label "has curator"@en ;
    rdfs:domain pretzel:Dataset ;
    rdfs:range pretzel:Curator ;
    rdfs:comment "Links a dataset to its curator information"@en .

pretzel:hasDatasetAccession a owl:ObjectProperty ;
    rdfs:label "has datasetaccession"@en ;
    rdfs:domain pretzel:Dataset ;
    rdfs:range pretzel:DatasetAccession ;
    rdfs:comment "Links a dataset to its datasetaccession information"@en .

pretzel:hasProject a owl:ObjectProperty ;
    rdfs:label "has project"@en ;
    rdfs:domain pretzel:Dataset ;
    rdfs:range pretzel:Project ;
    rdfs:comment "Links a dataset to its project information"@en .
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


    # 2. Full-text indexing 
    def _fulltext_chunks(self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS) -> Dict[str, float]:
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query("""
            CALL db.index.fulltext.queryNodes('idx_chunk_text', $q) YIELD node, score
            RETURN node.chunk_id AS cid, score ORDER BY score DESC LIMIT $k
        """, params={"q": q, "k": k})
        return {r["cid"]: r["score"] for r in res if r.get("cid")}
    
    # 3. Use Reciprocal Rank Fusion (RRF) instead of min-max normalized weights
    def _rrf_fusion(self, vector_scores: Dict[str, float], ft_scores: Dict[str, float], k_penalty=60) -> Dict[str, float]:
        rrf_scores = {}
        for rankings in [vector_scores, ft_scores]:
            # Sort by score descending to get rank
            sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            for rank, (cid, _) in enumerate(sorted_items):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_penalty + rank + 1))
        return rrf_scores
    
    # 4. Deduplicate chunks before reranking and prompt assembly. 
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
    
    # Rerank chunks using embedding relevance + fused RRF score
    def _rerank_chunks_with_rrf_boost(self, query: str, chunks: List[dict], fused_scores: Dict[str, float], top_n: int = 8) -> List[dict]:
        if not chunks:
            return []
        query_emb = self.emb.embed_query(query)
        texts = [(c.get("text") or "")[:RERANK_MAX_TEXT_CHARS] for c in chunks]
        chunk_embs = self.emb.embed_documents(texts)
        max_fused = max(fused_scores.values()) if fused_scores else 1.0
        scored = []
        for chunk, emb in zip(chunks, chunk_embs):
            cid = chunk.get("chunk_id")
            rerank_score = self._cosine_similarity(query_emb, emb)
            rrf_score = fused_scores.get(cid, 0.0) / max_fused if max_fused else 0.0
            final_score = 0.7 * rerank_score + 0.3 * rrf_score
            scored.append((chunk, final_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_n]]

    # 5. Expand from seed chunks to nodes and chunks. 
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
    
    def expand_question(self, q: str) -> str:
        prompt_expand_user_question = f"""
        You are a professional plant biology RAG expert. Given user question in @@@@, expand user question to achieve better RAG. 
        If the expanded question is within 100 words, return the expanded question. 
        If the expanded question is longer than 100 words, condense the expanded question to 100 words and return. 
        @@@@
        {q}
        @@@@
        """
        response_text = self._llm_invoke(prompt_expand_user_question)
        return response_text.replace("/", " ").replace(":", " ").replace("\n", " ")
    
    def expand_query(self, q: str) -> list[str]:
        prompt_expand_user_query = f"""
        You are an expert in language and understanding semantic meaning for an organisation researching plant biology.
1. You read input from user in @@@@ and your job is to expand the user question into retrieval-optimised and stand alone atomic questions in order to search in scientific papers neo4j graph database by embedded vectors and keywords. 
        2. Preserve all exact biological entities from the user question. 
        3. If a question contains a short symbol or name, include likely textual variants that may appear in scientific papers. 
        4. Each question should be suitable for direct use in retrieval over scientific text. 
        5. If there will be more than 3 stand alone atomic questions, condense them and respond with maximum 3 stand alone atomic questions. 
        6. Work on the input question case by case. There is no need to break down into 3 stand alone atomic questions every time. 
    7. You must return JSON using the template below.
    Example:
    Input: How do I find genes near a marker of interest I have DNA sequence for?
    Output:
    {{
        "questions": [
            "...",
            "..."
        ]
    }}

        @@@@
        {q}
        @@@@
"""
        resp = self._llm_invoke(prompt_expand_user_query)
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return [
            s.replace("/", " ").replace(":", " ").replace("\n", " ") 
            for s in data.get("questions", [])
        ]    
    
    # Detect if the user question is asking about AGG accessions
    def _is_agg_accession_query(self, q: str) -> Tuple[bool, str, str]:
        prompt = f"""
        You are a plant biology expert.
        Your task is to determine whether the provided user's question is asking to search for, find, or check accessions in the Australian Grains Genebank (AGG).
        Return output as JSON only, with exactly these keys:
        {{
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

        User question: 
        {q}""".strip()
        text = self._llm_invoke(prompt)
        # Try strict JSON parse first
        try:
            data = json.loads(text)
            is_query = bool(data.get("is_agg_accession_query", False))
            accession_question = str(data.get("accession_question", "")).strip()
            species = str(data.get("species", "")).strip()
            return is_query, accession_question, species
        except Exception:
            pass
        # Fallback: extract JSON object if model adds extra text
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                is_query = bool(data.get("is_agg_accession_query", False))
                accession_question = str(data.get("accession_question", "")).strip()
                species = str(data.get("species", "")).strip()
                return is_query, accession_question, species
        except Exception:
            pass
        # Final fallback
        upper_text = text.upper()
        is_query = '"IS_AGG_ACCESSION_QUERY": TRUE' in upper_text or '"IS_AGG_ACCESSION_QUERY":TRUE' in upper_text
        return is_query, "", ""

    
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
        all_fused: Dict[str, float] = {}
        for eq in expanded_queries:
            v = self._vector_chunks_metadata(eq)
            f = self._fulltext_chunks_metadata(eq)
            fused = self._rrf_fusion(v, f)
            for nid, score in fused.items():
                all_fused[nid] = all_fused.get(nid, 0.0) + score

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


        # Fetch full node properties 
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

    def break_down_question(self, question: str) -> list[str]:
        prompt = [
            {"role": "system", "content": break_down_question_instruction},
            {"role": "user", "content": f"The user question to rewrite: '{question}'"},
        ]
        text = self._llm_invoke(prompt)
        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        if text.startswith("```"):
            text = text.removeprefix("```").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()
        try:
            return json.loads(text).get("questions", [])
        except json.JSONDecodeError:
            logger.warning("break_down_question: failed to decode JSON response")
        return []
    
    def format_sub_questions(self, sub_questions: list[str]) -> str:
        sub_questions_string = ""
        for index, sub_question in enumerate(sub_questions):
            sub_questions_string += f"{index+1}: {sub_question}\n"
        return sub_questions_string.rstrip('\n')
    
    generate_cypher_query_prompt_template = PromptTemplate.from_template(generate_cypher_query_prompt_template_string)

    def generate_cypher_queries(self, indexed_sub_questions: str) -> list[dict[str, Any]]:
        generate_cypher_query_full_prompt = self.generate_cypher_query_prompt_template.format(
            user_questions=indexed_sub_questions, schema=schema_string, terminology=terminology_string)
        prompt = [{"role": "user", "content": generate_cypher_query_full_prompt}]
        response = self._llm_invoke(prompt)
        try:
            output = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(output)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            logger.warning("generate_cypher_queries: failed to decode JSON response")
        return []
    
    def drop_embedding(self, obj: Any) -> Any:
        if isinstance(obj, Node):
            d = dict(obj); d.pop("embedding", None); return d
        if isinstance(obj, Relationship):
            d = dict(obj); d.pop("embedding", None); return d
        if isinstance(obj, Path):
            return {"nodes": [self.drop_embedding(n) for n in obj.nodes],
                    "relationships": [self.drop_embedding(r) for r in obj.relationships]}
        if isinstance(obj, dict):
            return {k: self.drop_embedding(v) for k, v in obj.items() if k != "embedding"}
        if isinstance(obj, (list, tuple, set)):
            return [self.drop_embedding(v) for v in obj]
        return obj

    def run_cypher_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        rows: list[dict[str, Any]] = []
        result = self.graph.query(query, params=parameters or {})
        for record in result:
            rows.append(self.drop_embedding(record))
        return rows
    
    
    def validate_readonly_cypher(self, query: str) -> bool:
        forbidden_patterns = [
                r"\bCREATE\b",
                r"\bMERGE\b",
                r"\bDELETE\b",
                r"\bDETACH\b",
                r"\bSET\b",
                r"\bREMOVE\b",
                r"\bDROP\b",
                r"\bLOAD\s+CSV\b",
                r"\bCALL\s+DBMS\b",
                r"\bAPOC\b",
                r"\bCREATE\s+INDEX\b",
                r"\bCREATE\s+CONSTRAINT\b",
            ]
        upper = re.sub(r"\s+", " ", query.upper())
        return not any(re.search(pattern, upper) for pattern in forbidden_patterns)
    
    
    def _ensure_limit(self, query: str, limit: int = 50) -> str:
        if re.search(r"\bLIMIT\b", query, re.IGNORECASE):
            return query
        return query.rstrip(";") + f" LIMIT {limit}"


    def get_data_from_cypher_query(self, cypher_query: str) -> list[dict[str, Any]]:
        cypher_query = cypher_query.strip()
        if not cypher_query:
            return []
        if not self.validate_readonly_cypher(cypher_query):
            logger.warning("Blocked unsafe Cypher: %s", cypher_query)
            return []
        cypher_query = self._ensure_limit(cypher_query)
        try:
            return self.run_cypher_query(cypher_query)
        except Exception as e:
            logger.warning("Cypher failed: %s", e)
            return []


    def tool_generate_and_run_cypher_query(self, user_question: str) -> tuple[list[dict[str, Any]], bool]:
        # Run LLM-generated Cypher on metadata_graph. 
        sub_questions = self.break_down_question(user_question) or [user_question]
        sub_questions.append(user_question)
        sub_questions_string = self.format_sub_questions(sub_questions)
        generated_cypher_queries = self.generate_cypher_queries(sub_questions_string)
        results = []
        cypher_data_has_data = False
        for obj in generated_cypher_queries:
            data = self.get_data_from_cypher_query(obj.get("generated_cypher_query", ""))
            if not cypher_data_has_data and data:
                cypher_data_has_data = True
            results.append({
                "sub_question_id": obj.get("sub_question_id"),
                "sub_question": obj.get("sub_question"),
                "generated_cypher_query": obj.get("generated_cypher_query"),
                "data_from_generated_cypher_query": data
            })
        return results, cypher_data_has_data

    def _search_metadata_cypher(self, q: str) -> str:
        # Run LLM-generated Cypher against metadata_graph. 
        data_retrieved, has_data = self.tool_generate_and_run_cypher_query(q)
        if not has_data:
            return ""
        parts = []
        for d in data_retrieved:
            parts.append(
                f"Sub-question: {d['sub_question']}\n"
                f"Cypher: {d['generated_cypher_query']}\n"
                f"Data: {json.dumps(d['data_from_generated_cypher_query'])}"
            )
        return "\n\n".join(parts)
    
    def tool_choice(self, functions: list[dict], instruction: str):
        # Use LangChain ChatGoogleGenerativeAI to choose tool calls. Returns a list of tool call dicts like: [{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}]
        llm_with_tools = self.llm.bind_tools(functions)
        response = llm_with_tools.invoke(instruction)
        return getattr(response, "tool_calls", []) or []

    def _get_metadata_context(self, query: str, expanded_queries: list[str]) -> str:
        # LLM tool selection decides whether to use cypher or hybrid (or both) for metadata_graph, then runs the selected search(es). 
        tools = [
            {
                "name": "generate_and_run_cypher_query",
                "description": "for data retrieval questions, generate cypher query, run cypher query, and retrieve data",
                "parameters": {
                    "type": "object",
                    "properties": {"user_question": {"type": "string"}},
                    "required": ["user_question"]
                }
            },
            {
                "name": "hybrid_searching_keyword_and_vector",
                "description": "for questions best answered by text analysis, run hybrid keyword + vector search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 40}
                    },
                    "required": ["query"]
                }
            }
        ]
        instruction = (
            "Based on a user question, choose 1 or 2 tools for metadata_graph search. "
            "If 2 tools are applicable choose both.\nUser question: " + query
        )
        tool_calls = self.tool_choice(tools, instruction)
        context_parts = []
        hybrid_already_run = False

        # Iterate all parts to handle single or dual tool selection
        for tool_call in tool_calls:
            fn = tool_call.get("name")
            if fn == "generate_and_run_cypher_query":
                result = self._search_metadata_cypher(query)
                if result:
                    context_parts.append(f"### Metadata Graph (Cypher):\n{result}")
                total_length = sum(len(s) for s in context_parts)
                if total_length < MIN_CYPHER_CHARACTERS and not hybrid_already_run:
                    result = self._search_metadata_hybrid(expanded_queries)
                    if result:
                        context_parts.append(f"### Metadata Graph (Hybrid Search):\n{result}")
                        hybrid_already_run = True
                        break
            elif fn == "hybrid_searching_keyword_and_vector":
                if not hybrid_already_run:
                    result = self._search_metadata_hybrid(expanded_queries)
                    if result:
                        context_parts.append(f"### Metadata Graph (Hybrid Search):\n{result}")
                        hybrid_already_run = True
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
        # Accumulate RRF scores across all expanded queries
        all_fused: Dict[str, float] = {}
        for eq in expanded_queries:
            v = self._vector_chunks_pretzel(eq)
            f = self._fulltext_chunks_pretzel(eq)
            fused = self._rrf_fusion(v, f)
            for nid, score in fused.items():
                all_fused[nid] = all_fused.get(nid, 0.0) + score

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
        # 1. Literature graph search  
        try:
            expanded_question = self.expand_question(q)
            logger.info(f"expanded_question: {expanded_question}")
        except Exception as e:
            logger.warning("Question expansion failed: %s", e)
            expanded_question = q
        try:
            expanded_queries = self.expand_query(expanded_question) # (q) or [q]
            expanded_queries.insert(0, q)
            logger.info(f"expanded_queries: {expanded_queries}")
            logger.info(f"Start query.")
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            expanded_queries = [q]
        end_time = time.perf_counter()
        logger.info(f"1. Analysis of question: {end_time - start_time:0.1f} sec.")

        prompt = global_instruction_and_information + f"""\n\n\nYou are a plant biology RAG expert. concisely and directly answer user question in @@@@ based ONLY on the provided Context Chunks, Entity Relationships (eg. [Source: ...md] Marker-Trait Associations -[MARKER]-> Significant Markers), and Context from Metadata Graph. 
        Cite sources after facts by appending [Source: ]. 
        If file name is like Surname_Year.pdf.md, use Surname Year only and do not include pdf.md. eg. [Source: Wallwork 2022] 
        If file name is like title.pdf.md, use complete file name ending with .pdf.md]. eg. [Source: An_island_of_receptor-like_genes_at_the_Rrs13_locu.pdf.md] 
        If source is from an Entity Relationship, use relevant [Source: Surname Year] or [Source: File Name.pdf.md]. Do not cite [Source: Entity Relationship]. Never cite [Source: Entity Relationship].
        If the source is Metadata Graph, use [Source: Metadata Graph]. 
        Double check citing source. 
        Do not confuse Entity Relationships with Metadata Graph. 
        Do not cite [Source: Background Information] or instructions. Never cite [Source: Background Information]. 

        Do not make up content in answer. 
        If unsure, say "No information available". 
        """
        start_time = time.perf_counter()
        if expanded_queries:
            added_chunk_keys = set()
            for expanded_query in expanded_queries:
                # 1. Retrieve & Hybrid Fusion (RRF)
                v = self._vector_chunks(expanded_query, k)
                f = self._fulltext_chunks(expanded_query, k)
                fused = self._rrf_fusion(v, f)

                # 2. Get Top Seed Chunks & Expand in Graph
                seed_cids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]
                seeded_chunks, expanded_chunks, triples = self._expand_hops(seed_cids)

                # Deduplicate before reranking
                all_chunks_deduplicated = self._dedupe_chunks(seeded_chunks + expanded_chunks)

                # 3. Build final context
                context_chunks, total_characters = [], 0
                for chunk in all_chunks_deduplicated:                    
                    cid = chunk.get("chunk_id")
                    source_path = chunk.get("source_path", "")
                    text = chunk.get("text", "")

                    key = cid if cid else (source_path, hash(text))
                    if key in added_chunk_keys:
                        continue
                    added_chunk_keys.add(key)

                    if not text:
                        continue
                    denominator = max(1, len(expanded_queries))
                    if total_characters + len(text) > max_context_chars / denominator:
                        break
                    context_chunks.append(f"[Source: {chunk.get('source_path', '')}] {text}\n")
                    total_characters += len(text)

                triple_summ = "\n".join(triples[:MAX_TRIPLES])

                prompt += f"""
                    For sub-question
                    {expanded_query}

                    ### Context Chunks are:
                    {os.linesep.join(context_chunks)}

                    ### Entity Relationships are:
                    {triple_summ}

                    """
                
        end_time = time.perf_counter()
        logger.info(f"2. Vector and full-text searching, graph expansion, de-duplication, re-ranking by semantic relevance, form context: {end_time - start_time:0.1f} sec.")


        # 2: Metadata graph search (cypher or hybrid via LLM tool selection) 
        start_time = time.perf_counter()
        metadata_context = self._get_metadata_context(q, expanded_queries)
        if metadata_context:
            prompt += f"""\n\n\n
### [Source: Metadata Graph]:
{metadata_context}"""
        end_time = time.perf_counter()
        logger.info(f"3. Metadata graph search (cypher or hybrid by LLM tool selection): {end_time - start_time:0.1f} sec.")


        # 3. Pretzel documentaion search 
        if "pretzel" in q.lower():
            start_time = time.perf_counter()
            pretzel_context = self._get_pretzel_context(expanded_queries)
            if pretzel_context:
                prompt += f"""\n\n\n
                ### [Source: Pretzel Documentation]: 
                {pretzel_context}"""
            end_time = time.perf_counter()
            logger.info(f"Pretzel documentation search (hybrid): {end_time - start_time:0.1f} sec.")


        # 3: Answer user question based on all retrived context. 
        prompt += f"""



            Analysis of user question: 
            {expanded_question}

            User Question:
            @@@@
            {q}
            @@@@
            Answer:"""
        logger.info(f"Prompt and context: {prompt}")
        
        start_time = time.perf_counter()
        resp = self.llm.invoke(prompt)
        answer = (getattr(resp, "text", None) or getattr(resp, "content", "") or str(resp)).strip()
        usage_metadata = getattr(resp, "usage_metadata", {}) or {}
        end_time = time.perf_counter()
        logger.info(f"4. Call LLM to answer: {end_time - start_time:0.1f} sec.")


        # If user question is about AGG accessions, run accession lookup pipeline
        start_time = time.perf_counter()
        is_agg_accession_query, accession_question, species = self._is_agg_accession_query(q)
        end_time = time.perf_counter()
        logger.info(f"5. Analysis of accession question and species from LLM answer: {end_time - start_time:0.1f} sec.")

        logger.info("is_agg_accession_query: %s", is_agg_accession_query)
        logger.info("accession_question: %s", accession_question)
        logger.info("species: %s", species)

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