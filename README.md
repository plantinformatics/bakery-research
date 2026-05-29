# Bakery-Research

This repository contains plant biology GraphRAG notebooks built with Neo4j, Google Gemini, LangChain, vector search, full-text search, graph traversal, and Bakery-Genebank accession lookup when applicable.

3 main Python programs:

1. `BuildLiteraturePretzelDocuementationGraph.py`
2. `BuildMetadataGraph.py`
3. `Query.py`

## Overview

### Literature and Pretzel Documentation Graph

`BuildLiteraturePretzelDocuementationGraph.py` builds a Neo4j graph from Markdown documents.

It supports:

- loading Markdown files recursively
- splitting documents by Markdown headers and text chunks
- embedding chunks with Gemini embeddings
- storing chunks as Neo4j `Chunk` nodes
- creating vector and full-text indexes
- extracting graph entities and relationships using an LLM
- linking chunks to extracted entities with `MENTIONS`
- loading Pretzel documentation as `PretzelFunction` nodes

Supported modes:

```python
mode = "build"
mode = "add"
mode = "add_pretzel_functions"
```

### Metadata Graph

`BuildMetadataGraph.py` builds a structured Neo4j metadata graph from Pretzel metadata CSV files.

It loads metadata for:

- alignments
- curators
- accessions
- genetic maps
- genomes
- projects
- QTLs
- VCFs

Main node labels include:

```text
Dataset
MetadataGraph
GenomeDataset
AlignmentDataset
GeneticMapDataset
QTLDataset
VCFDataset
Curator
DatasetAccession
Project
```

Main relationships include:

```text
hasCurator
hasDatasetAccession
isPartOf
alignedToGenome
alignedTo
definedIn
```

### Query Pipeline

`Query.py` runs the GraphRAG query workflow.

It performs:

- question expansion
- hybrid vector and full-text search
- Reciprocal Rank Fusion
- graph expansion from retrieved chunks
- metadata graph retrieval using hybrid search or generated Cypher
- Pretzel documentation search when the question mentions Pretzel
- final answer generation with source citations
- Bakery-Genebank lookup through API when applicable 

## Prerequisites

### Python

Use Python 3.14 or later.

### Neo4j

A running Neo4j database is required.

### Google Gemini API

A Google Gemini API key is required for:

- Gemini chat models
- Gemini embedding model


## Configuration

### Environment Variables

Set the following environment variables before running the scripts.

Linux or macOS:

```bash
export GOOGLE_API_KEY="your-google-api-key"
export NEO4J_URI="your-neo4j-uri"
export NEO4J_USERNAME="your-username"
export NEO4J_PASSWORD="your-password"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your-google-api-key"
$env:NEO4J_URI="your-neo4j-uri"
$env:NEO4J_USERNAME="your-username"
$env:NEO4J_PASSWORD="your-password"
```

### Literature Graph Configuration

Edit these values in `BuildLiteraturePretzelDocuementationGraph.py`:

```python
mode = "build"
md_dir = r"path\to\markdown\literature"
add_dir = r"path\to\additional\markdown"
pretzel_functions_dir = r"path\to\pretzel\documentation"
extract_nodes = False
schema = r"path\to\shapes.ttl"
```

### Metadata Graph Configuration

Edit Neo4j credentials and CSV paths in `BuildMetadataGraph.py`:

```python
NEO4J_URI = "your-neo4j-uri"
NEO4J_USER = "your-username"
NEO4J_PASS = "your-password"
```

Expected CSV files:

```text
251027_Metadata_Fields_Update(Alignment).csv
251027_Metadata_Fields_Update(Curator).csv
251027_Metadata_Fields_Update(DatasetAccession).csv
251027_Metadata_Fields_Update(Genetic Map).csv
251027_Metadata_Fields_Update(Genome).csv
251027_Metadata_Fields_Update(Project).csv
251027_Metadata_Fields_Update(QTL).csv
251027_Metadata_Fields_Update(VCF).csv
```

### Bakery-Genebank API Configuration

`Query.py` can call an instance of Bakery-Genebank via API:

```python
ACCESSION_API_URL = "your-bakery-genebank-api-url"
ACCESSION_API_TOKEN = "your-bakery-genebank-api-token"
ACCESSION_API_TIMEOUT = 120
```

## Usage

### 1. Build the Metadata Graph

```bash
python BuildMetadataGraph.py
```

### 2. Build the Literature Graph

Set the mode in `BuildLiteraturePretzelDocuementationGraph.py`:

```python
mode = "build"
```

Run:

```bash
python BuildLiteraturePretzelDocuementationGraph.py
```

### 3. Add New Literature Documents

Set:

```python
mode = "add"
extract_nodes = True
```

Run:

```bash
python BuildLiteraturePretzelDocuementationGraph.py
```

If `extract_nodes` is `False`, the script only stores chunks without graph extraction.

### 4. Add Pretzel Documentation

Set:

```python
mode = "add_pretzel_functions"
```

Run:

```bash
python BuildLiteraturePretzelDocuementationGraph.py
```

### 5. Query the GraphRAG System

```bash
python Query.py "User question"
```

## Neo4j Indexes

The pipeline creates vector and full-text indexes for retrieval.

### Literature Graph

```text
vector
idx_chunk_text
idx_node_name
```

### Metadata Graph

```text
metadata_vector_index
metadata_fulltext_index
```

### Pretzel Documentation Graph

```text
pretzel_functions_vector
idx_pretzel_function_text
```

## Query Workflow

`Query.py` follows this high-level workflow:

1. Expand the user question.
2. Generate retrieval-optimised sub-questions.
3. Search literature chunks using vector and full-text search.
4. Fuse search results using Reciprocal Rank Fusion.
5. Expand retrieved chunks through graph relationships.
6. Search the metadata graph using hybrid search or generated Cypher.
7. Search Pretzel documentation if the question mentions Pretzel.
8. Build a final prompt using retrieved context.
9. Generate an answer with citations.
10. If relevant, call the Bakery-Genebank API and append accession lookup results.

## Logs and Outputs

Build logs:

```text
build.log
add.log
addpretzelfunctions.log
```

Failed document batches:

```text
failed_docs.jsonl
```

## Safety Controls

`Query.py` validates generated Cypher before execution.

Blocked Cypher operations include:

```text
CREATE
MERGE
DELETE
DETACH
SET
REMOVE
DROP
LOAD CSV
CALL DBMS
APOC
CREATE INDEX
CREATE CONSTRAINT
```

This keeps generated query-time Cypher read-only.
