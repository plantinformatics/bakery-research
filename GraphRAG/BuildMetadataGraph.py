import os, csv
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, basic_auth
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = ""

os.environ["NEO4J_URI"] = ""
os.environ["NEO4J_USERNAME"] = ""
os.environ["NEO4J_PASSWORD"] = ""

GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_EMBEDDING_DIMS = 3072

FILES = {
    "alignment": r"251027_Metadata_Fields_Update(Alignment).csv",
    "curator": r"251027_Metadata_Fields_Update(Curator).csv",
    "accession": r"251027_Metadata_Fields_Update(DatasetAccession).csv",
    "genetic_map": r"251027_Metadata_Fields_Update(Genetic Map).csv",
    "genome": r"251027_Metadata_Fields_Update(Genome).csv",
    "project": r"251027_Metadata_Fields_Update(Project).csv",
    "qtl": r"251027_Metadata_Fields_Update(QTL).csv",
    "vcf": r"251027_Metadata_Fields_Update(VCF).csv",
}

emb_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)

def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def clean(v):
    if v is None:
        return None
    v = str(v).strip()
    return None if v in {"", "#N/A", "n/a", "N/A"} else v

def first(r, keys):
    for k in keys:
        v = clean(r.get(k))
        if v:
            return v
    return None

def put(p, k, v):
    v = clean(v)
    if v:
        p[k] = v

def text_from(p: Dict[str, Any], extra: List[Any] = None) -> str:
    keys = [
        "displayName", "shortName", "type", "tags", "crop", "species",
        "dataSource", "publication", "comments", "categories",
        "curatorName", "projectName", "projectDescription", "accessionName",
        "alignmentType", "markerType", "parentNames", "populationType",
        "platform", "definedIn"
    ]
    vals = [p.get(k) for k in keys] + (extra or [])
    return " | ".join(str(v).strip() for v in vals if clean(v))

def embed(text: str) -> Optional[List[float]]:
    if not clean(text):
        return None
    try:
        return [float(x) for x in emb_model.embed_query(text)]
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return None

def dataset_props(r):
    p = {}
    mapping = {
        "shortName": ["shortName"],
        "displayName": ["displayName"],
        "type": ["type"],
        "tags": ["tags"],
        "crop": ["Crop"],
        "species": ["species", "Species"],
        "dataSource": ["Data Source"],
        "licensingOfOriginalData": ["Licensing of original data"],
        "publication": ["Publication"],
        "comments": ["Comments"],
        "categories": ["Categories"],
        "genomeId": ["Genome _id"],
        "alignmentType": ["Alignment type"],
        "markerType": ["Marker type"],
        "parentNames": ["Parent names"],
        "populationType": ["Population type"],
        "accessionName": ["Accession name"],
        "projectRef": ["Project ID"],
        "ebiEnaId": ["EBI-ENA ID"],
        "panbarlexName": ["PanBARLEX name"],
        "platform": ["platform"],
        "definedIn": ["Defined in"],
    }
    for out_key, in_keys in mapping.items():
        put(p, out_key, first(r, in_keys))
    return p

# Prepare data 
def prepare_curators(rows):
    out = []
    for r in rows:
        p = {}
        put(p, "contact", first(r, ["Contact"]))
        curator = first(r, ["Curator"])
        out.append({"curatorName": curator, "props": p, "embedding": embed(text_from({"curatorName": curator, **p}))})
    return out

def prepare_accessions(rows):
    out = []
    for r in rows:
        p = {}
        for k, cols in {
            "aliases": ["Aliases"],
            "growthHabit": ["Growth Habit"],
            "origin": ["Origin"],
            "pedigree": ["Pedigree"],
            "yearOfRelease": ["Year of release"],
            "marketClass": ["Market class"],
            "features": ["Features"],
        }.items():
            put(p, k, first(r, cols))
        acc = first(r, ["Accession name"])
        out.append({"accessionName": acc, "props": p, "embedding": embed(text_from({"accessionName": acc, **p}))})
    return out

def prepare_projects(rows):
    out = []
    for r in rows:
        p = {}
        put(p, "projectName", first(r, ["Project name"]))
        put(p, "projectDescription", first(r, ["Project description"]))
        pid = first(r, ["Project ID"])
        out.append({"projectId": pid, "props": p, "embedding": embed(text_from({"projectId": pid, **p}))})
    return out

def prepare_dataset(rows):
    out = []
    for r in rows:
        p = dataset_props(r)
        id_ = first(r, ["_id", "id"])
        curator = first(r, ["Curator"])
        out.append({
            "id": id_,
            "props": p,
            "curatorName": curator,
            "embedding": embed(text_from({**p, "curatorName": curator}))
        })
    return out


SCHEMA = [
    "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT curator_name IF NOT EXISTS FOR (c:Curator) REQUIRE c.curatorName IS UNIQUE",
    "CREATE CONSTRAINT accession_name IF NOT EXISTS FOR (a:DatasetAccession) REQUIRE a.accessionName IS UNIQUE",
    "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.projectId IS UNIQUE",
    f"""CREATE VECTOR INDEX metadata_vector_index IF NOT EXISTS
        FOR (n:MetadataGraph) ON (n.embedding)
        OPTIONS {{indexConfig: {{`vector.dimensions`: {GEMINI_EMBEDDING_DIMS},
                                `vector.similarity_function`: 'cosine'}}}}""",
    """CREATE FULLTEXT INDEX metadata_fulltext_index IF NOT EXISTS
       FOR (n:MetadataGraph) ON EACH [n.displayName, n.shortName, n.comments,
            n.publication, n.categories, n.crop, n.species, n.curatorName,
            n.projectName, n.projectDescription, n.accessionName]"""
]

Q_LOAD_CURATORS = """
UNWIND $rows AS row
WITH row WHERE row.curatorName IS NOT NULL
MERGE (c:Curator:MetadataGraph {curatorName: row.curatorName})
SET c += row.props
FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END | SET c.embedding = row.embedding)
"""

Q_LOAD_ACCESSIONS = """
UNWIND $rows AS row
WITH row WHERE row.accessionName IS NOT NULL
MERGE (a:DatasetAccession:MetadataGraph {accessionName: row.accessionName})
SET a += row.props
FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END | SET a.embedding = row.embedding)
"""

Q_LOAD_PROJECTS = """
UNWIND $rows AS row
WITH row WHERE row.projectId IS NOT NULL
MERGE (p:Project:MetadataGraph {projectId: row.projectId})
SET p += row.props
FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END | SET p.embedding = row.embedding)
"""

def q_load_dataset(extra_label: str) -> str:
    return f"""
UNWIND $rows AS row
WITH row WHERE row.id IS NOT NULL
MERGE (d:Dataset:{extra_label}:MetadataGraph {{id: row.id}})
SET d += row.props
FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END | SET d.embedding = row.embedding)

WITH row, d
OPTIONAL MATCH (c:Curator:MetadataGraph)
WHERE row.curatorName IS NOT NULL
  AND c.curatorName = row.curatorName
FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
  MERGE (d)-[:hasCurator]->(c)
)
"""

Q_LOAD_GENOME = q_load_dataset("GenomeDataset") + """
WITH row, d
OPTIONAL MATCH (a:DatasetAccession:MetadataGraph)
WHERE row.props.accessionName IS NOT NULL
  AND a.accessionName = row.props.accessionName
FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END |
  MERGE (d)-[:hasDatasetAccession]->(a)
)

WITH row, d
OPTIONAL MATCH (p:Project:MetadataGraph)
WHERE row.props.projectRef IS NOT NULL
  AND (toLower(p.projectId) = toLower(row.props.projectRef)
       OR toLower(p.projectName) = toLower(row.props.projectRef))
FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
  MERGE (d)-[:is_part_of]->(p)
)
/* If no Project row matches Genome.Project ID/name, relationship is not possible and is skipped. */
"""

Q_LOAD_ALIGNMENT = q_load_dataset("AlignmentDataset") + """
WITH row, d
OPTIONAL MATCH (g:GenomeDataset:MetadataGraph)
WHERE d.genomeId IS NOT NULL
  AND g.id = d.genomeId
FOREACH (_ IN CASE WHEN g IS NULL THEN [] ELSE [1] END |
  MERGE (d)-[:ALIGNED_TO_GENOME]->(g)
)
/* If Genome _id is blank, Alignment->Genome relationship is not possible. */
"""

Q_LOAD_GENETIC_MAP = q_load_dataset("GeneticMapDataset")
Q_LOAD_QTL = q_load_dataset("QTLDataset")
Q_LOAD_VCF = q_load_dataset("VCFDataset") + """
/* VCF CSV has Project ID but no Defined in column in provided data; DEFINED_IN relationship is not possible here. */
WITH row, d
OPTIONAL MATCH (p:Project:MetadataGraph)
WHERE row.props.projectRef IS NOT NULL
  AND (toLower(p.projectId) = toLower(row.props.projectRef)
       OR toLower(p.projectName) = toLower(row.props.projectRef))
FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
  MERGE (d)-[:is_part_of]->(p)
)
"""

Q_LINK_QTL_DEFINED_IN = """
MATCH (q:QTLDataset:MetadataGraph)
WHERE q.definedIn IS NOT NULL
OPTIONAL MATCH (d:Dataset:MetadataGraph)
WHERE (d:GenomeDataset OR d:GeneticMapDataset)
  AND (toLower(d.id) = toLower(q.definedIn))
FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
  MERGE (q)-[:DEFINED_IN]->(d)
)
/* If Defined in does not match Genome/Genetic Map id/shortName/displayName, relationship is skipped. 
OR toLower(coalesce(d.shortName, "")) = toLower(q.definedIn)
       OR toLower(coalesce(d.displayName, "")) = toLower(q.definedIn)
*/
"""

Q_LINK_GM_TO_ALIGNMENT = """
MATCH (gm:GeneticMapDataset:MetadataGraph), (al:AlignmentDataset:MetadataGraph)
WHERE gm.markerType IS NOT NULL
  AND al.markerType IS NOT NULL
  AND toLower(trim(gm.markerType)) = toLower(trim(al.markerType))
MERGE (gm)-[:ALIGNED_TO]->(al)
"""


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        for q in SCHEMA:
            session.run(q)

        curator_rows = load_csv(FILES["curator"])
        curators = prepare_curators(curator_rows)
        if curators:
            session.run(Q_LOAD_CURATORS, rows=curators)

        accession_rows = load_csv(FILES["accession"])
        accessions = prepare_accessions(accession_rows)
        if accessions:
            session.run(Q_LOAD_ACCESSIONS, rows=accessions)

        project_rows = load_csv(FILES["project"])
        projects = prepare_projects(project_rows)
        if projects:
            session.run(Q_LOAD_PROJECTS, rows=projects)

        genome_rows = load_csv(FILES["genome"])
        genomes = prepare_dataset(genome_rows)
        if genomes:
            session.run(Q_LOAD_GENOME, rows=genomes)

        alignment_rows = load_csv(FILES["alignment"])
        alignments = prepare_dataset(alignment_rows)
        if alignments:
            session.run(Q_LOAD_ALIGNMENT, rows=alignments)

        genetic_map_rows = load_csv(FILES["genetic_map"])
        genetic_maps = prepare_dataset(genetic_map_rows)
        if genetic_maps:
            session.run(Q_LOAD_GENETIC_MAP, rows=genetic_maps)

        qtl_rows = load_csv(FILES["qtl"])
        qtls = prepare_dataset(qtl_rows)
        if qtls:
            session.run(Q_LOAD_QTL, rows=qtls)

        vcf_rows = load_csv(FILES["vcf"])
        vcfs = prepare_dataset(vcf_rows)
        if vcfs:
            session.run(Q_LOAD_VCF, rows=vcfs)

        session.run(Q_LINK_QTL_DEFINED_IN)
        session.run(Q_LINK_GM_TO_ALIGNMENT)

    driver.close()

if __name__ == "__main__":
    main()