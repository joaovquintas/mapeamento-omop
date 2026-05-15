import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus import Collection, connections
from tqdm import tqdm


DEFAULT_MILVUS_URI = "http://localhost:19530"

DEFAULT_SEMANTIC_COLLECTION_NAME = "devices_semantic"
DEFAULT_LEXICAL_COLLECTION_NAME = "devices_lexical"

DEFAULT_SEMANTIC_FIELD = "name_embedding"
DEFAULT_LEXICAL_FIELD = "sparse_name"


def connect_milvus(
    uri: str = DEFAULT_MILVUS_URI,
    alias: str = "default",
) -> None:
    """
    Conecta ao Milvus.
    """

    print(f"[CandidateFinder] Connecting to Milvus: {uri}")

    connections.connect(
        alias=alias,
        uri=uri,
    )


def load_collection(collection_name: str) -> Collection:
    """
    Carrega uma collection do Milvus.
    """

    collection = Collection(collection_name)
    collection.load()

    return collection


def parse_embedding(value) -> np.ndarray:
    """
    Converte embedding salvo como string/lista para np.ndarray.
    """

    if isinstance(value, list):
        return np.array(value, dtype=np.float32)

    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    return np.array(
        ast.literal_eval(value),
        dtype=np.float32,
    )


def search_semantic(
    collection: Collection,
    query_embedding: np.ndarray,
    top_k: int = 20,
    semantic_field: str = DEFAULT_SEMANTIC_FIELD,
) -> list[dict]:
    """
    Busca semântica usando vetor denso.
    """

    search_params = {
        "metric_type": "IP",
        "params": {
            "ef": 128,
        },
    }

    results = collection.search(
        data=[query_embedding],
        anns_field=semantic_field,
        param=search_params,
        limit=top_k,
        output_fields=[
            "concept_id",
            "concept_code",
            "concept_name",
        ],
    )

    hits = []

    if results and len(results) > 0:
        for hit in results[0]:
            hits.append(
                {
                    "concept_id": str(hit.entity.get("concept_id")),
                    "concept_code": str(hit.entity.get("concept_code")),
                    "concept_name": str(hit.entity.get("concept_name")),
                    "semantic_score": float(hit.score),
                    "source": "semantic",
                }
            )

    return hits


def search_lexical(
    collection: Collection,
    query_text: str,
    top_k: int = 20,
    lexical_field: str = DEFAULT_LEXICAL_FIELD,
) -> list[dict]:
    """
    Busca lexical usando BM25 no Milvus.
    """

    search_params = {
        "metric_type": "BM25",
        "params": {},
    }

    results = collection.search(
        data=[query_text],
        anns_field=lexical_field,
        param=search_params,
        limit=top_k,
        output_fields=[
            "concept_id",
            "concept_code",
            "concept_name",
        ],
    )

    hits = []

    if results and len(results) > 0:
        for hit in results[0]:
            hits.append(
                {
                    "concept_id": str(hit.entity.get("concept_id")),
                    "concept_code": str(hit.entity.get("concept_code")),
                    "concept_name": str(hit.entity.get("concept_name")),
                    "lexical_score": float(hit.score),
                    "source": "lexical",
                }
            )

    return hits


def merge_candidates(
    semantic_hits: list[dict],
    lexical_hits: list[dict],
    max_total: int = 20,
) -> list[dict]:
    """
    Junta candidatos semânticos e léxicos removendo duplicatas.
    """

    merged = {}

    for hit in semantic_hits + lexical_hits:
        concept_id = hit["concept_id"]

        if concept_id not in merged:
            merged[concept_id] = {
                "concept_id": hit["concept_id"],
                "concept_code": hit["concept_code"],
                "concept_name": hit["concept_name"],
                "semantic_score": None,
                "lexical_score": None,
            }

        if hit["source"] == "semantic":
            current_score = merged[concept_id]["semantic_score"]

            if current_score is None or hit["semantic_score"] > current_score:
                merged[concept_id]["semantic_score"] = hit["semantic_score"]

        if hit["source"] == "lexical":
            current_score = merged[concept_id]["lexical_score"]

            if current_score is None or hit["lexical_score"] > current_score:
                merged[concept_id]["lexical_score"] = hit["lexical_score"]

    candidates = list(merged.values())

    candidates.sort(
        key=lambda item: max(
            item["semantic_score"] or 0,
            item["lexical_score"] or 0,
        ),
        reverse=True,
    )

    return candidates[:max_total]


def find_candidates(
    input_file: str,
    output_file: str,
    milvus_uri: str = DEFAULT_MILVUS_URI,
    semantic_collection_name: str = DEFAULT_SEMANTIC_COLLECTION_NAME,
    lexical_collection_name: str = DEFAULT_LEXICAL_COLLECTION_NAME,
    top_k_semantic: int = 20,
    top_k_lexical: int = 20,
    max_total_candidates: int = 20,
) -> pd.DataFrame:
    """
    Busca candidatos semânticos e léxicos para cada termo normalizado.

    Entrada esperada:
    - term
    - normalized_en
    - embedding

    Saída:
    - term
    - normalized_en
    - rank
    - concept_id
    - concept_code
    - concept_name
    - semantic_score
    - lexical_score
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_path)

    required_columns = [
        "term",
        "normalized_en",
        "embedding",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    connect_milvus(uri=milvus_uri)

    print("[CandidateFinder] Loading collections...")

    semantic_collection = load_collection(semantic_collection_name)
    lexical_collection = load_collection(lexical_collection_name)

    all_results = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Finding candidates",
    ):
        term = row["term"]
        normalized_en = row["normalized_en"]

        try:
            embedding = parse_embedding(row["embedding"])
        except Exception as error:
            print(f"[CandidateFinder] Error parsing embedding at row {idx}: {error}")
            continue

        semantic_hits = search_semantic(
            collection=semantic_collection,
            query_embedding=embedding,
            top_k=top_k_semantic,
        )

        lexical_hits = search_lexical(
            collection=lexical_collection,
            query_text=str(normalized_en),
            top_k=top_k_lexical,
        )

        merged_candidates = merge_candidates(
            semantic_hits=semantic_hits,
            lexical_hits=lexical_hits,
            max_total=max_total_candidates,
        )

        for rank, candidate in enumerate(merged_candidates, start=1):
            all_results.append(
                {
                    "term": term,
                    "normalized_en": normalized_en,
                    "rank": rank,
                    "concept_id": candidate["concept_id"],
                    "concept_code": candidate["concept_code"],
                    "concept_name": candidate["concept_name"],
                    "semantic_score": candidate["semantic_score"],
                    "lexical_score": candidate["lexical_score"],
                }
            )

    results_df = pd.DataFrame(all_results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print(f"[CandidateFinder] Saved: {output_path}")

    return results_df