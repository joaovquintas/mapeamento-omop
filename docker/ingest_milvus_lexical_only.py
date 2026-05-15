#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest RxNorm into Milvus using lexical-only search over `concept_name`.

This script:
- reads `rxnorm_combined_filtered.csv` in streaming mode
- creates a collection with BM25 sparse index generated from `concept_name`
- avoids dense embeddings/HNSW to reduce Milvus RAM usage during load/search
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)
from pymilvus.exceptions import MilvusException

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "datasets" / "snomed_devices_standard.csv"
CSV_CHUNK_SIZE = 50_000
BATCH_SIZE = 10_240
FLUSH_EVERY: Optional[int] = None
MAX_CONCEPT_NAME_LEN = 1024

MILVUS_URI = "http://localhost:19530"
MILVUS_USER = ""
MILVUS_PASSWORD = ""
MILVUS_TOKEN = ""
COLLECTION_NAME = "devices_lexical"
RESET_COLLECTION = True  # cuidado: se True, sempre recria a collection


def release_collection(collection_name: str) -> None:
    if not utility.has_collection(collection_name):
        return
    try:
        Collection(name=collection_name).release()
        print(f"[RELEASE] collection '{collection_name}' released from memory")
    except MilvusException as e:
        print(f"[WARN] release failed for '{collection_name}': {e}")


def ensure_collection(collection_name: str, reset: bool = False) -> Collection:
    if utility.has_collection(collection_name) and reset:
        release_collection(collection_name)
        utility.drop_collection(collection_name)

    analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}

    if utility.has_collection(collection_name):
        return Collection(name=collection_name)

    fields = [
        FieldSchema(name="concept_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="concept_code", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(
            name="concept_name",
            dtype=DataType.VARCHAR,
            max_length=MAX_CONCEPT_NAME_LEN,
            enable_match=True,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
        ),
        FieldSchema(name="sparse_name", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="domain_id", dtype=DataType.VARCHAR, max_length=64),
    ]

    fn_bm25 = Function(
        name="bm25_name",
        function_type=FunctionType.BM25,
        input_field_names=["concept_name"],
        output_field_names="sparse_name",
    )

    schema = CollectionSchema(fields=fields, description="RxNorm lexical-only collection")
    schema.add_function(fn_bm25)
    return Collection(name=collection_name, schema=schema, consistency_level="Bounded")


def ensure_indexes(collection: Collection) -> None:
    existing_index_fields = {idx.field_name for idx in collection.indexes}
    if "sparse_name" in existing_index_fields:
        return

    print("[INDEX] creating BM25 sparse index...")
    collection.create_index(
        field_name="sparse_name",
        index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
    )


def build_rows(chunk: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in chunk.iterrows():
        concept_id = row.get("concept_id")
        concept_name = str(row.get("concept_name") or "")[:MAX_CONCEPT_NAME_LEN]
        if not concept_id or pd.isna(concept_id) or not concept_name:
            continue

        rows.append(
            {
                "concept_id": int(concept_id),
                "concept_code": str(row.get("concept_code") or ""),
                "concept_name": concept_name,
                "domain_id": str(row.get("domain_id") or ""),
            }
        )
    return rows


def insert_rows(collection: Collection, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    entities = [
        [row["concept_id"] for row in rows],
        [row["concept_code"] for row in rows],
        [row["concept_name"] for row in rows],
        [row["domain_id"] for row in rows],
    ]
    mr = collection.insert(entities)
    return mr.insert_count or 0


def ingest() -> None:
    if MILVUS_TOKEN:
        connections.connect("default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    else:
        connections.connect(
            "default",
            uri=MILVUS_URI,
            user=MILVUS_USER or None,
            password=MILVUS_PASSWORD or None,
        )

    release_collection(COLLECTION_NAME)
    collection = ensure_collection(COLLECTION_NAME, reset=RESET_COLLECTION)

    try:
        total_rows = 0
        total_valid_rows = 0
        total_inserted = 0
        rows_since_flush = 0
        batch_idx = 0

        csv_chunks = pd.read_csv(INPUT_CSV, dtype=str, chunksize=CSV_CHUNK_SIZE)
        for csv_chunk_idx, csv_chunk in enumerate(csv_chunks):
            total_rows += len(csv_chunk)

            filtered_chunk = csv_chunk[csv_chunk["concept_id"].notna()].copy()
            filtered_chunk = filtered_chunk[filtered_chunk["concept_name"].notna()]

            if filtered_chunk.empty:
                print(f"[CSV CHUNK {csv_chunk_idx}] rows={len(csv_chunk)}, valid_rows=0")
                continue

            filtered_chunk["concept_id_int"] = filtered_chunk["concept_id"].astype(int)
            filtered_chunk = filtered_chunk.sort_values(by="concept_id_int").reset_index(drop=True)
            total_valid_rows += len(filtered_chunk)

            print(
                f"[CSV CHUNK {csv_chunk_idx}] rows={len(csv_chunk)}, "
                f"valid_rows={len(filtered_chunk)}"
            )

            for start in range(0, len(filtered_chunk), BATCH_SIZE):
                chunk = filtered_chunk.iloc[start:start + BATCH_SIZE]
                if chunk.empty:
                    continue

                first_id = int(chunk["concept_id"].iloc[0])
                concept_name_lengths = chunk["concept_name"].fillna("").str.len()
                print(
                    f"[BATCH {batch_idx}] first_concept_id={first_id}, rows={len(chunk)}, "
                    f"mean_chars={concept_name_lengths.mean():.1f}, max_chars={int(concept_name_lengths.max())}"
                )

                rows = build_rows(chunk)
                inserted = insert_rows(collection, rows)
                total_inserted += inserted
                rows_since_flush += inserted

                if FLUSH_EVERY is not None and rows_since_flush >= FLUSH_EVERY:
                    try:
                        collection.flush()
                        print(f"[FLUSH] persisted {rows_since_flush} rows since last flush")
                        rows_since_flush = 0
                    except MilvusException as e:
                        print(f"[WARN] periodic flush failed: {e}")

                print(f"Inserted batch: {inserted}, total_inserted={total_inserted}")
                batch_idx += 1

        if rows_since_flush > 0:
            try:
                collection.flush()
                print(f"[FLUSH] final flush persisted {rows_since_flush} rows")
            except MilvusException as e:
                print(f"[WARN] final flush failed: {e}")

        ensure_indexes(collection)
        print(
            f"Done. CSV rows={total_rows}, valid_rows={total_valid_rows}, "
            f"inserted={total_inserted}, collection={COLLECTION_NAME}"
        )
    finally:
        release_collection(COLLECTION_NAME)


if __name__ == "__main__":
    ingest()
