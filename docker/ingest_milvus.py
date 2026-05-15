#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest SNOMED Disorders (disorders.csv) into Milvus with:
- concept_name text field
- BM25 sparse vector (syntactic)
- MedTE dense embedding (semantic)
- Resume capability: skips concept_id already in collection
"""

import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Function,
    FunctionType,
)
from pymilvus.exceptions import MilvusException
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "datasets" / "snomed_devices_standard.csv"
CSV_CHUNK_SIZE = 50_000
BATCH_SIZE = 10240
EMBED_BATCH_SIZE = 128
FLUSH_EVERY: Optional[int] = None
MAX_CONCEPT_NAME_LEN = 1024

MILVUS_URI = "http://localhost:19530"
MILVUS_USER = ""
MILVUS_PASSWORD = ""
MILVUS_TOKEN = ""
COLLECTION_NAME = "devices_semantic"
RESET_COLLECTION = True  # cuidado: se True, sempre começa do zero

#EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_MODEL = "MohammadKhodadad/MedTE-cl15-step-8000"


class EmbeddingEncoder:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = self._load_model(self.device)
        except Exception as e:
            if self.device == "cpu":
                raise
            print(f"[WARN] failed to load embedding model on {self.device}: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model = self._load_model(self.device)

    def _load_model(self, device: str) -> SentenceTransformer:
        model = SentenceTransformer(self.model_name, device=device)
        model.eval()
        return model

    def encode(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        try:
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device,
            )
        except SystemError as e:
            if self.device == "cpu" or "PyCFunction" not in str(e):
                raise
            print(f"[WARN] encode failed on {self.device}: {e}. Retrying on CPU.")
            self.device = "cpu"
            self.model = self._load_model(self.device)
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device,
            )
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        return embs.astype(np.float32)


def release_collection(collection_name: str) -> None:
    if not utility.has_collection(collection_name):
        return
    try:
        Collection(name=collection_name).release()
        print(f"[RELEASE] collection '{collection_name}' released from memory")
    except MilvusException as e:
        print(f"[WARN] release failed for '{collection_name}': {e}")


def ensure_collection(collection_name: str, dim: int, reset: bool = False, load: bool = False) -> Collection:
    if utility.has_collection(collection_name):
        if reset:
            release_collection(collection_name)
            utility.drop_collection(collection_name)

    analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="concept_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="concept_code", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(
                name="concept_name",
                dtype=DataType.VARCHAR,
                max_length=1024,
                enable_match=True,
                enable_analyzer=True,
                analyzer_params=analyzer_params,
            ),
            FieldSchema(name="sparse_name", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="name_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="domain_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="vocabulary_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="concept_class_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="standard_concept", dtype=DataType.VARCHAR, max_length=4),
            FieldSchema(name="valid_start_date", dtype=DataType.INT64),
            FieldSchema(name="valid_end_date", dtype=DataType.INT64),
            FieldSchema(name="invalid_reason", dtype=DataType.VARCHAR, max_length=4),
        ]

        fn_bm25 = Function(
            name="bm25_name",
            function_type=FunctionType.BM25,
            input_field_names=["concept_name"],
            output_field_names="sparse_name",
        )

        schema = CollectionSchema(fields=fields, description="SNOMED Disorders")
        schema.add_function(fn_bm25)

        collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")
    else:
        collection = Collection(name=collection_name)

    if load:
        try:
            collection.load()
        except MilvusException:
            pass
    return collection


def ensure_indexes(collection: Collection) -> None:
    existing_index_fields = {idx.field_name for idx in collection.indexes}

    if "sparse_name" not in existing_index_fields:
        print("[INDEX] creating BM25 sparse index...")
        collection.create_index(
            field_name="sparse_name",
            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
        )

    if "name_embedding" not in existing_index_fields:
        print("[INDEX] creating HNSW dense index...")
        collection.create_index(
            field_name="name_embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 32, "efConstruction": 200},
            },
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
                "concept_code": row.get("concept_code") or "",
                "concept_name": concept_name,
                "domain_id": row.get("domain_id") or "",
                "vocabulary_id": row.get("vocabulary_id") or "",
                "concept_class_id": row.get("concept_class_id") or "",
                "standard_concept": row.get("standard_concept") or "",
                "valid_start_date": row.get("valid_start_date") or "0",
                "valid_end_date": row.get("valid_end_date") or "0",
                "invalid_reason": row.get("invalid_reason") or "",
            }
        )
    return rows


def load_existing_ids(collection: Collection) -> set[int]:
    existing: set[int] = set()
    try:
        collection.load()
        # consulta simples: pega todos concept_id já inseridos
        offset = 0
        step = 16384
        while True:
            res = collection.query(
                expr="concept_id >= 0",
                output_fields=["concept_id"],
                limit=step,
                offset=offset,
            )
            if not res:
                break
            for r in res:
                existing.add(int(r["concept_id"]))
            offset += len(res)
    except MilvusException:
        # se a collection estiver vazia ou query não suportar offset grande,
        # simplesmente assume que não tem nada
        pass
    finally:
        try:
            collection.release()
        except MilvusException:
            pass
    print(f"Found {len(existing)} existing concept_ids in Milvus.")
    return existing


def safe_upsert(collection: Collection, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    schema_fields = {f.name: f for f in collection.schema.fields}
    field_names = [f for f in schema_fields.keys() if schema_fields[f].dtype != DataType.SPARSE_FLOAT_VECTOR]

    cols: Dict[str, List[Any]] = {fn: [] for fn in field_names}

    for r in rows:
        cols["concept_id"].append(int(r["concept_id"]))
        cols["concept_code"].append(str(r.get("concept_code") or ""))
        cols["concept_name"].append(str(r.get("concept_name") or "")[:MAX_CONCEPT_NAME_LEN])
        cols["name_embedding"].append(list(r["name_embedding"]))
        cols["domain_id"].append(str(r.get("domain_id") or ""))
        cols["vocabulary_id"].append(str(r.get("vocabulary_id") or ""))
        cols["concept_class_id"].append(str(r.get("concept_class_id") or ""))
        cols["standard_concept"].append(str(r.get("standard_concept") or ""))
        cols["valid_start_date"].append(int(r.get("valid_start_date") or 0))
        cols["valid_end_date"].append(int(r.get("valid_end_date") or 0))
        cols["invalid_reason"].append(str(r.get("invalid_reason") or ""))

    entities = [cols[fn] for fn in field_names]
    mr = collection.insert(entities)
    return mr.insert_count or 0


def ingest():
    if MILVUS_TOKEN:
        connections.connect("default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    else:
        connections.connect("default", uri=MILVUS_URI, user=MILVUS_USER or None, password=MILVUS_PASSWORD or None)

    encoder = EmbeddingEncoder(EMBEDDING_MODEL)
    print(f"Embedding model device: {encoder.device}")
    probe = encoder.encode(["probe"], batch_size=1)
    dim = int(probe.shape[1])

    release_collection(COLLECTION_NAME)
    collection = ensure_collection(COLLECTION_NAME, dim, reset=RESET_COLLECTION, load=False)
    #existing_ids = load_existing_ids(collection)

    try:
        total_inserted = 0
        total_skipped = 0
        rows_since_flush = 0
        total_rows = 0
        total_filtered_rows = 0
        batch_idx = 0

        csv_chunks = pd.read_csv(INPUT_CSV, dtype=str, chunksize=CSV_CHUNK_SIZE)
        for csv_chunk_idx, csv_chunk in enumerate(csv_chunks):
            total_rows += len(csv_chunk)

            filtered_chunk = csv_chunk[csv_chunk["concept_id"].notna()].copy()
            filtered_chunk = filtered_chunk[
                (filtered_chunk["standard_concept"] == "S")
                & (filtered_chunk["invalid_reason"].isna())
            ]

            if filtered_chunk.empty:
                print(
                    f"[CSV CHUNK {csv_chunk_idx}] rows={len(csv_chunk)}, "
                    "filtered_rows=0"
                )
                continue

            filtered_chunk["concept_id_int"] = filtered_chunk["concept_id"].astype(int)
            filtered_chunk = filtered_chunk.sort_values(by="concept_id_int").reset_index(drop=True)
            total_filtered_rows += len(filtered_chunk)

            print(
                f"[CSV CHUNK {csv_chunk_idx}] rows={len(csv_chunk)}, "
                f"filtered_rows={len(filtered_chunk)}"
            )

            for start in range(0, len(filtered_chunk), BATCH_SIZE):
                chunk = filtered_chunk.iloc[start:start + BATCH_SIZE]
                if len(chunk) == 0:
                    continue

                first_id = int(chunk["concept_id"].iloc[0])
                concept_name_lengths = chunk["concept_name"].fillna("").str.len()
                print(
                    f"[BATCH {batch_idx}] first_concept_id={first_id}, rows={len(chunk)}, "
                    f"mean_chars={concept_name_lengths.mean():.1f}, max_chars={int(concept_name_lengths.max())}"
                )

                rows = build_rows(chunk)
                if not rows:
                    batch_idx += 1
                    continue

                texts = [f"passage: {r['concept_name']}" for r in rows]
                embs = encoder.encode(texts, batch_size=EMBED_BATCH_SIZE)

                for i, r in enumerate(rows):
                    r["name_embedding"] = embs[i]

                inserted = safe_upsert(collection, rows)
                total_inserted += inserted
                rows_since_flush += inserted

                if FLUSH_EVERY is not None and rows_since_flush >= FLUSH_EVERY:
                    try:
                        collection.flush()
                        print(f"[FLUSH] persisted {rows_since_flush} rows since last flush")
                        rows_since_flush = 0
                    except MilvusException as e:
                        print(f"[WARN] periodic flush failed: {e}")

                print(f"Inserted batch: {inserted}, total_inserted={total_inserted}, total_skipped={total_skipped}")
                batch_idx += 1

        if rows_since_flush > 0:
            try:
                collection.flush()
                print(f"[FLUSH] final flush persisted {rows_since_flush} rows")
            except MilvusException as e:
                print(f"[WARN] final flush failed: {e}")

        ensure_indexes(collection)
        print(
            f"Done. CSV rows={total_rows}, filtered_rows={total_filtered_rows}, "
            f"inserted={total_inserted}, skipped_existing={total_skipped}"
        )
    finally:
        release_collection(COLLECTION_NAME)


if __name__ == "__main__":
    ingest()
