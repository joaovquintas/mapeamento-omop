from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_EMBEDDING_MODEL_NAME = "MohammadKhodadad/MedTE-cl15-step-8000"


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """
    Carrega o modelo de embeddings.
    Usa CUDA se disponível.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Embeddings] Loading model on: {device}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    model.eval()

    return model


def encode_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 128,
    prefix: str = "passage:",
) -> list[list[float]]:
    """
    Gera embeddings normalizados L2.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepared_texts = [
        f"{prefix} {text}" for text in texts
    ]

    all_embeddings = []

    for i in tqdm(
        range(0, len(prepared_texts), batch_size),
        desc="Generating embeddings",
    ):
        batch = prepared_texts[i:i + batch_size]

        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device,
        )

        embeddings = embeddings / (
            np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            ) + 1e-12
        )

        all_embeddings.extend(
            embeddings.astype(np.float32).tolist()
        )

    return all_embeddings


def generate_embeddings(
    input_file: str,
    output_file: str,
    text_column: str = "normalized_en",
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Lê o CSV normalizado e adiciona uma coluna `embedding`.

    Entrada esperada:
    - term
    - is_valid_term
    - normalized_en

    Saída:
    - mesmas colunas
    - embedding
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    df = df[df[text_column].notna()].copy()
    df = df.reset_index(drop=True)

    texts = df[text_column].astype(str).tolist()

    print(f"[Embeddings] Total texts: {len(texts)}")

    model = load_embedding_model(model_name=model_name)

    embeddings = encode_texts(
        texts=texts,
        model=model,
        batch_size=batch_size,
    )

    df["embedding"] = embeddings

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"[Embeddings] Saved: {output_path}")

    return df