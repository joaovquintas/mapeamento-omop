import time
from pathlib import Path

from agents import (
    process_normalizer,
    generate_embeddings,
    find_candidates,
    process_reranker,
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATASETS_DIR / "medical_device_dataset.csv"

NORMALIZER_OUTPUT_FILE = OUTPUTS_DIR / "output_normalizer.csv"
EMBEDDINGS_OUTPUT_FILE = OUTPUTS_DIR / "medical_devices_with_embeddings.csv"
CANDIDATES_OUTPUT_FILE = OUTPUTS_DIR / "candidates_results.csv"
RERANKER_OUTPUT_FILE = OUTPUTS_DIR / "reranker_results.csv"


# =========================================================
# MODELS
# =========================================================

OLLAMA_MODEL_NAME = "qwen2.5:7b"

#deepseek-r1:8b - muito pesado para rodar - faltou vram
#"hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:latest" - coerente ~10
# qwen2.5:7b mais coerente - levou 11 min
# gemma4:e2b 

EMBEDDING_MODEL_NAME = "MohammadKhodadad/MedTE-cl15-step-8000"


# =========================================================
# MILVUS
# =========================================================

MILVUS_URI = "http://localhost:19530"

SEMANTIC_COLLECTION_NAME = "devices_semantic"
LEXICAL_COLLECTION_NAME = "devices_lexical"


# =========================================================
# SETTINGS
# =========================================================

TERM_COLUMN = "term"
NORMALIZED_COLUMN = "normalized_en"

EMBEDDING_BATCH_SIZE = 128

TOP_K_SEMANTIC = 20
TOP_K_LEXICAL = 20
MAX_TOTAL_CANDIDATES = 20

TOP_N_CANDIDATES_FOR_RERANKER = 10

RERANKER_MAX_WORKERS = 4
RERANKER_SAVE_EVERY = 25
RERANKER_RESUME = True


# =========================================================
# CONTROL
# =========================================================

RUN_NORMALIZER = True
RUN_EMBEDDINGS = True
RUN_CANDIDATE_FINDER = True
RUN_RERANKER = True


def ensure_directories() -> None:
    """
    Garante que as pastas necessárias existem.
    """

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def check_input_file() -> None:
    """
    Valida se o arquivo de entrada existe.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nArquivo de entrada não encontrado:\n{INPUT_FILE}\n\n"
            f"Coloque o arquivo input3.csv dentro da pasta datasets/."
        )


def print_elapsed_time(start_time: float) -> None:
    """
    Mostra tempo total de execução.
    """

    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time

    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60

    print("\n======================================")
    print("PIPELINE FINALIZADO")
    print("======================================")

    print("\nResultado final:")
    print(f"{RERANKER_OUTPUT_FILE}")

    print(
        f"\nTempo total: "
        f"{hours:02d}h "
        f"{minutes:02d}m "
        f"{seconds:05.2f}s"
    )


def main() -> None:
    start_time = time.perf_counter()

    ensure_directories()
    check_input_file()

    print("\n======================================")
    print("INICIANDO PIPELINE")
    print("======================================")

    print("\nInput file:")
    print(INPUT_FILE)

    # =====================================================
    # STEP 1 - NORMALIZER
    # =====================================================

    if RUN_NORMALIZER:
        print("\n[1/4] NORMALIZER")
        print("--------------------------------------")

        process_normalizer(
            input_file=str(INPUT_FILE),
            output_file=str(NORMALIZER_OUTPUT_FILE),
            term_column=TERM_COLUMN,
            model_name=OLLAMA_MODEL_NAME,
        )

        print("\nNormalizer finalizado.")
        print(f"Output: {NORMALIZER_OUTPUT_FILE}")

    else:
        print("\n[1/4] NORMALIZER IGNORADO")

    # =====================================================
    # STEP 2 - EMBEDDINGS
    # =====================================================

    if RUN_EMBEDDINGS:
        print("\n[2/4] EMBEDDINGS")
        print("--------------------------------------")

        generate_embeddings(
            input_file=str(NORMALIZER_OUTPUT_FILE),
            output_file=str(EMBEDDINGS_OUTPUT_FILE),
            text_column=NORMALIZED_COLUMN,
            model_name=EMBEDDING_MODEL_NAME,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        print("\nEmbeddings finalizados.")
        print(f"Output: {EMBEDDINGS_OUTPUT_FILE}")

    else:
        print("\n[2/4] EMBEDDINGS IGNORADO")

    # =====================================================
    # STEP 3 - CANDIDATE FINDER
    # =====================================================

    if RUN_CANDIDATE_FINDER:
        print("\n[3/4] CANDIDATE FINDER")
        print("--------------------------------------")

        find_candidates(
            input_file=str(EMBEDDINGS_OUTPUT_FILE),
            output_file=str(CANDIDATES_OUTPUT_FILE),
            milvus_uri=MILVUS_URI,
            semantic_collection_name=SEMANTIC_COLLECTION_NAME,
            lexical_collection_name=LEXICAL_COLLECTION_NAME,
            top_k_semantic=TOP_K_SEMANTIC,
            top_k_lexical=TOP_K_LEXICAL,
            max_total_candidates=MAX_TOTAL_CANDIDATES,
        )

        print("\nCandidate finder finalizado.")
        print(f"Output: {CANDIDATES_OUTPUT_FILE}")

    else:
        print("\n[3/4] CANDIDATE FINDER IGNORADO")

    # =====================================================
    # STEP 4 - RERANKER
    # =====================================================

    if RUN_RERANKER:
        print("\n[4/4] RERANKER")
        print("--------------------------------------")

        process_reranker(
            input_file=str(CANDIDATES_OUTPUT_FILE),
            output_file=str(RERANKER_OUTPUT_FILE),
            model_name=OLLAMA_MODEL_NAME,
            top_n_candidates=TOP_N_CANDIDATES_FOR_RERANKER,
            max_workers=RERANKER_MAX_WORKERS,
            save_every=RERANKER_SAVE_EVERY,
            resume=RERANKER_RESUME,
        )

        print("\nReranker finalizado.")
        print(f"Output: {RERANKER_OUTPUT_FILE}")

    else:
        print("\n[4/4] RERANKER IGNORADO")

    # =====================================================
    # FINAL TIMER
    # =====================================================

    print_elapsed_time(start_time)


if __name__ == "__main__":
    main()