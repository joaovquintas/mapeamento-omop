import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from ollama import chat
from tqdm import tqdm

from agents.prompts import RERANKER_PROMPT


DEFAULT_MODEL_NAME = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:latest"


def clean_json_response(response_text: str) -> str:
    """
    Remove markdown/code fences e tenta extrair JSON.
    """

    if not response_text:
        return ""

    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        return match.group()

    return cleaned


def render_prompt(
    template: str,
    term_original: str,
    normalized_en: str,
    candidates_json: str,
) -> str:
    """
    Preenche placeholders sem usar .format(),
    porque o prompt contém exemplos JSON com chaves.
    """

    return (
        template
        .replace("{term_original}", term_original)
        .replace("{normalized_en}", normalized_en)
        .replace("{candidates_json}", candidates_json)
    )


def safe_float(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    if pd.isna(value):
        return None

    try:
        return int(value)
    except Exception:
        return None


def safe_str(value):
    if pd.isna(value):
        return None

    return str(value)


def empty_reranker_result(
    term_original,
    normalized_en,
    reasoning: str,
) -> dict:
    return {
        "term": term_original,
        "normalized_en": normalized_en,
        "selected_concept_id": None,
        "selected_concept_code": None,
        "selected_concept_name": None,
        "reasoning": reasoning,
    }


def build_candidates_json(
    group: pd.DataFrame,
    top_n: int = 10,
) -> str:
    """
    Monta JSON dos candidatos para enviar ao LLM.
    """

    group = group.sort_values("rank").head(top_n)

    candidates = []

    for _, row in group.iterrows():
        candidates.append(
            {
                "rank": safe_int(row.get("rank")),
                "concept_id": safe_str(row.get("concept_id")),
                "concept_code": safe_str(row.get("concept_code")),
                "concept_name": safe_str(row.get("concept_name")),
                "semantic_score": safe_float(row.get("semantic_score")),
                "lexical_score": safe_float(row.get("lexical_score")),
            }
        )

    return json.dumps(
        candidates,
        ensure_ascii=False,
        indent=2,
    )


def validate_candidate_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "term",
        "normalized_en",
        "rank",
        "concept_id",
        "concept_code",
        "concept_name",
        "semantic_score",
        "lexical_score",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def create_term_key(term_original, normalized_en) -> tuple[str, str]:
    term_original = "" if pd.isna(term_original) else str(term_original)
    normalized_en = "" if pd.isna(normalized_en) else str(normalized_en)

    return term_original, normalized_en


def rerank_candidates(
    term_original: str,
    normalized_en: str,
    candidates_json: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """
    Chama o LLM para escolher o melhor candidato.
    """

    prompt = render_prompt(
        template=RERANKER_PROMPT,
        term_original=term_original,
        normalized_en=normalized_en,
        candidates_json=candidates_json,
    )

    try:
        response = chat(
            model=model_name,
            format="json",
            options={
                "temperature": 0,
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_text = response["message"]["content"]

    except Exception as error:
        print(f"[Ranker] Erro ao chamar Ollama para termo '{term_original}': {error}")

        return empty_reranker_result(
            term_original=term_original,
            normalized_en=normalized_en,
            reasoning=f"Erro ao chamar o modelo Ollama: {error}",
        )

    cleaned_json = clean_json_response(response_text)

    try:
        parsed = json.loads(cleaned_json)

        return {
            "term": term_original,
            "normalized_en": normalized_en,
            "selected_concept_id": parsed.get("selected_concept_id"),
            "selected_concept_code": parsed.get("selected_concept_code"),
            "selected_concept_name": parsed.get("selected_concept_name"),
            "reasoning": parsed.get("reasoning"),
        }

    except json.JSONDecodeError:
        print("[Ranker] Erro ao fazer parse do JSON retornado pelo LLM.")
        print(f"[Ranker] Termo: {term_original}")
        print(f"[Ranker] Resposta original: {response_text}")
        print(f"[Ranker] Resposta limpa: {cleaned_json}")

        return empty_reranker_result(
            term_original=term_original,
            normalized_en=normalized_en,
            reasoning="Erro ao processar JSON retornado pelo LLM",
        )

    except Exception as error:
        return empty_reranker_result(
            term_original=term_original,
            normalized_en=normalized_en,
            reasoning=f"Erro inesperado ao processar resposta do LLM: {error}",
        )


def process_one_group(args) -> dict:
    """
    Processa um grupo de candidatos.
    Usado pelo ThreadPoolExecutor.
    """

    (
        term_original,
        normalized_en,
        group,
        top_n_candidates,
        model_name,
    ) = args

    term_original = "" if pd.isna(term_original) else str(term_original)
    normalized_en = "" if pd.isna(normalized_en) else str(normalized_en)

    if group.empty:
        return empty_reranker_result(
            term_original=term_original,
            normalized_en=normalized_en,
            reasoning="Nenhum candidato disponível para reranking",
        )

    candidates_json = build_candidates_json(
        group=group,
        top_n=top_n_candidates,
    )

    return rerank_candidates(
        term_original=term_original,
        normalized_en=normalized_en,
        candidates_json=candidates_json,
        model_name=model_name,
    )


def process_reranker(
    input_file: str,
    output_file: str,
    model_name: str = DEFAULT_MODEL_NAME,
    top_n_candidates: int = 10,
    max_workers: int = 4,
    save_every: int = 25,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Processa candidatos e gera resultado final reranqueado.

    Recursos:
    - paralelismo;
    - retomada automática;
    - salvamento parcial.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_path)

    validate_candidate_columns(df)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    processed_terms = set()

    if resume and output_path.exists():
        print(f"[Ranker] Existing output found: {output_path}")
        print("[Ranker] Loading previous results to resume...")

        previous_df = pd.read_csv(output_path)

        if not previous_df.empty:
            results = previous_df.to_dict("records")

            for _, row in previous_df.iterrows():
                key = create_term_key(
                    row.get("term"),
                    row.get("normalized_en"),
                )
                processed_terms.add(key)

        print(f"[Ranker] Already processed terms: {len(processed_terms)}")

    grouped = df.groupby(
        ["term", "normalized_en"],
        dropna=False,
    )

    tasks = []

    for (term_original, normalized_en), group in grouped:
        term_key = create_term_key(
            term_original,
            normalized_en,
        )

        if term_key in processed_terms:
            continue

        tasks.append(
            (
                term_original,
                normalized_en,
                group.copy(),
                top_n_candidates,
                model_name,
            )
        )

    print(f"[Ranker] Total terms: {len(grouped)}")
    print(f"[Ranker] Pending terms: {len(tasks)}")
    print(f"[Ranker] Top candidates per term: {top_n_candidates}")
    print(f"[Ranker] Parallel workers: {max_workers}")

    if not tasks:
        print("[Ranker] Nothing to process.")
        return pd.DataFrame(results)

    completed_since_save = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one_group, task)
            for task in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Reranking terms",
        ):
            try:
                result = future.result()

            except Exception as error:
                print(f"[Ranker] Erro inesperado em tarefa paralela: {error}")

                result = empty_reranker_result(
                    term_original=None,
                    normalized_en=None,
                    reasoning=f"Erro inesperado em tarefa paralela: {error}",
                )

            results.append(result)
            completed_since_save += 1

            if completed_since_save >= save_every:
                partial_df = pd.DataFrame(results)
                partial_df.to_csv(output_path, index=False)

                print(f"[Ranker] Partial save: {len(results)} results saved.")

                completed_since_save = 0

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    print(f"[Ranker] Saved final results to: {output_path}")
    print(f"[Ranker] Total results: {len(results_df)}")

    return results_df