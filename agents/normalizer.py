import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Tuple

from ollama import chat

from agents.prompts import NORMALIZER_PROMPT


DEFAULT_MODEL_NAME = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:latest"


def normalize_text(text: str) -> str:
    """
    Normalização simples de texto:
    - lowercase
    - remove acentos
    - remove tabs/quebras de linha
    - remove espaços duplicados
    """

    if not text:
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("utf-8")

    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def call_normalizer_llm(
    term: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """
    Chama o LLM via Ollama para normalizar um termo.
    """

    prompt = NORMALIZER_PROMPT.replace("{term_original}", term)

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

    try:
        return response["message"]["content"].strip()
    except (KeyError, TypeError):
        return ""


def parse_normalizer_response(response_text: str) -> Tuple[bool, str]:
    """
    Faz parse da resposta JSON do normalizador.
    Retorna:
    - is_valid_term
    - normalized_en
    """

    if not response_text:
        return False, ""

    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if not match:
            print(f"Nenhum JSON encontrado na resposta: {response_text}")
            return False, ""

        data = json.loads(match.group())

        is_valid = bool(data.get("is_valid_term", False))

        normalized = data.get("normalized_en", "")
        normalized = normalize_text(normalized)

        return is_valid, normalized

    except json.JSONDecodeError:
        print(f"Erro ao fazer parse do JSON: {response_text}")
        return False, ""


def normalize_term(
    term: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """
    Normaliza um único termo.
    """

    response_text = call_normalizer_llm(
        term=term,
        model_name=model_name,
    )

    is_valid, normalized_en = parse_normalizer_response(response_text)

    return {
        "term": term,
        "is_valid_term": is_valid,
        "normalized_en": normalized_en,
    }


def process_normalizer(
    input_file: str,
    output_file: str,
    term_column: str = "term",
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict]:
    """
    Processa um CSV de termos e gera um CSV normalizado.

    Entrada esperada:
    - coluna `term`

    Saída:
    - term
    - is_valid_term
    - normalized_en
    """

    results = []

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        if not reader.fieldnames or term_column not in reader.fieldnames:
            raise ValueError(f"CSV precisa ter coluna '{term_column}'")

        for i, row in enumerate(reader, start=1):
            term = row.get(term_column, "")

            print(f"[Normalizer] [{i}] Processando: {term}")

            result = normalize_term(
                term=term,
                model_name=model_name,
            )

            results.append(result)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "term",
            "is_valid_term",
            "normalized_en",
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[Normalizer] CSV salvo em: {output_path}")

    return results