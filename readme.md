# Mapeamento OMOP - Automação

Pipeline para normalização, busca semântica e reranqueamento de termos médicos utilizando LLMs, embeddings vetoriais e Milvus.

---

# Estrutura do Projeto

```bash
project/
│
├── agents/
│   ├── normalizer.py
│   ├── embeddings.py
│   ├── candidate_finder.py
│   ├── reranker.py
│   └── prompts.py
│
├── datasets/
├── docker/
├── outputs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Pipeline

1. Normalização dos termos médicos
2. Geração de embeddings
3. Busca híbrida no Milvus
4. Reranqueamento com LLM
5. Exportação dos resultados

---

# Tecnologias

* Python
* Ollama
* Sentence Transformers
* Milvus
* Docker
* Pandas

---

# Instalação

## Clone o projeto

```bash
git clone <repo>
cd projeto
```

## Ambiente virtual

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Instale as dependências

```bash
pip install -r requirements.txt
```

---

# Subindo o Milvus

```bash
cd docker
docker compose up -d
```

---

# Ingestão de Dados

```bash
python docker/ingest_milvus.py
```

---

# Executando o Pipeline

```bash
python main.py
```

---

# Outputs

Arquivos gerados em:

```bash
outputs/
```

Exemplos:

* normalized_terms.csv
* candidate_matches.csv
* final_reranked.csv

---

# Requirements

```txt
pandas
numpy
pymilvus
sentence-transformers
ollama
scikit-learn
tqdm
python-dotenv
```

---

# Objetivo

Automatizar o mapeamento terminológico médico utilizando IA Generativa, NLP e busca vetorial para melhorar consistência semântica e reduzir esforço manual.

---

# Autor

João Vitor Quintas dos Santos

