# ClimateFact: A RAG workflow for climate change fact verification

> Ariel Arévalo Alvarado, Andrik Acuña Castillo, Anthony Sánchez Ramírez, Erick Andrés Sibaja Li
>
> School of Computer Science and Informatics, University of Costa Rica

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-green.svg)](https://github.com/langchain-ai/langgraph)
[![CI](https://github.com/arielarevalo/climatefact/workflows/CI/badge.svg)](https://github.com/arielarevalo/climatefact/actions)

## Paper

**Download:** [English (PDF)](paper/climatefact_en.pdf) | [Español (PDF)](paper/climatefact_es.pdf)

### Abstract

This project presents a fact-checking workflow called ClimateFact that aims to evaluate climate change-related statements using trustworthy scientific knowledge bases. Motivated by the prevalence of climate misinformation and the lack of accessible, domain-specific verification tools, ClimateFact uses Retrieval-Augmented Generation (RAG) to identify and explain inconsistencies between claims and authoritative sources such as the IPCC reports.

The workflow generates a knowledge base using embeddings and regex-based indexing informed by domain-specific ontologies. It then retrieves relevant passages using a multi-step retrieval and re-ranking pipeline, followed by a natural language inference (NLI) model to assess the factual alignment of claims. Finally, an LLM generates grounded explanations marking contradictions with references.

### Citation

```bibtex
@inproceedings{arevalo2025climatefact,
  author    = {Ar\'{e}valo Alvarado, Ariel and Acu\~{n}a Castillo, Andrik
               and S\'{a}nchez Ram\'{i}rez, Anthony and Sibaja Li, Erick Andr\'{e}s},
  title     = {{ClimateFact}: A {RAG} workflow for climate change fact verification},
  institution = {School of Computer Science and Informatics, University of Costa Rica},
  year      = {2025}
}
```

## Architecture

The system uses a 4-stage LangGraph pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIMATE FACT CHECKER                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │   1. SENTENCE SEGMENTATION            │
          │   Split text into individual claims   │
          └───────────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │   2. KNOWLEDGE RETRIEVAL              │
          │   Search climate science databases    │
          │   (Hybrid: Regex + Dense Embeddings)  │
          └───────────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │   3. CONTRADICTION DETECTION          │
          │   NLI model classifies relationships  │
          │   (ENTAILMENT/CONTRADICTION/NEUTRAL)  │
          └───────────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │   4. REPORT GENERATION                │
          │   Synthesize findings with citations  │
          └───────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | LangGraph 0.4.8, LangChain Core 0.3.66 |
| **NLI Model** | Azure AI Inference (deberta-v3-large) |
| **Embeddings** | Azure OpenAI (text-embedding-3-large) |
| **Retrieval** | Hybrid (Regex + Dense Vector Search) |
| **UI** | Streamlit 1.45.1 |
| **Language** | Python 3.12+ |
| **Tracing** | LangSmith 0.3.45 |
| **NLP** | spaCy 3.8.7, NLTK 3.9.1 |

## Project Structure

```
climatefact/
├── paper/                       # Academic paper (IEEE format)
│   ├── es/                      # Spanish version
│   │   ├── main.tex
│   │   ├── references.bib
│   │   └── sections/            # Per-section .tex files
│   ├── en/                      # English version
│   │   ├── main.tex
│   │   ├── references.bib
│   │   └── sections/
│   ├── IEEEtran.cls             # Shared IEEE class file
│   ├── build.sh                 # Compiles both PDFs
│   ├── climatefact_es.pdf       # Pre-built Spanish PDF
│   └── climatefact_en.pdf       # Pre-built English PDF
├── climatefact/                 # Main application package
│   ├── src/climatefact/
│   │   ├── workflows/           # LangGraph workflows
│   │   │   └── contradiction_detection/
│   │   ├── etl/                 # Data processing pipelines
│   │   ├── app.py               # Streamlit application
│   │   └── cli.py               # Command-line interface
│   ├── evals/                   # Evaluation system
│   └── tests/                   # Test suite
├── concept_extraction/          # Concept extraction utilities
├── sentence_extraction/         # Sentence segmentation tools
├── pdf_extraction/              # PDF-to-markdown conversion
├── infra/                       # Infrastructure configs
└── docs/                        # Documentation & data
```

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry 2.0+
- Azure OpenAI API access (embeddings)
- Azure AI Inference endpoint (NLI model)

### Installation

```bash
git clone https://github.com/arielarevalo/climatefact.git
cd climatefact/climatefact
poetry install
cp .env.example .env   # Edit with your API keys
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LANGSMITH_API_KEY` | LangSmith API key for tracing |
| `LANGSMITH_TRACING` | Enable tracing (set to `true`) |
| `LANGSMITH_PROJECT` | Project name (e.g., `climatefact`) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_INFERENCE_CREDENTIAL` | Azure AI Inference credential |
| `AZURE_INFERENCE_ENDPOINT` | Azure AI Inference endpoint URL |

### Running the Application

```bash
# Streamlit Web Interface
poetry run streamlit run src/climatefact/app.py

# LangGraph Development Server
poetry run langgraph dev
```

## Evaluation

The evaluation system measures retrieval quality (Recall@k, Precision@k, MRR@k, nDCG@k) and NLI classification performance (per-class F1, confusion matrix). See the [paper](paper/climatefact_en.pdf) for detailed results and analysis.

```bash
cd climatefact/evals
poetry run python run_full_pipeline_evaluation.py
```

## Development

```bash
# Lint and format
poetry run ruff check --fix .
poetry run ruff format .

# Type checking
uvx ty check

# Run tests
poetry run pytest
```

## Building the Paper

Requires a LaTeX distribution with `pdflatex` and `bibtex` (e.g., TeX Live).

```bash
cd paper
./build.sh
```

This produces `paper/climatefact_es.pdf` and `paper/climatefact_en.pdf`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Ariel Arévalo Alvarado

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Powered by Azure AI services for embeddings and natural language inference
- Climate science data sourced from IPCC AR6 Synthesis Report
- NLI model: DeBERTa-v3-large fine-tuned on MNLI dataset
