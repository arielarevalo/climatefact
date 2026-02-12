# Climate Fact Checker

AI-powered contradiction detection in climate-related statements using RAG and NLI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-green.svg)](https://github.com/langchain-ai/langgraph)
[![CI](https://github.com/arielarevalo/climatefact/workflows/CI/badge.svg)](https://github.com/arielarevalo/climatefact/actions)

## What It Does

Climate Fact Checker analyzes text for contradictions against established climate science. Given a statement like:

**Input:**
```
"The Earth is getting colder due to increased solar activity."
```

**Output:**
```
CONTRADICTION DETECTED
Statement: "The Earth is getting colder due to increased solar activity."
Evidence: "Global temperatures have increased by 1.1°C since pre-industrial times." (IPCC AR6)
Confidence: 0.94
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
├── climatefact/                  # Main package
│   ├── src/climatefact/
│   │   ├── workflows/           # LangGraph workflows
│   │   │   └── contradiction_detection/
│   │   │       ├── contradiction_detection/  # Main workflow
│   │   │       ├── retrieval/                # Knowledge retrieval subgraph
│   │   │       ├── generation/               # Report generation subgraph
│   │   │       └── stores/                   # Data storage utilities
│   │   ├── etl/                 # Data processing pipelines
│   │   │   ├── gold_candidate_building/      # Build evaluation datasets
│   │   │   └── knowledge_building/           # Build knowledge bases
│   │   ├── app.py               # Streamlit application
│   │   └── cli.py               # Command-line interface
│   ├── evals/                   # Evaluation system
│   │   ├── metrics/             # Retrieval & NLI metrics
│   │   ├── reports/             # Generated evaluation reports
│   │   └── run_full_*.py        # Evaluation runners
│   └── tests/                   # Test suite
├── concept_extraction/          # Concept extraction utilities
├── sentence_extraction/         # Sentence segmentation tools
├── pdf_extraction/              # PDF-to-markdown conversion
├── infra/                       # Infrastructure configs
│   └── model/                   # Model deployment
└── docs/                        # Documentation & data
    ├── IPCC_AR6_SYR_FullVolume_markdown/
    └── test_pages/
```

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Poetry 2.0+ (dependency management)
- Azure OpenAI API access (embeddings)
- Azure AI Inference endpoint (NLI model)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd climatefact
   ```

2. **Install Poetry 2** (if not already installed)
   ```bash
   # Install the latest version (Poetry 2.x)
   curl -sSL https://install.python-poetry.org | python3 -

   # Or upgrade existing Poetry to 2.x
   poetry self update

   # Verify Poetry 2.x is installed
   poetry --version  # Should show 2.x.x
   ```

3. **Install dependencies**
   ```bash
   cd climatefact
   poetry install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see table below)
   ```

5. **Install the package**
   ```bash
   poetry run python -m pip install -e .
   ```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | Yes |
| `LANGSMITH_TRACING` | Enable tracing (set to `true`) | Yes |
| `LANGSMITH_PROJECT` | Project name (e.g., `climatefact`) | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_INFERENCE_CREDENTIAL` | Azure AI Inference credential | Yes |
| `AZURE_INFERENCE_ENDPOINT` | Azure AI Inference endpoint URL | Yes |

### Running the Application

**Streamlit Web Interface:**
```bash
poetry run streamlit run src/climatefact/app.py
```

**LangGraph Development Server:**
```bash
poetry run langgraph dev
```

**Direct Python API:**
```python
from climatefact.workflows.contradiction_detection import graph

state = {
    "input_text": "Arctic ice is expanding rapidly.",
    "passages_jsonl_path": "data/passages.jsonl",
    "queries": [],
    "retrieved_data_for_queries": [],
    "contradiction_results": [],
    "report": ""
}

result = graph.invoke(state)
print(result["report"])
```

## Data Pipeline

### Input Format

The system expects a JSONL file with climate science passages. Each line contains:

```jsonl
{"text": "Global temperatures have increased by 1.1°C since pre-industrial times.", "source": "IPCC AR6", "url": "https://example.com"}
{"text": "Arctic sea ice extent has declined at a rate of 13% per decade.", "source": "NASA", "year": 2023}
{"text": "CO2 concentrations have increased from 280 ppm to over 420 ppm.", "source": "NOAA"}
```

**Required fields:**
- `text` (string): The passage content

**Optional fields:**
- `source` (string): Source attribution
- `url` (string): Reference URL
- `year` (int): Publication year
- `embedding` (array): Pre-computed embedding vector

### ETL Pipelines

**Knowledge Building:**
```bash
cd climatefact
poetry run python -m climatefact.etl.knowledge_building.main
```

**Gold Candidate Building (for evaluation):**
```bash
poetry run python -m climatefact.etl.gold_candidate_building.main
```

## Evaluation

The `climatefact/evals/` directory contains a comprehensive evaluation system:

### Metrics

**Retrieval Quality:**
- Recall@k, Precision@k (coverage and relevance)
- MRR@k (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)

**NLI Classification:**
- Accuracy, Precision, Recall, F1-Score
- Per-class metrics (ENTAILMENT, CONTRADICTION, NEUTRAL)
- Confusion matrix analysis

### Running Evaluations

**Full Pipeline Evaluation:**
```bash
cd climatefact/evals
poetry run python run_full_pipeline_evaluation.py
```

**NLI Model Evaluation:**
```bash
poetry run python run_full_nli_evaluation.py
```

**Retrieval Evaluation:**
```bash
poetry run python run_full_retrieval_evaluation.py
```

### Evaluation Data

Gold standard format (`data/evaluation/gold_set.jsonl`):
```json
{
  "id": "c3131090-32a0-4500-a13f-d3f9a3a70fa4",
  "claim": "There's even more proof that humans are behind the rise in extreme weather events.",
  "evidence": "0ba3237d-0621-4876-b813-6a8caa86cfad",
  "entailment": "ENTAILMENT"
}
```

Results are saved to `climatefact/evals/reports/` with timestamps.

## Development

### Quality Checks

```bash
# Lint and format
poetry run ruff check --fix .
poetry run ruff format .

# Type checking
uvx ty check

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/climatefact --cov-report=term-missing
```

### Poetry 2 Commands

```bash
# Activate virtual environment
poetry shell

# Add new dependencies
poetry add <package-name>

# Add development dependencies
poetry add --group dev <package-name>

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Build package
poetry build
```

### Extending Workflows

To add new processing nodes:

1. Create a node function in the appropriate workflow module
2. Update the graph definition in `__init__.py`
3. Add type definitions to `types.py`
4. Write unit and integration tests
5. Update evaluation scripts if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Ariel Arévalo Alvarado

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Powered by Azure AI services for embeddings and natural language inference
- Climate science data sourced from IPCC AR6 Synthesis Report
- NLI model: DeBERTa-v3-large fine-tuned on MNLI dataset
