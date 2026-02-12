# Climate Fact Checker

A Streamlit application for detecting contradictions in climate-related statements using advanced natural language processing and retrieval-augmented generation (RAG) techniques.

## Overview

Climate Fact Checker is an AI-powered tool that analyzes text for potential contradictions against established climate science knowledge. The application uses LangGraph workflows to:

1. **Segment text** into individual sentences
2. **Retrieve relevant passages** from climate science databases
3. **Detect contradictions** using natural language inference (NLI) models
4. **Generate comprehensive reports** highlighting any contradictions found

## Features

- 🔍 **Automated fact-checking** of climate-related statements
- 📝 **Sentence segmentation** for granular analysis
- 🔗 **Knowledge retrieval** from authoritative climate sources
- 🤖 **AI-powered contradiction detection** using NLI models
- 📊 **Detailed reporting** with source attribution
- 🌐 **Streamlit web interface** for easy interaction

## Architecture

The application is built using a modular LangGraph workflow architecture:

```
Input Text → Sentence Segmentation → Knowledge Retrieval → Contradiction Detection → Report Generation
```

### Core Components

- **Contradiction Detection Workflow**: Main orchestration workflow
- **Retrieval Subgraph**: Handles knowledge base queries
- **Generation Subgraph**: Creates detailed reports
- **ETL Modules**: Data processing for knowledge building

## Installation

### Prerequisites

- Python 3.12+
- Poetry 2.0+

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd proyecto
   ```

2. **Install Poetry 2** (if not already installed)
   ```bash
   # Install the latest version (Poetry 2.x)
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Or upgrade existing Poetry to 2.x
   poetry self update
   
   # Verify Poetry 2.x is installed
   poetry --version
   ```

3. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and endpoints
   ```

5. **Required Environment Variables**
   ```bash
   # LangSmith tracing
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=climatefact
   
   # Azure OpenAI for embeddings
   AZURE_OPENAI_API_KEY=your-azure-openai-key
   AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint
   
   # Azure AI for NLI model
   AZURE_INFERENCE_CREDENTIAL=your-azure-inference-key
   AZURE_INFERENCE_ENDPOINT=your-azure-inference-endpoint
   ```

## Usage

### Running the Streamlit App

```bash
poetry run streamlit run src/climatefact/app.py
```

### Using LangGraph Workflows Directly

```bash
poetry shell
python
```

```python
from climatefact.workflows.contradiction_detection import graph

# Example input
state = {
    "input_text": "The Earth is getting colder due to increased solar activity.",
    "passages_jsonl_path": "path/to/your/climate_passages.jsonl",
    "queries": [],
    "retrieved_data_for_queries": [],
    "contradiction_results": [],
    "report": ""
}

# Run the workflow
result = graph.invoke(state)
print(result["report"])
```

### JSONL File Format

Your passages JSONL file should contain one JSON object per line with at least a "text" field:

```jsonl
{"text": "Global temperatures have increased by 1.1°C since pre-industrial times.", "source": "IPCC AR6", "url": "https://example.com"}
{"text": "Arctic sea ice extent has declined at a rate of 13% per decade.", "source": "NASA", "year": 2023}
{"text": "CO2 concentrations have increased from 280 ppm to over 420 ppm.", "source": "NOAA"}
```

### API Development

The application can also be deployed as an API using LangGraph's built-in server. First you must install the application:

```bash
poetry run python -m pip install -e .
```

At which point you can start the LangGraph server:

```bash
poetry run langgraph dev
```

This will start the development server with all declared graphs available at the configured endpoint.

## Project Structure

```
src/climatefact/
├── workflows/
│   └── contradiction_detection/
│       ├── contradiction_detection/    # Main workflow
│       ├── retrieval/                 # Knowledge retrieval
│       ├── generation/                # Report generation
│       └── stores/                    # Data storage
├── etl/
│   ├── gold_candidate_building/       # ETL for candidate data
│   └── knowledge_building/            # ETL for knowledge base
└── app.py                            # Streamlit application
```

## Workflow Details

### 1. Sentence Segmentation
- Splits input text into individual sentences
- Handles edge cases for climate-specific terminology
- Prepares queries for the retrieval system

### 2. Knowledge Retrieval
- Searches authoritative climate science databases
- Returns relevant passages with source attribution
- Supports multiple retrieval strategies

### 3. Contradiction Detection
- Uses NLI models to identify contradictions
- Compares statements against retrieved passages
- Classifies relationships as: contradiction, entailment, or neutral

### 4. Report Generation
- Synthesizes findings into comprehensive reports
- Includes source citations and confidence scores
- Provides actionable insights for fact-checkers

## Development

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

# Check Poetry version (should be 2.x)
poetry --version
```

### Adding New Nodes

To extend the workflow with new processing nodes:

1. Create a new node function in the appropriate module
2. Update the workflow graph in `__init__.py`
3. Add necessary type definitions in `types.py`

### Configuration

The application uses:
- **LangGraph** for workflow orchestration
- **LangSmith** for observability and tracing
- **Azure OpenAI** for embeddings and language models
- **Streamlit** for the web interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install dependencies: `poetry install`
4. Make your changes
5. Run tests: `poetry run pytest`
6. Format code: `poetry run ruff format . && poetry run ruff check --fix .`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Powered by Azure AI services for natural language processing
