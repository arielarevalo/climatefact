# Hybrid Concept Extraction System

A sophisticated climate science concept extraction system that combines regex-based pattern matching with modern NLP techniques including Named Entity Recognition (NER) to extract and categorize climate-related concepts from3. **Low extraction quality**
   - Adjust `CONFIDENCE_THRESHOLDS` in `config.py`
   - Add domain-specific patterns
   - Consider using a more advanced spaCy model

4. **Input format validation errors**
   - Ensure your input file is in valid JSONL format (one JSON object per line)
   - Each line should have `"id"` and `"text"` fields
   - Test with a small sample first to verify formatentific text.

## Features

- **Hybrid Extraction**: Combines regex patterns, spaCy NER, NLTK NER, Transformers NER, and EntityRuler
- **Climate Science Focus**: Specialized patterns for climate science terminology
- **Multiple Output Formats**: Both traditional regex-based and modern hybrid indices
- **Comprehensive Statistics**: Detailed extraction metrics and performance data
- **Flexible Configuration**: Easily customizable patterns and thresholds
- **Data Validation**: Built-in JSONL file validation

## Installation

### Prerequisites

- Python 3.10+ (tested with Python 3.10.7)
- Virtual environment (recommended)

### Setup

1. **Clone/Download the project**
   ```bash
   cd concept_extraction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**
   ```bash
   # Download spaCy English model
   python -m spacy download en_core_web_sm
   
   # NLTK data will be downloaded automatically on first run
   # Transformers models will be downloaded automatically if needed
   ```

**Optional GPU Acceleration:**
For faster processing with transformers models, you can install PyTorch with GPU support:
```bash
# For CUDA (check your CUDA version)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# For CPU-only (default behavior)
# No additional installation needed
```

## Usage

### Basic Usage

```bash
# Using hybrid extraction (recommended)
python scripts/build_concept_index.py --input_file data.jsonl --output_file concept_index.json --method hybrid

# Using regex-only extraction
python scripts/build_concept_index.py --input_file data.jsonl --output_file concept_index.json --method regex

# With verbose logging
python scripts/build_concept_index.py --input_file data.jsonl --output_file concept_index.json --method hybrid --verbose
```

### Input Format

Input files must be in JSONL format (one JSON object per line):

```json
{"id": "unique_sentence_id", "text": "The sentence text to process..."}
{"id": "another_id", "text": "Another sentence for concept extraction."}
```

### Data Validation

Your input files should be in JSONL format with the following structure. The system will validate the format automatically during processing, but you can verify your data format by testing with a small sample first.

### Setup Script

For quick setup and demo, use the provided setup script:

```bash
# Make the script executable
chmod +x setup_and_run.sh

# Run setup only (installs models and dependencies)
./setup_and_run.sh

# Run with demo data (setup + demo extraction)
./setup_and_run.sh --demo

# Show help
./setup_and_run.sh --help
```

The setup script will:
- Check that you're in a virtual environment
- Install spaCy models automatically
- Download NLTK data
- Optionally run a demonstration with sample climate science data

## Output Structure

The system generates a comprehensive concept index with two main components:

### 1. Regex Index (Traditional)

```json
{
  "regex_index": {
    "CLIMATE_VAR": {
      "temperature": [
        {
          "concept": "temperature",
          "sentences": ["sent_001", "sent_005"]
        }
      ]
    }
  }
}
```

### 2. Hybrid Index (Enhanced)

```json
{
  "hybrid_index": {
    "CLIMATE_VAR": {
      "global temperature": {
        "sentences": [
          {
            "sentence_id": "sent_001",
            "confidence": 0.95,
            "source": "spacy",
            "start": 15,
            "end": 32
          }
        ],
        "total_occurrences": 1,
        "unique_sentences": 1,
        "avg_confidence": 0.95,
        "sources": ["spacy"]
      }
    }
  }
}
```

### 3. Statistics

```json
{
  "statistics": {
    "total_sentences": 1000,
    "extraction_methods": {
      "spacy": 2500,
      "nltk": 1800,
      "regex": 3200
    },
    "concept_types": {
      "CLIMATE_VAR": 450,
      "EMISSION": 380
    },
    "avg_concepts_per_sentence": 4.2,
    "max_concepts_per_sentence": 15
  }
}
```

## Concept Categories

The system categorizes concepts into the following types:

- **AGENCIES**: Organizations (IPCC, IEA, etc.)
- **CLIMATE_VAR**: Climate variables (temperature, precipitation, etc.)
- **EMISSION**: Emissions and greenhouse gases (CO₂, CH₄, etc.)
- **POLICY**: Policy terms (adaptation, mitigation, etc.)
- **REPORTS**: Report identifiers (AR6, SR1.5, etc.)
- **SCENARIO**: Scenarios and pathways (SSP, RCP, etc.)
- **TECHNOLOGY**: Technologies (CCS, renewable energy, etc.)
- **IMPACT**: Climate impacts (sea level rise, heatwaves, etc.)
- **ECONOMIC**: Economic terms (GDP, costs, etc.)
- **TEMPORAL**: Time-related terms (dates, periods, etc.)
- **LOCATION**: Geographic locations
- **PERSON**: People and names
- **QUANTITATIVE**: Numbers and measurements
- **SOCIAL**: Social aspects
- **MODELS**: Climate models (CMIP6, MAGICC, etc.)

## Configuration

The system is highly configurable through the `config.py` file:

### Key Configuration Options

- **SPACY_MODEL**: spaCy model to use (default: "en_core_web_sm")
- **TRANSFORMERS_NER_MODEL**: Hugging Face model for NER
- **CONFIDENCE_THRESHOLDS**: Minimum confidence scores for each method
- **EXTRACTION_PRIORITY**: Priority order for overlapping concepts
- **BATCH_SIZE**: Processing batch size
- **LOG_LEVEL**: Logging verbosity

### Adding Custom Patterns

1. **Regex Patterns**: Add to `REGEX_MAP` in `config.py`
2. **EntityRuler Patterns**: Add to `ENTITY_RULER_PATTERNS`
3. **Domain-Specific Patterns**: Add to `DOMAIN_SPECIFIC_PATTERNS`

Example:
```python
REGEX_MAP["new_concept"] = re.compile(r'\bnew_pattern\b', re.IGNORECASE)
PATTERN_SCHEMA["new_concept"] = "CUSTOM_CATEGORY"
```

## Architecture

### Extraction Methods

1. **EntityRuler**: Exact pattern matching with spaCy
2. **Regex**: Traditional regular expression matching
3. **spaCy NER**: Statistical named entity recognition
4. **NLTK NER**: Rule-based named entity recognition
5. **Transformers NER**: Deep learning-based NER
6. **Domain-Specific**: Custom climate science patterns

### Concept Merging

The system intelligently merges overlapping concepts based on:
- Source priority (EntityRuler > Domain-Specific > spaCy > Transformers > NLTK > Regex)
- Span length (longer spans preferred)
- Confidence scores (higher confidence preferred)

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests
5. Submit a pull request

### Code Structure

```
concept_extraction/
├── scripts/
│   └── build_concept_index.py         # Main extraction script
├── config.py                          # Configuration and constants
├── setup_and_run.sh                   # Setup automation script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**
   - The system downloads NLTK data automatically on first run
   - If issues persist, manually download: `python -c "import nltk; nltk.download('all')"`

3. **Memory issues with large files**
   - Reduce `BATCH_SIZE` in `config.py`
   - Process files in smaller chunks

4. **Low extraction quality**
   - Adjust `CONFIDENCE_THRESHOLDS` in `config.py`
   - Add domain-specific patterns
   - Consider using a more advanced spaCy model

### Performance Tips

- Use hybrid extraction for best results (combines multiple NLP methods)
- Adjust batch size in `config.py` based on available memory
- For large datasets, consider processing in chunks
- GPU acceleration with PyTorch can significantly speed up transformers models
- Pre-validate input files to avoid processing errors
- Use regex-only method for faster processing when NER accuracy isn't critical

## System Status & Notes

**Current Status:** ✅ Active and Functional

**What's Working:**
- ✅ Hybrid concept extraction (regex + NER)
- ✅ Multiple NLP backends (spaCy, NLTK, Transformers)
- ✅ Climate science specialized patterns
- ✅ Automated setup script with demo
- ✅ Comprehensive statistics and reporting
- ✅ Flexible configuration system

**Recent Updates (June 2025):**
- Updated Python version requirement to 3.10+
- Improved setup script with better error handling
- Enhanced documentation and troubleshooting
- Removed dependency on non-existent validation script
- Added optional GPU acceleration information

## Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

### Areas for Contribution

- Additional regex patterns for climate science terms
- New extraction methods
- Performance optimizations
- Documentation improvements
- Test coverage expansion

## Acknowledgments

- Built with spaCy, NLTK, and Transformers
- Climate science terminology based on IPCC reports
- Inspired by the need for automated climate literature analysis
