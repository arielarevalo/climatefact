# Climate Fact-Checking Evaluation System

This directory contains a comprehensive evaluation system for assessing the quality of retrieval, NLI (Natural Language Inference), and the complete climate fact-checking pipeline using standard information retrieval and classification metrics.

## Structure

```
evals/
├── metrics/               # Evaluation metrics and evaluators
│   ├── __init__.py
│   ├── retrieval_metrics.py    # Core retrieval metrics implementation
│   ├── nli_metrics.py          # NLI classification metrics
│   ├── nli_evaluator.py        # NLI evaluation pipeline
│   ├── nli_evaluator_new.py    # Updated NLI evaluator
│   ├── evaluator.py            # Retrieval evaluation pipeline
│   └── prompts/                # Evaluation prompts
├── reports/               # Generated evaluation reports
├── logs/                  # Evaluation logs
├── evaluation_config.json # Configuration file
├── run_full_nli_evaluation.py      # NLI evaluation runner
├── run_full_pipeline_evaluation.py # Full pipeline evaluation runner
└── README.md             # This file
```

## Metrics Implemented

### Retrieval Quality Metrics
- **Recall@k**: Fraction of relevant items retrieved in top-k results
- **Precision@k**: Fraction of retrieved items that are relevant in top-k results

### Ranking Quality Metrics
- **MRR@k**: Mean Reciprocal Rank - reciprocal of rank of first relevant item
- **nDCG@k**: Normalized Discounted Cumulative Gain - measures ranking quality with position-based discounting

### NLI Classification Metrics
- **Accuracy**: Overall classification accuracy across all labels
- **Precision**: Per-class and macro/micro precision scores
- **Recall**: Per-class and macro/micro recall scores
- **F1-Score**: Per-class and macro/micro F1 scores
- **Confusion Matrix**: Detailed error analysis by label pair

## Usage

### 1. Full Pipeline Evaluation

Run the complete retrieval pipeline evaluation:

```bash
cd /Users/ariel/workspace/ci0129/proyecto/climatefact/evals
python run_full_pipeline_evaluation.py
```

### 2. NLI Evaluation

Test the NLI model performance:

```bash
python run_full_nli_evaluation.py
```

### 3. Configuration

Edit `evaluation_config.json` to customize:
- K values for evaluation (default: [1, 3, 5, 10])
- Data file paths (points to `/Users/ariel/workspace/ci0129/proyecto/data/`)
- Output directories
- Retrieval methods to evaluate (regex, dense, hybrid)
- Evaluation settings (logging, report generation)

## Data Requirements

### Gold Standard Dataset
The system expects a gold standard file (`data/evaluation/gold_set.jsonl`) with entries containing:
- `claim`: The query text
- `evidence`: ID of the relevant passage
- `id`: Unique identifier for the entry
- `entailment`: Classification label (ENTAILMENT, CONTRADICTION, NEUTRAL)

Example entry:
```json
{
  "id": "c3131090-32a0-4500-a13f-d3f9a3a70fa4",
  "claim": "There's even more proof that humans are behind the rise in extreme weather events.",
  "evidence": "0ba3237d-0621-4876-b813-6a8caa86cfad",
  "entailment": "ENTAILMENT"
}
```

### Passages Dataset
Passages file (`data/passages.jsonl`) with entries containing:
- `id`: Unique passage identifier
- `text`: Passage content
- `embedding`: Optional embedding vector
- `source`: Source metadata

Example entry:
```json
{
  "id": "d33fa30e-cfc3-4bf9-84bb-605d917d2e56",
  "embedding": [0.018790819, ..., 0.025113294],
  "text": "Climate change is caused by human activities...",
  "source": {"name": "IPCC_AR6_SYR_FullVolume_19.md", "page": 20}
}
```

## Output

### Console Output
The evaluation prints a comprehensive summary including:
- Individual method performance
- Metric comparisons across methods
- Performance rankings
- NLI classification results with per-class metrics

### Report Files
Generated in the `reports/` directory:
- `retrieval_evaluation_YYYYMMDD_HHMMSS.json`: Retrieval results
- `nli_evaluation_YYYYMMDD_HHMMSS.json`: NLI classification results
- `full_pipeline_evaluation_YYYYMMDD_HHMMSS.json`: Complete pipeline results
- `*_detailed.json`: Detailed analysis with recommendations

### Log Files
Generated in the `logs/` directory:
- `nli_evaluation.log`: NLI evaluation logs
- `full_pipeline_evaluation.log`: Complete pipeline evaluation logs

### Report Contents
- Evaluation metadata (timestamp, dataset size, methods)
- Individual method results for all k values
- Comparative analysis across methods
- Performance recommendations
- Confusion matrices and error analysis (for NLI)

## Extending the System

### Adding New Retrieval Methods

1. Implement the retrieval method following the existing pattern
2. Add evaluation logic in `FullPipelineEvaluationRunner.run_comprehensive_evaluation()`
3. Update the configuration file to enable the new method

### Adding New Metrics

1. Add metric calculation method to `RetrievalMetrics` or `NLIMetrics` class
2. Update evaluation methods to include the new metric
3. Update report generation to include the new metric

### Adding New NLI Models

1. Implement the model interface in the NLI evaluator
2. Update `NLIEvaluator` to support the new model
3. Configure the model parameters in the evaluation config

## Example Results

### Retrieval Evaluation
```
RETRIEVAL EVALUATION RESULTS SUMMARY
======================================================

REGEX_RETRIEVAL:
---------------
Top-1 Results:
  Recall@1:    0.2500
  Precision@1: 0.2500
  MRR@1:       0.2500
  nDCG@1:      0.2500

Top-5 Results:
  Recall@5:    0.6500
  Precision@5: 0.1300
  MRR@5:       0.3750
  nDCG@5:      0.4250
```

### NLI Evaluation
```
NLI EVALUATION RESULTS SUMMARY
======================================================

Overall Accuracy: 0.8500

Per-Class Results:
ENTAILMENT:
  Precision: 0.8200
  Recall:    0.8700
  F1-Score:  0.8400

CONTRADICTION:
  Precision: 0.8800
  Recall:    0.8200
  F1-Score:  0.8500

NEUTRAL:
  Precision: 0.8500
  Recall:    0.8600
  F1-Score:  0.8550
```

## Dependencies

The evaluation system requires:
- Standard Python libraries (json, logging, os, sys, pathlib, datetime)
- The existing climatefact retrieval and NLI modules
- LangGraph for workflow evaluation
- Access to the gold standard and passages datasets
- NLI model dependencies (transformers, torch, etc.)

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure all required data files exist in the specified paths
   - Check `/Users/ariel/workspace/ci0129/proyecto/data/evaluation/gold_set.jsonl`
   - Check `/Users/ariel/workspace/ci0129/proyecto/data/passages.jsonl`
2. **Import errors**: Check that the Python path includes the climatefact source directory
3. **No results**: Verify that the retrieval methods are finding matching passages
4. **NLI model errors**: Ensure the NLI model is properly loaded and configured
5. **Memory issues**: Large datasets may require sampling or batch processing

### Debugging

- Set log level to DEBUG in the configuration for verbose output
- Check the generated log files in the logs directory
- Use the individual evaluation scripts to isolate issues
- Verify data format matches expected schema

## Performance Considerations

- The evaluation processes all queries sequentially
- Large datasets may take significant time to evaluate
- Consider sampling for initial testing with large datasets
- Memory usage scales with the number of passages and queries
- NLI evaluation may be computationally intensive depending on model size
- Pipeline evaluation tests the complete workflow and may take longer
- Use background processing for long-running evaluations
