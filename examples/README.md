# Examples

Runnable example scripts demonstrating core ML Eval Toolkit functionality.

## Scripts

### `basic_evaluation.py`

Loads the sample dataset and runs BLEU + ROUGE metrics, printing per-sample and aggregated scores.

```bash
python examples/basic_evaluation.py
```

### `model_comparison.py`

Compares two metric configurations side-by-side (BLEU-4/ROUGE-L vs BLEU-2/ROUGE-1) and identifies the best config per metric.

```bash
python examples/model_comparison.py
```

### `prompt_regression.py`

Demonstrates the regression testing workflow: establish a baseline, run a new evaluation, and check for score degradations.

```bash
python examples/prompt_regression.py
```

## Sample Data

`sample_data.json` contains 5 input/output pairs covering translation, summarization, Q&A, and unit conversion tasks. Use it as a template for your own evaluation datasets.

### Dataset Format

```json
{
  "samples": [
    {
      "input": "Your prompt or question",
      "expected_output": "The reference/ground truth answer",
      "actual_output": "The model-generated answer"
    }
  ]
}
```
