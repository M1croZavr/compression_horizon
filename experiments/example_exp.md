# Example Experiment Plan

## Hypothesis
Describe the hypothesis you want to test.

## Setup
- **Model**: e.g., `HuggingFaceTB/SmolLM2-1.7B`
- **Sequence lengths**: e.g., `[32, 64, 128]`
- **Loss type**: e.g., `cosine` / `cross_entropy`
- **Seeds**: e.g., `[42]`

## Training
```bash
PYTHONPATH=./src python scripts/jobs/run_training.py --dry
```

## Evaluation
```bash
PYTHONPATH=./src python scripts/jobs/run_evaluation.py --dry
```

## Expected outcome
Describe what you expect to observe.

## Results
_To be filled after running the experiment._
