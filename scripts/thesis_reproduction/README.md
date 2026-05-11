# Thesis reproduction

Reproduces a curated subset of paper numbers from *Progressive Cramming:
Reliable Token Compression and What It Reveals* (ICML 2026 submission), running
on the refactored library code. Each experiment is a thin shell wrapper that
launches a clean training run and then runs the comparison against the paper's
reported values.

The folder layout is designed to scale: today only `full_cramming/` is
populated, but `progressive/`, `low_dim/`, `attention_hijacking/` etc. plug
into the same pattern.

## Layout

```
scripts/thesis_reproduction/
├── README.md           # this file
├── train.py            # clean training entry-point (~80 LOC)
├── analyze.py          # loads saved Dataset, compares with expected.json
├── expected.json       # paper-anchor values per experiment
└── experiments/
    └── full_cramming/
        └── pythia_160m.sh
```

Each experiment:

* lives at a stable key `<family>/<model>` (e.g. `full_cramming/pythia_160m`),
* writes its results to `artifacts/thesis_reproduction/<family>/<model>/`,
* has a paired entry in `expected.json` with the paper anchor it targets.

## How to run

```bash
# 0. (Optional, for closest match to paper) install flash-attn.
#    Without it, the script auto-falls-back to PyTorch sdpa attention,
#    which is numerically very close in bf16 but not identical.
uv pip install flash-attn --no-build-isolation

# 1. Run the experiment (training + post-hoc comparison).
bash scripts/thesis_reproduction/experiments/full_cramming/pythia_160m.sh

# 2. Or compare an already-trained run against the paper.
uv run python scripts/thesis_reproduction/analyze.py \
    --experiment full_cramming/pythia_160m
```

Output of `analyze.py` is a `OUR | PAPER | DELTA | VERDICT` table where
`VERDICT` is one of:

* `OK` — within 2σ of the paper mean (or zero deviation for fixed-budget
  metrics like `compressed_tokens`),
* `WARN` — within 3σ but outside 2σ,
* `FAIL` — beyond 3σ; investigate the refactor.

## Currently included

| Experiment | Paper anchor | Samples | Time on A100 | Numbers we expect |
|---|---|---|---|---|
| `full_cramming/pythia_160m` | Table 11 (Appendix C) | 50 | ~5-10 min | 32 tokens, IG 105 ± 20, acc 0.684 ± 0.175 |
| `full_cramming/llama_3_2_1b` | Table 11 (Appendix C) | 10 | ~15-25 min | 512 tokens, IG 1965 ± 244, **acc 0.998 ± 0** |
| `progressive/smollm2_135m` | Table 13 (Appendix F) | 50 | ~15-30 min | **38.5 ± 14.1 tokens, IG 168 ± 66** |

Llama-3.2-1B is the strongest single-experiment correctness check: paper expects
accuracy 0.998 with zero std (essentially perfect reconstruction). If our refactor
preserves the cramming math, the run **must** hit ≥ 0.99. If it lands below 0.95
on Llama-3.2-1B, the refactor has a real bug — independent of sample-variance
explanations that complicate Pythia-160M's interpretation.

## To be added later

* `full_cramming/pythia_14b` — Table 1 main claim.
* `full_cramming/llama_32_1b` — Table 11 mid-size anchor.
* `progressive/smollm2_135m` — Table 13 anchor.
* `progressive/smollm2_17b` — Table 6 main claim.
* `low_dim/*` — Tables 2, 17 ablation.
* `attention_hijacking/smollm2_17b` — Table 3.
* `downstream/hellaswag_smollm2_17b` — Table 5/7 (requires eval-script refactor).
