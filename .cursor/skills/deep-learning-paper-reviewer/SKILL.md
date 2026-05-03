---
name: deep-learning-paper-reviewer
description: Reviews deep learning / machine learning scientific papers for technical correctness, clarity, novelty, experimental rigor, and reproducibility. Use when the user asks to review a paper, draft a NeurIPS/ICLR/ICML-style review, provide reviewer feedback on LaTeX, or prepare rebuttal responses.
---

# Deep Learning Scientific Paper Reviewer

## Scope

Use this skill to produce actionable, evidence-based reviews of ML/DL papers (typically LaTeX). Optimize for:
- Technical correctness and soundness
- Clear writing and well-scoped claims
- Experimental rigor and fair comparisons
- Reproducibility, limitations, and ethics

## Quick start (default workflow)

1. **Identify the venue style** (if unspecified, assume NeurIPS/ICLR/ICML norms).
2. **Extract claims**:
   - Main contributions (1–3 bullets).
   - Key empirical/theoretical claims (what exactly is asserted).
3. **Check technical soundness**:
   - Definitions and notation introduced before use.
   - Assumptions stated; edge cases addressed.
   - Objective, training setup, and evaluation metrics consistent with claims.
4. **Check experimental quality**:
   - Baselines: strong and relevant; properly tuned.
   - Ablations: isolate each contribution.
   - Reporting: mean/variance, seeds, compute budget, failure cases.
   - Fairness: same data, model size, prompt format, training steps, etc.
5. **Check writing/structure**:
   - Abstract matches actual results and scope.
   - Figures/tables readable; captions self-contained.
   - Related work accurately positioned (no straw-manning).
6. **Make recommendations**:
   - List the smallest set of changes that would most improve the paper.
   - Separate **major issues** vs **minor issues** vs **nits**.

If the user provides only a section (e.g., abstract), review only that scope and explicitly state what you did not evaluate.

## Output format (copy/paste template)

Provide the review in this structure unless the user requests a different one:

```markdown
## Summary
- [1–3 bullets: what the paper does and why it matters]

## Contributions
- [C1]
- [C2]
- [C3 if needed]

## Strengths
- [S1]
- [S2]

## Weaknesses / Concerns
- **Major**: [must-fix issues; tie each to a claim, result, or missing control]
- **Minor**: [nice-to-fix issues]
- **Nits**: [typos, phrasing, formatting]

## Questions for the authors
- [Q1]
- [Q2]

## Suggested experiments / analyses
- [E1: concrete setup, baseline, metric]
- [E2]

## Reproducibility checklist (quick)
- **Compute**: [hardware/time reported?]
- **Evaluation**: [metrics + protocol unambiguous?]

## Limitations & ethics
- [limits, failure modes, misuse risks]

## Overall assessment
- **Novelty**: [low/medium/high + why]
- **Soundness**: [low/medium/high + why]
- **Clarity**: [low/medium/high + why]
- **Confidence**: [low/medium/high + why]
```

## Review heuristics (what to look for)

### Claims vs evidence
- Flag over-claims (e.g., “solves”, “guarantees”, “significantly”) without proper evidence.
- Ensure the paper distinguishes **correlation** vs **causal** interpretations.
- Check whether improvements are robust across datasets/seeds/hyperparams.

### Common experimental pitfalls in DL papers
- Missing strong baselines or missing tuning for baselines.
- Inadequate ablations (multiple changes at once).
- Leakage (test set peeking; prompt/data contamination not discussed when relevant).
- Cherry-picked metrics/slices; no full distribution or failure cases.
- Comparison across different compute/model sizes without normalization.

### LaTeX / presentation hygiene (when reviewing source)
- Undefined references (`\ref{}`, `\cite{}`), inconsistent capitalization, missing captions.
- Notation drift (same symbol used for different things).
- Figures: unreadable fonts, axes unlabeled, missing units.
- Tables: unclear bolding; missing variance; missing dataset details.

## When asked to propose edits

If the user wants wording changes, propose:
- **A revised paragraph** (keep meaning, improve precision and flow).
- **A “before/after”** snippet when helpful.
- Avoid changing technical content unless explicitly asked.

## Tone and severity

- Prefer direct, constructive language.
- For each **Major** concern, include:
  - **what is wrong**
  - **why it matters**
  - **how to fix / what evidence would resolve it**

