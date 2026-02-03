---
name: scientific-paper-writing
description: Write and improve scientific deep learning papers with fluent, precise English. Use when writing LaTeX papers, editing academic text, improving clarity and flow, or ensuring proper technical terminology and mathematical notation.
---

# Scientific Deep Learning Paper Writing

## Core Principles

### 1. Clarity and Precision
- Use active voice when possible: "We introduce" not "A method is introduced"
- Be specific: "achieves 99.2% accuracy" not "achieves high accuracy"
- Avoid vague qualifiers: prefer "substantially" or "marginally" over "very" or "quite"

### 2. Technical Accuracy
- Use standard deep learning terminology consistently
- Define notation on first use: "Let $\mathbf{x} \in \mathbb{R}^d$ denote..."
- Maintain consistent variable names throughout

### 3. Academic Tone
- Present contributions confidently but avoid overclaiming
- Acknowledge limitations explicitly
- Use hedging appropriately: "suggests" vs "proves", "may" vs "will"

## Paper Structure Guidelines

### Abstract (150-250 words)
Structure: Problem → Method → Results → Implications

**Pattern:**
```
[Context/Problem]. [Prior limitation]. We [method/contribution].
[Key technique 1] and [Key technique 2]. [Main finding 1].
[Main finding 2]. [Implication/Conclusion].
```

**Example:**
> Token cramming compresses sequences into learned embeddings with near-perfect reconstruction, but prior work used fixed token budgets and 99% accuracy thresholds. We introduce progressive cramming: sequentially adding tokens until 100% accuracy, with activation alignment and low-dimensional projection. This achieves reliable perfect reconstruction and enables precise measurement of per-model information gain. Analysis reveals optimization trajectories occupy surprisingly low-dimensional manifolds. Attention analysis shows compression tokens hijack 40-80% of attention mass in intermediate layers. However, evaluation on HellaSwag and ARC exposes a fundamental limit: models fail entirely, even with decompressed tokens in context.

### Introduction
**Structure:**
1. **Hook** (1-2 sentences): Compelling question or observation
2. **Context** (2-3 paragraphs): Related work, open questions
3. **Gap** (1 paragraph): What's missing or unclear
4. **Contribution** (1 paragraph): What this work addresses
5. **Contributions list** (bulleted): Specific items

**Language patterns:**
- Opening: "How much information can...?", "Recent work on X has shown...", "A fundamental question remains..."
- Transitions: "However,", "This leaves open...", "We address this by..."
- Contributions: "We introduce...", "We demonstrate...", "We show that..."

### Methods Section
**Structure:**
- Problem formulation with clear notation
- Algorithmic description (enumerate or pseudocode)
- Implementation details (if needed)

**Mathematical notation:**
- Define all variables: "Let $\mathcal{M}$ be a model with parameters $\theta$"
- Use consistent fonts: $\mathbf{x}$ (vectors), $\mathcal{X}$ (sets), $\mathbb{R}$ (reals)
- Number equations that are referenced: `\begin{equation}...\end{equation}`

**Writing style:**
- "We optimize $\mathbf{e}$ to minimize $\mathcal{L}(\mathbf{e}; \mathbf{x})$"
- "The objective function is defined as:"
- "We initialize $\mathbf{e}^{(0)}$ randomly and update via..."

### Results Section
**Structure:**
- Main findings first
- Supporting analysis
- Ablation studies
- Failure cases or limitations

**Language patterns:**
- "We find that...", "Analysis reveals...", "Evaluation demonstrates..."
- "Surprisingly, ...", "In contrast, ...", "Notably, ..."
- "This suggests...", "This indicates...", "This implies..."

### Section Organization
```latex
% Use clear section labels
\section{Method}
\label{sec:method}

\subsection{Progressive Token Addition}
\label{sec:progressive}

% Use horizontal rules for major breaks
% =============================================================================
\section{Results}
% =============================================================================
```

## Common Language Patterns

### Describing Methods
- "We introduce [method name]: [brief description]"
- "Our approach [key characteristic] by [mechanism]"
- "To address [problem], we [solution]"

### Describing Results
- "We achieve [metric] of [value], [comparison]"
- "Evaluation on [dataset] demonstrates [finding]"
- "Analysis reveals [insight]"

### Making Claims
- Strong: "We demonstrate that...", "Our results show..."
- Moderate: "This suggests...", "These findings indicate..."
- Weak: "This may imply...", "One interpretation is..."

### Comparisons
- "In contrast to [prior work], we [difference]"
- "Unlike [method], our approach [advantage]"
- "Compared to [baseline], we achieve [improvement]"

## Common Pitfalls to Avoid

### Language Issues
- ❌ "We can see that..." → ✅ "We observe that..." or "Analysis shows..."
- ❌ "It is important to note..." → ✅ "Notably, ..." or "Critically, ..."
- ❌ "In order to..." → ✅ "To..."
- ❌ "Due to the fact that..." → ✅ "Because..." or "Since..."
- ❌ "A lot of" → ✅ "Many" or "Numerous"

### Technical Issues
- ❌ Inconsistent notation (mixing $\mathbf{x}$, $x$, $\vec{x}$)
- ❌ Undefined variables in equations
- ❌ Missing article ("the model" vs "model")
- ❌ Plural/singular mismatches ("the data are" vs "the dataset is")

### Structure Issues
- ❌ Vague contributions ("we improve performance")
- ❌ Missing motivation for design choices
- ❌ Results without interpretation
- ❌ Claims without evidence

## Editing Checklist

When reviewing or improving text:

1. **Clarity**
   - [ ] Each sentence has one clear idea
   - [ ] Technical terms defined on first use
   - [ ] Pronouns have clear antecedents

2. **Flow**
   - [ ] Paragraphs have clear topic sentences
   - [ ] Transitions connect ideas logically
   - [ ] Related ideas grouped together

3. **Precision**
   - [ ] Specific numbers instead of vague terms
   - [ ] Appropriate hedging (may/suggests vs demonstrates)
   - [ ] Claims match evidence strength

4. **Consistency**
   - [ ] Notation consistent throughout
   - [ ] Terminology consistent
   - [ ] Tense consistent (present for general facts, past for experiments)

5. **Grammar**
   - [ ] Subject-verb agreement
   - [ ] Proper articles (a/an/the)
   - [ ] Parallel structure in lists

## Examples

### Before (unclear)
> The method does well on the task. It uses a neural network to process the data. The results are good.

### After (clear)
> Our method achieves 94.2% accuracy on HellaSwag, a 3.5% improvement over the baseline. We process input sequences through a transformer encoder, then apply a learned compression head. Evaluation demonstrates that this approach preserves semantic information while reducing sequence length by 8×.

### Before (vague claim)
> The model learns good representations.

### After (specific)
> Analysis of learned embeddings reveals that the model captures semantic structure: similar concepts cluster in embedding space, with cosine similarity >0.85 for semantically related pairs.

### Before (weak structure)
> We tried different things. Some worked better. The best one was used.

### After (clear method)
> We ablate three design choices: activation alignment, low-dimensional projection, and progressive token addition. Table 2 shows that all three components contribute: removing any one reduces accuracy by 5-12%. We adopt the full configuration for all experiments.
