# Scientific Paper Writing - Reference Guide

## Advanced LaTeX Patterns

### Theorem Environments
```latex
\begin{theorem}
If condition $P$ holds, then result $Q$ follows.
\end{theorem}
\begin{proof}
Proof text here.
\end{proof}
```

### Algorithm Pseudocode
```latex
\begin{algorithm}[t]
\caption{Progressive Cramming}
\label{alg:progressive}
\begin{algorithmic}[1]
\REQUIRE Sequence $\mathbf{x} = (x_1, \ldots, x_n)$
\ENSURE Compression embeddings $\mathbf{e}^{(1)}, \ldots, \mathbf{e}^{(n)}$
\STATE Initialize $\mathbf{e}^{(0)} \sim \mathcal{N}(0, \sigma^2 I)$
\FOR{$k = 1$ to $n$}
    \STATE Set target $\mathbf{x}^{(k)} = (x_1, \ldots, x_k)$
    \STATE Optimize $\mathbf{e}^{(k)} = \arg\min_{\mathbf{e}} \mathcal{L}(\mathbf{e}; \mathbf{x}^{(k)})$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### Multi-line Equations
```latex
\begin{align}
\mathcal{L}(\mathbf{e}; \mathbf{x}) &= -\sum_{i=1}^{n} \log p(x_i \mid \mathbf{e}, x_{<i}) \\
&= -\sum_{i=1}^{n} \log \frac{\exp(s(x_i))}{\sum_{v \in \mathcal{V}} \exp(s(v))}
\end{align}
```

## Deep Learning Terminology

### Standard Terms
- **Model architecture**: "transformer", "encoder-decoder", "autoregressive"
- **Training**: "optimize", "minimize loss", "fine-tune", "train from scratch"
- **Evaluation**: "evaluate on", "achieve [metric]", "outperform"
- **Representations**: "embeddings", "hidden states", "activations"
- **Attention**: "attention weights", "attention patterns", "attention mass"

### Common Phrases
- "We freeze the base model and optimize only [component]"
- "The model achieves [metric] of [value] on [dataset]"
- "Analysis of [component] reveals [finding]"
- "We ablate [component] to assess its contribution"
- "Evaluation demonstrates [capability/limitation]"

## Section-Specific Patterns

### Related Work
**Structure:**
1. Directly related work (most similar)
2. Broader related areas
3. Positioning of this work

**Language:**
- "Our work builds upon [prior work], which [contribution]"
- "In contrast, [other work] focuses on [different aspect]"
- "We extend [prior method] by [novel contribution]"
- "Unlike [prior work], we [key difference]"

### Experimental Setup
**Required elements:**
- Datasets with statistics
- Model configurations
- Training details (optimizer, learning rate, epochs)
- Evaluation metrics
- Hardware/compute

**Pattern:**
```
We evaluate on [dataset] containing [statistics].
We use [model] with [configuration].
Training uses [optimizer] with learning rate [lr] for [epochs] epochs.
We report [metrics] averaged over [runs] random seeds.
```

### Discussion
**Structure:**
- Interpret main findings
- Connect to broader implications
- Acknowledge limitations
- Suggest future directions

**Language:**
- "These findings suggest that..."
- "One interpretation is that..."
- "This has implications for..."
- "A limitation of our approach is..."
- "Future work could explore..."

## Citation Patterns

### Introducing Work
- "Recent work by \citet{author2024} demonstrates..."
- "\citet{author2024} show that..."
- "Following \citet{author2024}, we..."

### Comparing Work
- "In contrast to \citet{author2024}, we..."
- "Unlike \citet{author2024}, our method..."
- "While \citet{author2024} focus on X, we address Y"

### Building on Work
- "We extend the approach of \citet{author2024} by..."
- "Building on \citet{author2024}, we introduce..."
- "Our work is inspired by \citet{author2024}, who..."

## Common Sentence Structures

### Problem Statement
- "[Phenomenon] has been shown to [observation], but [limitation]"
- "While [prior work] achieves [result], [gap] remains"
- "[Question] is a fundamental challenge in [field]"

### Method Description
- "To address [problem], we [approach]"
- "Our key insight is that [observation] enables [solution]"
- "We propose [method], which [mechanism]"

### Result Presentation
- "We find that [observation] across [conditions]"
- "Evaluation on [dataset] demonstrates [finding]"
- "Analysis reveals [insight] that [implication]"

## Transition Phrases

### Between Paragraphs
- "However, ..." (contrast)
- "Moreover, ..." (addition)
- "In particular, ..." (specification)
- "Notably, ..." (emphasis)
- "Critically, ..." (importance)
- "Surprisingly, ..." (unexpected finding)

### Within Paragraphs
- "Specifically, ..."
- "For example, ..."
- "In contrast, ..."
- "Furthermore, ..."
- "Additionally, ..."

## Hedging and Confidence

### Strong Claims (with evidence)
- "We demonstrate that..."
- "Our results show..."
- "Evaluation confirms..."
- "Analysis proves..."

### Moderate Claims
- "This suggests that..."
- "These findings indicate..."
- "We observe that..."
- "This implies..."

### Tentative Claims
- "This may indicate..."
- "One interpretation is..."
- "It is possible that..."
- "This could suggest..."

## Mathematical Writing

### Defining Notation
```latex
% Good: Clear and complete
Let $\mathcal{M}: \mathcal{V}^* \to \mathbb{R}^{|\mathcal{V}|}$
denote an autoregressive language model, where $\mathcal{V}$
is the vocabulary and $\mathcal{V}^*$ denotes sequences over $\mathcal{V}$.

% Bad: Unclear
Let $M$ be a model.
```

### Describing Equations
- "Equation (1) defines the objective function as..."
- "The loss in Equation (2) consists of two terms:..."
- "We minimize Equation (3) with respect to $\theta$"

### Algorithmic Descriptions
- Use numbered lists for step-by-step procedures
- Use "We [action]" for active descriptions
- Use "The algorithm [does]" for passive descriptions

## Figure and Table Captions

### Figure Captions
- Start with capital letter
- End with period
- Explain what the figure shows, not just what it is
- Reference specific elements if needed

**Good:**
> Figure 2: Attention patterns across layers. Compression tokens (positions 0-3) capture 40-80% of attention mass in layers 8-16, regardless of sequence length. This suggests a fixed attention hijacking mechanism.

**Bad:**
> Figure 2: Attention patterns.

### Table Captions
- Similar structure to figures
- May include key statistics in caption
- Reference specific rows/columns if relevant

**Good:**
> Table 1: Performance on HellaSwag and ARC. Progressive cramming achieves perfect reconstruction but fails on downstream tasks, indicating that reconstruction does not imply semantic preservation.

## Common Errors

### Grammar
- ❌ "The data shows..." → ✅ "The data show..." (data is plural)
- ❌ "Less parameters" → ✅ "Fewer parameters"
- ❌ "Different than" → ✅ "Different from"
- ❌ "Comprised of" → ✅ "Comprises" or "Composed of"

### Style
- ❌ "In this paper, we..." → ✅ "We..." (obvious from context)
- ❌ "It should be noted that..." → ✅ "Notably, ..."
- ❌ "As can be seen from..." → ✅ "As shown in..." or "Figure X shows..."

### Technical
- ❌ "The neural network learns..." → ✅ "The model learns..." or "Training learns..."
- ❌ "We train the model on data" → ✅ "We train the model on [dataset name]"
- ❌ "Good results" → ✅ "High accuracy" or specific metric
