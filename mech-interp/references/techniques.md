# Mechanistic Interpretability Techniques

This reference covers the core techniques used to understand how neural networks compute.

## Logit Lens

**What it does**: Projects intermediate layer representations through the unembedding matrix to see what tokens the model is "thinking about" at each layer.

**Intuition**: If the residual stream at layer N already predicts the final token, later layers may be refining rather than computing from scratch.

### Basic Implementation

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
tokens = model.to_tokens("The capital of France is")

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

# Get residual stream at each layer
n_layers = model.cfg.n_layers
logit_lens_results = []

for layer in range(n_layers):
    # Get residual stream after layer
    resid = cache["resid_post", layer]  # [batch, seq, d_model]

    # Apply final layer norm
    resid_normed = model.ln_final(resid)

    # Project through unembedding
    layer_logits = resid_normed @ model.W_U  # [batch, seq, vocab]

    # Get top predictions
    top_tokens = layer_logits[0, -1].topk(5)  # Last position
    logit_lens_results.append({
        "layer": layer,
        "top_tokens": model.to_str_tokens(top_tokens.indices),
        "probs": torch.softmax(layer_logits[0, -1], dim=-1)[top_tokens.indices]
    })

# Print results
for result in logit_lens_results:
    print(f"Layer {result['layer']}: {result['top_tokens']}")
```

### Variations

**Tuned Lens**: Learns affine transformations per layer for more accurate predictions.

```python
# pip install tuned-lens
from tuned_lens import TunedLens

tuned_lens = TunedLens.from_pretrained("gpt2-small")
predictions = tuned_lens(residual_stack, model)
```

**Causal Lens**: Projects from causally meaningful directions.

### Interpretation Guidelines

- **Early layers**: Often show input-related tokens
- **Middle layers**: Computation happens here
- **Late layers**: Should converge to final answer
- **Sudden jumps**: May indicate where key computation happens

### Common Artifacts

- Garbage tokens appearing consistently (artifacts of unembedding geometry)
- High-frequency tokens dominating early layers
- Layer norm effects can distort comparisons

## Activation Patching

**What it does**: Replaces activations from one run with activations from another to measure causal effects of specific components.

### Types of Patching

| Type | Method | Use Case |
|------|--------|----------|
| **Resample ablation** | Replace with activation from different input | Measure causal effect |
| **Zero ablation** | Set to zero | Measure if component is necessary |
| **Mean ablation** | Replace with dataset mean | Cleaner than zero ablation |
| **Noise ablation** | Add Gaussian noise | Test robustness |

### Basic Implementation

```python
def activation_patching(
    model,
    clean_tokens,
    corrupted_tokens,
    hook_point,
    position,
    metric_fn,
):
    """Patch activations from clean run into corrupted run."""

    # Get clean activations
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens)
    clean_act = clean_cache[hook_point]

    # Baseline: corrupted run without patching
    corrupted_logits = model(corrupted_tokens)
    baseline_metric = metric_fn(corrupted_logits)

    # Define patching hook
    def patch_hook(activation, hook):
        # Replace at specific position
        activation[:, position, :] = clean_act[:, position, :]
        return activation

    # Run with patching
    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_point, patch_hook)]
    )
    patched_metric = metric_fn(patched_logits)

    # Return effect of patching
    return patched_metric - baseline_metric
```

### Patching Across Positions/Layers

```python
def patch_all_positions_and_layers(model, clean_tokens, corrupted_tokens, metric_fn):
    """Create a position x layer patching heatmap."""
    n_layers = model.cfg.n_layers
    seq_len = clean_tokens.shape[1]

    results = torch.zeros(n_layers, seq_len)

    for layer in range(n_layers):
        for pos in range(seq_len):
            effect = activation_patching(
                model,
                clean_tokens,
                corrupted_tokens,
                f"blocks.{layer}.hook_resid_post",
                pos,
                metric_fn
            )
            results[layer, pos] = effect

    return results
```

### Path Patching

**What it does**: Patches along specific paths through the network to isolate circuit components.

```python
def path_patching(model, clean_tokens, corrupted_tokens, source_hook, dest_hook, metric_fn):
    """
    Measure effect of information flowing from source to dest.

    Only patches the contribution of source component to dest component,
    not the direct effect on the residual stream.
    """
    # Get clean and corrupted caches
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # This is conceptually what path patching does:
    # 1. Run corrupted forward pass up to source
    # 2. At source, patch in clean activation
    # 3. Continue to dest, measuring only the effect through that path
    # 4. Compare to baseline

    # Implementation requires careful hook ordering
    # See TransformerLens documentation for full implementation
    pass
```

## Direct Logit Attribution (DLA)

**What it does**: Decomposes the final logit of a specific token into contributions from each model component.

**Key insight**: The residual stream is a sum of contributions, so we can attribute logits to each term.

### Implementation

```python
def direct_logit_attribution(model, tokens, target_token_idx):
    """
    Compute how much each component contributes to the target token's logit.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Get unembedding direction for target token
    target_direction = model.W_U[:, target_token_idx]  # [d_model]

    # Final position
    pos = -1

    contributions = {}

    # Embedding contribution
    embed_out = cache["hook_embed"][:, pos, :]
    contributions["embed"] = (embed_out @ target_direction).item()

    # Each layer's attention and MLP contribution
    for layer in range(model.cfg.n_layers):
        # Attention output (before adding to residual)
        attn_out = cache["attn_out", layer][:, pos, :]
        contributions[f"L{layer}_attn"] = (attn_out @ target_direction).item()

        # MLP output
        mlp_out = cache["mlp_out", layer][:, pos, :]
        contributions[f"L{layer}_mlp"] = (mlp_out @ target_direction).item()

    return contributions
```

### Per-Head Attribution

```python
def per_head_attribution(model, tokens, target_token_idx):
    """
    Compute logit contribution from each attention head.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    target_direction = model.W_U[:, target_token_idx]
    pos = -1

    head_contributions = {}

    for layer in range(model.cfg.n_layers):
        # z is the output before W_O: [batch, pos, n_heads, d_head]
        z = cache["z", layer][:, pos, :, :]  # [batch, n_heads, d_head]

        # W_O projects from d_head to d_model: [n_heads, d_head, d_model]
        W_O = model.W_O[layer]

        for head in range(model.cfg.n_heads):
            # This head's contribution
            head_out = z[0, head, :] @ W_O[head]  # [d_model]
            contribution = (head_out @ target_direction).item()
            head_contributions[f"L{layer}H{head}"] = contribution

    return head_contributions
```

## Circuit Analysis

**Goal**: Identify the minimal subgraph of the model that implements a specific behavior.

### QK and OV Circuit Decomposition

The attention mechanism can be decomposed:

```
Attention(x) = softmax(x W_Q @ (x W_K)^T / sqrt(d_k)) @ x W_V @ W_O
```

**QK Circuit**: Determines what attends to what
- Operates on positional/identity information
- `W_Q @ W_K^T` gives the effective query-key interaction

**OV Circuit**: Determines what information gets moved
- `W_V @ W_O` gives the effective "copy" operation

```python
def analyze_ov_circuit(model, layer, head):
    """Analyze what the OV circuit copies."""
    W_V = model.W_V[layer, head]  # [d_model, d_head]
    W_O = model.W_O[layer, head]  # [d_head, d_model]

    # Effective OV matrix
    W_OV = W_V @ W_O  # [d_model, d_model]

    # Compose with embeddings to see token-to-token effect
    # What token does this head promote when attending to token i?
    full_OV = model.W_E @ W_OV @ model.W_U  # [vocab, vocab]

    return full_OV

def analyze_qk_circuit(model, layer, head):
    """Analyze attention patterns."""
    W_Q = model.W_Q[layer, head]  # [d_model, d_head]
    W_K = model.W_K[layer, head]  # [d_model, d_head]

    # Effective QK matrix
    W_QK = W_Q @ W_K.T  # [d_model, d_model]

    # With embeddings: which tokens attend to which?
    full_QK = model.W_E @ W_QK @ model.W_E.T  # [vocab, vocab]

    return full_QK
```

### Composition Patterns

Heads can compose: the output of one head becomes input to another.

- **Q-composition**: Head B's queries use Head A's output
- **K-composition**: Head B's keys use Head A's output
- **V-composition**: Head B's values use Head A's output

```python
def measure_q_composition(model, layer_A, head_A, layer_B, head_B):
    """Measure how much head_B's queries depend on head_A's output."""
    # Requires layer_A < layer_B

    W_OV_A = model.W_V[layer_A, head_A] @ model.W_O[layer_A, head_A]
    W_Q_B = model.W_Q[layer_B, head_B]

    # Composition score
    composition = W_OV_A @ W_Q_B
    score = composition.norm().item()

    return score
```

### Circuit Validation

Once you've hypothesized a circuit:

1. **Necessity**: Ablate each component, measure behavior change
2. **Sufficiency**: Run only the circuit components, check if behavior preserved
3. **Minimality**: Can any component be removed without losing behavior?

## Probing

**What it does**: Trains a classifier on intermediate activations to detect if information is present.

### Linear Probe

```python
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)

def train_probe(model, dataset, layer, target_fn, n_classes):
    """
    Train a probe to predict some property from activations.

    Args:
        target_fn: Function that extracts labels from examples
    """
    probe = LinearProbe(model.cfg.d_model, n_classes)
    optimizer = torch.optim.Adam(probe.parameters())
    criterion = nn.CrossEntropyLoss()

    for batch in dataset:
        tokens = model.to_tokens(batch["text"])

        # Get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        acts = cache["resid_post", layer]  # [batch, seq, d_model]

        # Get labels
        labels = target_fn(batch)

        # Train
        logits = probe(acts[:, -1, :])  # Use last position
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe
```

### When to Use Probing

- **Positive use**: Testing if information is represented anywhere
- **Caution**: High probe accuracy doesn't mean the model uses that information
- **Better**: Combine with activation patching to establish causal role

## Attention Pattern Analysis

### Extracting Patterns

```python
def get_attention_patterns(model, tokens):
    """Extract attention patterns for all layers/heads."""
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    patterns = {}
    for layer in range(model.cfg.n_layers):
        # pattern shape: [batch, n_heads, query_pos, key_pos]
        pattern = cache["pattern", layer]
        patterns[layer] = pattern

    return patterns
```

### Common Patterns

| Pattern | Description | How to Detect |
|---------|-------------|---------------|
| **Previous token** | Attends to position i-1 | Diagonal stripe one below main diagonal |
| **Induction** | Attends to token after previous occurrence | Complex, requires specific inputs |
| **Copying** | Attends to matching tokens | High attention to same-token positions |
| **BOS attention** | Attends to beginning of sequence | First column has high values |
| **Positional** | Based on relative position, not content | Consistent pattern across inputs |

### Induction Head Detection

```python
def detect_induction_heads(model, seq_length=50, n_random=10):
    """
    Induction heads complete [A][B] ... [A] -> [B]
    They show high attention from second [A] to first [B].
    """
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for _ in range(n_random):
        # Random repeated sequence: [random tokens] + [random tokens]
        half = torch.randint(0, model.cfg.d_vocab, (1, seq_length // 2))
        tokens = torch.cat([half, half], dim=1)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer in range(model.cfg.n_layers):
            pattern = cache["pattern", layer][0]  # [n_heads, q, k]

            for head in range(model.cfg.n_heads):
                # Check if positions in second half attend to position+1 in first half
                # This is the induction pattern
                for i in range(seq_length // 2, seq_length - 1):
                    source = i - seq_length // 2 + 1  # Where it should attend
                    if source < seq_length // 2:
                        scores[layer, head] += pattern[head, i, source]

    return scores / n_random
```
