# Visualization for Mechanistic Interpretability

This reference covers common visualization patterns for mech interp experiments.

## Attention Pattern Heatmaps

### Basic Matplotlib Pattern

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_pattern(pattern, tokens, title="Attention Pattern"):
    """
    Plot a single attention head's pattern.

    Args:
        pattern: [seq_len, seq_len] attention weights
        tokens: list of token strings
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(pattern, cmap="Blues")

    # Labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()

    return fig
```

### Multi-Head Attention Grid

```python
def plot_attention_heads(patterns, tokens, n_cols=4, layer=0):
    """
    Plot all attention heads in a grid.

    Args:
        patterns: [n_heads, seq_len, seq_len]
        tokens: list of token strings
    """
    n_heads = patterns.shape[0]
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for head in range(n_heads):
        ax = axes[head]
        im = ax.imshow(patterns[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"L{layer}H{head}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Layer {layer} Attention Patterns")
    plt.tight_layout()

    return fig
```

### Interactive with Plotly

```python
import plotly.graph_objects as go
import plotly.express as px

def plot_attention_interactive(pattern, tokens, title="Attention"):
    """Interactive attention heatmap with hover info."""
    fig = go.Figure(data=go.Heatmap(
        z=pattern,
        x=tokens,
        y=tokens,
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Key (attending to)",
        yaxis_title="Query (attending from)",
        width=700,
        height=600,
    )

    return fig
```

## Logit Lens Visualization

### Layer x Token Heatmap

```python
def plot_logit_lens(logit_lens_results, tokens, target_token=None):
    """
    Plot probability of target token (or max prob token) across layers.

    Args:
        logit_lens_results: [n_layers, seq_len, vocab_size] logits
        tokens: list of token strings
        target_token: token index to track, or None for max
    """
    n_layers, seq_len, _ = logit_lens_results.shape

    # Convert to probabilities
    probs = torch.softmax(torch.tensor(logit_lens_results), dim=-1)

    if target_token is not None:
        # Track specific token
        values = probs[:, :, target_token].numpy()
        title = f"Probability of target token across layers"
    else:
        # Track max probability
        values = probs.max(dim=-1).values.numpy()
        title = "Max probability across layers"

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])

    ax.set_xlabel("Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    return fig
```

### Top Predictions Per Layer

```python
def plot_top_predictions(logit_lens_results, model, position=-1, top_k=5):
    """
    Show top predicted tokens at each layer for a specific position.
    """
    n_layers = logit_lens_results.shape[0]

    fig, ax = plt.subplots(figsize=(15, 8))

    for layer in range(n_layers):
        logits = logit_lens_results[layer, position]
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        top_probs, top_idx = probs.topk(top_k)

        tokens = model.to_str_tokens(top_idx.unsqueeze(0))[0]

        for i, (tok, prob) in enumerate(zip(tokens, top_probs)):
            ax.text(layer, i, f"{tok}\n{prob:.2f}",
                    ha="center", va="center", fontsize=8)

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, top_k - 0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank")
    ax.set_title(f"Top {top_k} predictions at each layer (position {position})")
    ax.invert_yaxis()

    return fig
```

## Activation Patching Results

### Position x Layer Heatmap

```python
def plot_patching_results(results, tokens, title="Activation Patching"):
    """
    Plot patching effects across positions and layers.

    Args:
        results: [n_layers, seq_len] patching effects
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use diverging colormap centered at 0
    vmax = max(abs(results.min()), abs(results.max()))
    im = ax.imshow(results, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(results.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(results.shape[0])])

    ax.set_xlabel("Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Patching effect")
    plt.tight_layout()

    return fig
```

### Component Attribution Bar Chart

```python
def plot_component_attribution(contributions, title="Direct Logit Attribution"):
    """
    Bar chart of component contributions to target logit.
    """
    names = list(contributions.keys())
    values = list(contributions.values())

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))

    colors = ["green" if v > 0 else "red" for v in values]
    ax.barh(names, values, color=colors)

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Contribution to target logit")
    ax.set_title(title)

    plt.tight_layout()
    return fig
```

## SAE Feature Visualization

### Feature Activation Histogram

```python
def plot_feature_histogram(activations, feature_idx, bins=50):
    """
    Histogram of feature activations across dataset.
    """
    acts = activations[:, feature_idx].numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Only plot non-zero activations
    nonzero = acts[acts > 0]

    ax.hist(nonzero, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label=f"Zero ({(acts == 0).mean():.1%} of data)")

    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.set_title(f"Feature {feature_idx} activation distribution")
    ax.legend()

    plt.tight_layout()
    return fig
```

### Max Activating Examples Display

```python
def display_max_activating_examples(examples, feature_idx):
    """
    Display max activating examples with highlighted tokens.
    """
    print(f"\n=== Feature {feature_idx} Max Activating Examples ===\n")

    for i, ex in enumerate(examples[:10]):
        text = ex["text"]
        pos = ex["position"]
        act = ex["activation"]

        # Show context around activating position
        tokens = text.split()
        start = max(0, pos - 5)
        end = min(len(tokens), pos + 6)

        context = tokens[start:end]
        highlight_pos = pos - start

        # Mark the activating token
        if 0 <= highlight_pos < len(context):
            context[highlight_pos] = f">>>{context[highlight_pos]}<<<"

        print(f"{i+1}. (act={act:.3f}) {' '.join(context)}")
```

### Feature Dashboard

```python
def create_feature_dashboard(sae, model, feature_idx, examples):
    """
    Multi-panel dashboard for a single feature.
    """
    fig = plt.figure(figsize=(16, 10))

    # Layout: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Activation histogram (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    # ... histogram code ...

    # 2. Top tokens bar chart (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    # ... token frequency bar chart ...

    # 3. Feature direction in embedding space (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    # ... top/bottom tokens by feature direction ...

    # 4. Example contexts (bottom, spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    example_text = "\n".join([f"• {ex['text'][:100]}..." for ex in examples[:5]])
    ax4.text(0, 1, example_text, va="top", fontsize=9, family="monospace")

    plt.suptitle(f"Feature {feature_idx} Dashboard")

    return fig
```

## Training Metrics

### Loss Curves

```python
def plot_training_curves(metrics_log):
    """
    Plot training metrics over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    steps = range(len(metrics_log))

    # Reconstruction loss
    axes[0, 0].plot(steps, [m["recon_loss"] for m in metrics_log])
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reconstruction Loss")
    axes[0, 0].set_title("Reconstruction Loss")

    # L0
    axes[0, 1].plot(steps, [m["l0"] for m in metrics_log])
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("L0")
    axes[0, 1].set_title("Sparsity (L0)")

    # Dead features
    axes[1, 0].plot(steps, [m["dead_fraction"] for m in metrics_log])
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_title("Dead Feature Fraction")

    # Explained variance
    axes[1, 1].plot(steps, [m["explained_var"] for m in metrics_log])
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Explained Variance")
    axes[1, 1].set_title("Explained Variance")

    plt.tight_layout()
    return fig
```

### Pareto Frontier

```python
def plot_pareto_frontier(sweep_results):
    """
    Plot L0 vs reconstruction loss for different hyperparameters.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    l0s = [r["l0"] for r in sweep_results]
    losses = [r["recon_loss"] for r in sweep_results]
    l1s = [r["l1"] for r in sweep_results]

    scatter = ax.scatter(l0s, losses, c=np.log10(l1s), cmap="viridis", s=100)

    # Label points
    for i, r in enumerate(sweep_results):
        ax.annotate(f'L1={r["l1"]:.0e}', (l0s[i], losses[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("L0 (sparsity)")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Sparsity-Reconstruction Trade-off")

    plt.colorbar(scatter, label="log10(L1 coefficient)")
    plt.tight_layout()

    return fig
```

## Activation Projections

### PCA of Residual Stream

```python
from sklearn.decomposition import PCA

def plot_residual_pca(residuals, labels=None, title="Residual Stream PCA"):
    """
    Project residual stream activations to 2D with PCA.

    Args:
        residuals: [n_samples, d_model]
        labels: optional labels for coloring
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(residuals)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(projected[:, 0], projected[:, 1],
                            c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(projected[:, 0], projected[:, 1], alpha=0.7)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)

    plt.tight_layout()
    return fig
```

### Interactive Plotly Scatter

```python
def plot_interactive_pca(residuals, tokens, hover_texts):
    """
    Interactive 2D projection with hover information.
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(residuals)

    fig = go.Figure(data=go.Scatter(
        x=projected[:, 0],
        y=projected[:, 1],
        mode="markers",
        marker=dict(size=8, opacity=0.7),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>"
    ))

    fig.update_layout(
        title="Residual Stream PCA",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        width=800,
        height=600,
    )

    return fig
```

## CircuitsVis Integration

```python
import circuitsvis as cv

# Attention patterns (works in Jupyter)
cv.attention.attention_heads(
    attention=attention_pattern,  # [n_heads, seq, seq]
    tokens=tokens,
)

# Attention by token
cv.attention.attention_patterns(
    attention=attention_pattern,
    tokens=tokens,
)

# Logit lens visualization
cv.logit_lens.logit_lens(
    residual_stack=residual_stack,  # [n_layers, seq, d_model]
    model=model,
)
```

## Quick Gradio App

```python
import gradio as gr

def create_feature_explorer(sae, model):
    """Simple Gradio app for exploring SAE features."""

    def explore_feature(feature_idx, text):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        resid = cache["resid_pre", sae.cfg.hook_point_layer]
        _, feature_acts = sae(resid)

        acts = feature_acts[0, :, feature_idx].tolist()
        str_tokens = model.to_str_tokens(tokens)[0]

        result = []
        for tok, act in zip(str_tokens, acts):
            if act > 0.1:
                result.append(f"**{tok}** ({act:.2f})")
            else:
                result.append(tok)

        return " ".join(result)

    demo = gr.Interface(
        fn=explore_feature,
        inputs=[
            gr.Slider(0, sae.cfg.d_sae - 1, step=1, label="Feature Index"),
            gr.Textbox(label="Input Text"),
        ],
        outputs=gr.Markdown(label="Feature Activations"),
        title="SAE Feature Explorer",
    )

    return demo

# Launch with: demo.launch()
```
