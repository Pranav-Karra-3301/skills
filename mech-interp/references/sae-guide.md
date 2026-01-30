# Sparse Autoencoder (SAE) Guide

This reference covers training and analyzing sparse autoencoders for mechanistic interpretability.

## Why SAEs?

**The problem**: Neural networks represent many more concepts than they have dimensions (superposition hypothesis). Individual neurons are often polysemantic, responding to unrelated concepts.

**The solution**: SAEs learn to decompose activations into a larger, sparser set of features that are more likely to be monosemantic.

**Core equation**:
```
f(x) = ReLU(W_enc @ (x - b_dec) + b_enc)  # Encode: find active features
x_hat = W_dec @ f(x) + b_dec              # Decode: reconstruct
```

**Training objective**:
```
L = ||x - x_hat||² + λ * ||f(x)||₁
    reconstruction   sparsity
```

## Architecture Variants

| Variant | Activation | Sparsity Mechanism | When to Use |
|---------|------------|-------------------|-------------|
| **Standard** | ReLU | L1 penalty | General purpose, well understood |
| **TopK** | TopK | Fixed K active | Exact sparsity control needed |
| **BatchTopK** | TopK per batch | Better gradients than TopK | Training stability issues |
| **JumpReLU** | Discontinuous step | Sharper features | When standard has shrinkage issues |
| **Gated** | Separate gate network | Decouples magnitude/selection | Research frontier |

### Standard SAE

```python
class StandardSAE(nn.Module):
    def __init__(self, d_in, n_features):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x - self.b_dec)  # Center first
        x_hat = self.decode(f)
        return x_hat, f
```

### TopK SAE

```python
class TopKSAE(nn.Module):
    def __init__(self, d_in, n_features, k):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        pre_act = x @ self.W_enc + self.b_enc
        # Keep only top K activations
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
        f = torch.zeros_like(pre_act)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))
        return f

    def forward(self, x):
        f = self.encode(x - self.b_dec)
        x_hat = f @ self.W_dec + self.b_dec
        return x_hat, f
```

## Key Hyperparameters

### Expansion Factor (Width)

Ratio of SAE features to input dimensions.

| Factor | Features (d_model=768) | Trade-off |
|--------|----------------------|-----------|
| 4x | 3,072 | Fast training, may miss fine features |
| 8x | 6,144 | Good balance for exploration |
| 16x | 12,288 | Standard for publication |
| 32x | 24,576 | Captures rare features, expensive |
| 64x+ | 49,152+ | Research frontier, very expensive |

**Guidance**: Start with 8x for initial exploration, scale up if features seem polysemantic.

### L1 Coefficient (Sparsity Weight)

Controls sparsity vs reconstruction trade-off.

| L1 | Effect |
|----|--------|
| Too low | High L0, features not sparse enough |
| Just right | L0 ≈ 10-100, good reconstruction |
| Too high | Many dead features, poor reconstruction |

**Typical values**: 0.0001 to 0.01, depends on model and layer.

### Learning Rate

```python
# Typical schedule
lr_start = 1e-4        # Initial learning rate
lr_peak = 3e-4         # After warmup
warmup_steps = 1000    # Warmup period

# With cosine decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=lr_start / 10
)
```

### Batch Size

Larger is generally better (more stable gradients).

| Batch Size | Notes |
|------------|-------|
| 4096 | Standard, fits most GPUs |
| 8192 | Better if memory allows |
| 16384+ | For larger SAEs |

## Training Best Practices

### Setup

```python
import torch
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

config = LanguageModelSAERunnerConfig(
    # Model
    model_name="gpt2-small",
    hook_point="blocks.8.hook_resid_pre",
    hook_point_layer=8,
    d_in=768,

    # Architecture
    expansion_factor=16,
    architecture="standard",  # or "topk"

    # Training
    l1_coefficient=0.001,
    lr=3e-4,
    lr_scheduler_name="cosine",
    lr_warm_up_steps=1000,
    training_tokens=1_000_000_000,
    batch_size=4096,

    # IMPORTANT: Use autocast for speed
    autocast=True,

    # Dead feature handling
    dead_feature_window=5000,
    dead_feature_threshold=1e-8,
    feature_sampling_window=1000,

    # Data
    dataset_path="Skylion007/openwebtext",
    context_size=128,
    prepend_bos=True,

    # Checkpointing
    checkpoint_path="checkpoints/",
    n_checkpoints=10,

    # Logging
    log_to_wandb=True,
    wandb_project="sae-training",
)

runner = SAETrainingRunner(config)
sae = runner.run()
```

### Monitoring Training

**Key metrics to track**:

| Metric | Good Range | Problem If |
|--------|------------|------------|
| L0 (avg features active) | 10-100 | >200 (not sparse), <5 (too sparse) |
| Reconstruction loss | Decreasing | Flat or increasing |
| Explained variance | >95% | <80% indicates issues |
| Dead features | <5% | >20% indicates problems |

```python
# Compute metrics during training
def compute_metrics(sae, activations):
    with torch.no_grad():
        x_hat, f = sae(activations)

        # L0: average number of active features
        l0 = (f > 0).float().sum(dim=-1).mean()

        # Reconstruction loss
        recon_loss = (activations - x_hat).pow(2).mean()

        # Explained variance
        total_var = activations.var()
        residual_var = (activations - x_hat).var()
        explained_var = 1 - residual_var / total_var

        # Dead features
        active_per_feature = (f > 0).float().sum(dim=0)
        dead_fraction = (active_per_feature == 0).float().mean()

    return {
        "l0": l0.item(),
        "recon_loss": recon_loss.item(),
        "explained_var": explained_var.item(),
        "dead_fraction": dead_fraction.item(),
    }
```

### Dead Feature Resampling

Features that never activate are "dead" and waste capacity.

```python
def resample_dead_features(sae, activations, threshold=1e-8):
    """Reinitialize dead features with high-loss examples."""
    with torch.no_grad():
        x_hat, f = sae(activations)

        # Find dead features
        active_counts = (f > threshold).float().sum(dim=0)
        dead_mask = active_counts == 0

        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return

        # Find examples with highest reconstruction error
        errors = (activations - x_hat).pow(2).sum(dim=-1)
        _, top_idx = errors.topk(n_dead)

        # Reinitialize dead features to point toward high-error examples
        sae.W_enc[:, dead_mask] = activations[top_idx].T
        sae.W_dec[dead_mask, :] = activations[top_idx]

        # Normalize
        sae.W_dec[dead_mask, :] /= sae.W_dec[dead_mask, :].norm(dim=-1, keepdim=True)

        print(f"Resampled {n_dead} dead features")
```

## Analysis Workflow

### 1. Feature Activation Analysis

```python
def analyze_feature(sae, model, feature_idx, texts):
    """Find what activates a feature."""
    all_activations = []

    for text in texts:
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        resid = cache["resid_pre", sae.cfg.hook_point_layer]
        _, feature_acts = sae(resid)

        # Get activation of target feature at each position
        acts = feature_acts[0, :, feature_idx]  # [seq_len]
        str_tokens = model.to_str_tokens(tokens)

        for i, (tok, act) in enumerate(zip(str_tokens[0], acts)):
            if act > 0.1:  # Threshold
                all_activations.append({
                    "text": text,
                    "token": tok,
                    "position": i,
                    "activation": act.item(),
                })

    # Sort by activation strength
    all_activations.sort(key=lambda x: -x["activation"])
    return all_activations[:20]
```

### 2. Max Activating Examples

```python
def find_max_activating_examples(sae, model, feature_idx, dataset, n_examples=20):
    """Find inputs that maximally activate a feature."""
    top_examples = []

    for batch in dataset:
        tokens = model.to_tokens(batch)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        resid = cache["resid_pre", sae.cfg.hook_point_layer]
        _, feature_acts = sae(resid)

        # Max activation per sequence
        max_acts, max_pos = feature_acts[:, :, feature_idx].max(dim=1)

        for i, (act, pos) in enumerate(zip(max_acts, max_pos)):
            if act > 0:
                top_examples.append({
                    "text": batch[i],
                    "activation": act.item(),
                    "position": pos.item(),
                })

    # Sort and return top
    top_examples.sort(key=lambda x: -x["activation"])
    return top_examples[:n_examples]
```

### 3. Feature Steering

Test if a feature causally affects behavior.

```python
def steer_with_feature(model, sae, tokens, feature_idx, strength=1.0):
    """Add feature direction to residual stream."""
    feature_direction = sae.W_dec[feature_idx]  # [d_model]

    def steering_hook(activation, hook):
        # Add scaled feature direction
        activation[:, :, :] += strength * feature_direction
        return activation

    hook_point = f"blocks.{sae.cfg.hook_point_layer}.hook_resid_pre"
    steered_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_point, steering_hook)]
    )

    return steered_logits
```

### 4. Feature Ablation

Test if removing a feature changes behavior.

```python
def ablate_feature(model, sae, tokens, feature_idx):
    """Run model with specific feature zeroed out."""

    def ablation_hook(activation, hook):
        # Encode, zero feature, decode
        _, f = sae(activation)
        f[:, :, feature_idx] = 0
        reconstructed = sae.decode(f)
        return reconstructed

    hook_point = f"blocks.{sae.cfg.hook_point_layer}.hook_resid_pre"
    ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_point, ablation_hook)]
    )

    return ablated_logits
```

## Evaluation

### Reconstruction Quality

```python
def evaluate_reconstruction(sae, model, dataset):
    """Measure how well SAE reconstructions preserve model behavior."""
    original_losses = []
    reconstructed_losses = []

    for batch in dataset:
        tokens = model.to_tokens(batch)

        # Original forward pass
        orig_logits = model(tokens)
        orig_loss = F.cross_entropy(
            orig_logits[:, :-1].flatten(0, 1),
            tokens[:, 1:].flatten()
        )
        original_losses.append(orig_loss.item())

        # With SAE reconstruction
        def recon_hook(activation, hook):
            x_hat, _ = sae(activation)
            return x_hat

        hook_point = f"blocks.{sae.cfg.hook_point_layer}.hook_resid_pre"
        recon_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_point, recon_hook)]
        )
        recon_loss = F.cross_entropy(
            recon_logits[:, :-1].flatten(0, 1),
            tokens[:, 1:].flatten()
        )
        reconstructed_losses.append(recon_loss.item())

    orig_mean = sum(original_losses) / len(original_losses)
    recon_mean = sum(reconstructed_losses) / len(reconstructed_losses)

    print(f"Original loss: {orig_mean:.4f}")
    print(f"Reconstructed loss: {recon_mean:.4f}")
    print(f"Loss increase: {recon_mean - orig_mean:.4f}")

    # Loss recovered: how much of the gap from random to trained is preserved
    # (Closer to 1 is better)
```

### Sparsity-Reconstruction Pareto

Plot L0 vs reconstruction loss across different L1 coefficients to find the optimal trade-off.

```python
def pareto_sweep(base_config, l1_values):
    """Train SAEs with different L1 coefficients to find Pareto frontier."""
    results = []

    for l1 in l1_values:
        config = base_config.copy()
        config.l1_coefficient = l1

        runner = SAETrainingRunner(config)
        sae = runner.run()

        metrics = evaluate_sae(sae)
        results.append({
            "l1": l1,
            "l0": metrics["l0"],
            "recon_loss": metrics["recon_loss"],
        })

    return results
```
