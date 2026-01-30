# Mechanistic Interpretability Tools

This reference covers the main libraries and tools used in mechanistic interpretability research.

## TransformerLens

The most widely-used library for mechanistic interpretability. Provides clean access to model internals with a consistent interface.

### Installation

```bash
pip install transformer-lens
# With visualization extras
pip install transformer-lens[viz]
```

### Core Concepts

**HookedTransformer**: Wraps models with hooks at every layer.

```python
from transformer_lens import HookedTransformer

# Load a pretrained model
model = HookedTransformer.from_pretrained("gpt2-small")

# Key attributes
model.cfg                    # Model configuration
model.tokenizer             # Tokenizer
model.W_E                   # Embedding matrix
model.W_U                   # Unembedding matrix
model.W_pos                 # Positional embeddings
model.blocks[i].attn.W_Q    # Query weights for layer i
model.blocks[i].attn.W_K    # Key weights
model.blocks[i].attn.W_V    # Value weights
model.blocks[i].attn.W_O    # Output projection
model.blocks[i].mlp.W_in    # MLP input projection
model.blocks[i].mlp.W_out   # MLP output projection
```

**ActivationCache**: Stores all intermediate activations.

```python
# Run with caching
logits, cache = model.run_with_cache(tokens)

# Access cached activations
cache["resid_pre", layer]           # Residual stream before layer
cache["resid_post", layer]          # Residual stream after layer
cache["attn_out", layer]            # Attention output
cache["mlp_out", layer]             # MLP output
cache["pattern", layer]             # Attention patterns [batch, head, q_pos, k_pos]
cache["q", layer]                   # Query activations
cache["k", layer]                   # Key activations
cache["v", layer]                   # Value activations
cache["z", layer]                   # Attention output before W_O
```

### Hook Points Naming Convention

TransformerLens uses a consistent naming scheme:

```
hook_embed                      # Token embeddings
hook_pos_embed                  # Positional embeddings
blocks.{layer}.hook_resid_pre   # Residual stream input to layer
blocks.{layer}.attn.hook_q      # Query activations
blocks.{layer}.attn.hook_k      # Key activations
blocks.{layer}.attn.hook_v      # Value activations
blocks.{layer}.attn.hook_z      # Pre-output attention
blocks.{layer}.attn.hook_attn_scores  # Raw attention scores
blocks.{layer}.attn.hook_pattern      # Softmaxed attention
blocks.{layer}.hook_attn_out    # Attention sublayer output
blocks.{layer}.hook_resid_mid   # Between attention and MLP
blocks.{layer}.mlp.hook_pre     # MLP input
blocks.{layer}.mlp.hook_post    # MLP output (after nonlinearity)
blocks.{layer}.hook_mlp_out     # MLP sublayer output
blocks.{layer}.hook_resid_post  # Residual stream output of layer
ln_final.hook_normalized        # After final layer norm
```

### Running Hooks

```python
# Simple hook to read activations
def hook_fn(activation, hook):
    print(f"Shape at {hook.name}: {activation.shape}")
    return activation  # Return modified or original

# Run with hooks
model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)]
)

# Patching example
def patching_hook(activation, hook):
    # Replace with activations from another run
    activation[:, position, :] = corrupted_activations[:, position, :]
    return activation

# Use reset_hooks() when done
model.reset_hooks()
```

### Supported Models (50+)

- GPT-2 family: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- GPT-Neo: `EleutherAI/gpt-neo-125M`, etc.
- Pythia: `EleutherAI/pythia-70m` through `pythia-12b`
- Llama/Llama2: Various sizes (requires access)
- Mistral: `mistralai/Mistral-7B-v0.1`
- Gemma: `google/gemma-2b`, `google/gemma-7b`
- Many more via HuggingFace

### Memory-Efficient Caching

```python
# Cache only specific activations
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "resid_post" in name  # Only residual streams
)

# Remove intermediate activations from cache
cache.remove_batch_dim()  # If batch size is 1

# Use half precision
model.to(torch.float16)
```

## nnsight

Library for interacting with any PyTorch model (especially larger HuggingFace models).

### Installation

```bash
pip install nnsight
```

### Core Pattern

```python
from nnsight import LanguageModel

# Load any HuggingFace model
model = LanguageModel("meta-llama/Llama-2-7b-hf")

# Use tracer context for interventions
with model.trace("Hello world") as tracer:
    # Access internal activations
    layer_output = model.model.layers[0].output[0]

    # Save for later
    layer_output_saved = layer_output.save()

    # Modify activations
    model.model.layers[0].output[0][:, :, :] = modified_values

# Access saved values after context exits
print(layer_output_saved.value.shape)
```

### When to Use nnsight Over TransformerLens

- **Larger models**: Better memory management for 7B+ parameter models
- **Newer architectures**: Works with any HuggingFace model immediately
- **HF ecosystem integration**: Uses native HuggingFace tokenizers and configs
- **Custom models**: Can wrap any PyTorch module

### Common Patterns

```python
# Read multiple activations
with model.trace(tokens) as tracer:
    resid_0 = model.model.layers[0].output[0].save()
    resid_5 = model.model.layers[5].output[0].save()
    attn_pattern = model.model.layers[3].self_attn.attention_weights.save()

# Activation patching
with model.trace(clean_tokens) as clean_tracer:
    clean_resid = model.model.layers[layer].output[0].save()

with model.trace(corrupted_tokens) as patch_tracer:
    # Patch in clean activations
    model.model.layers[layer].output[0][:, pos, :] = clean_resid.value[:, pos, :]
    patched_logits = model.output.logits.save()
```

## nnterp

Unified interface bridging TransformerLens and nnsight patterns.

```bash
pip install nnterp
```

```python
from nnterp import load_model, get_residual_stream

# Consistent interface regardless of backend
model = load_model("gpt2-small", backend="transformerlens")
# OR
model = load_model("meta-llama/Llama-2-7b-hf", backend="nnsight")

# Same API for both
resid = get_residual_stream(model, tokens, layer=5)
```

## SAELens

Training and analyzing sparse autoencoders.

### Installation

```bash
pip install sae-lens
# For training
pip install sae-lens[train]
```

### Training SAEs

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

config = LanguageModelSAERunnerConfig(
    # Model settings
    model_name="gpt2-small",
    hook_point="blocks.8.hook_resid_pre",
    hook_point_layer=8,
    d_in=768,  # Model hidden size

    # SAE architecture
    expansion_factor=32,  # SAE hidden = 32 * 768
    architecture="standard",  # or "topk", "gated"

    # Training
    l1_coefficient=0.001,  # Sparsity weight
    lr=3e-4,
    lr_scheduler_name="constant",
    training_tokens=1_000_000_000,
    batch_size=4096,

    # Data
    dataset_path="Skylion007/openwebtext",
    context_size=128,

    # Logging
    log_to_wandb=True,
    wandb_project="sae-training",
)

runner = SAETrainingRunner(config)
sae = runner.run()
```

### Loading Pretrained SAEs

```python
from sae_lens import SAE

# Load from Hugging Face
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
)

# Get feature activations
feature_acts = sae.encode(residual_stream)  # [batch, seq, n_features]

# Reconstruct
reconstructed = sae.decode(feature_acts)

# Error
reconstruction_error = residual_stream - reconstructed
```

### Analyzing Features

```python
# Find max activating examples
from sae_lens.analysis import find_max_activating_examples

max_examples = find_max_activating_examples(
    sae=sae,
    model=model,
    feature_idx=42,
    dataset="openwebtext",
    n_examples=20,
)

# Feature dashboard
from sae_lens.analysis import FeatureDashboard

dashboard = FeatureDashboard(sae, model)
dashboard.show(feature_idx=42)
```

## Other Tools

### SAE-Vis

Visualization for SAE features.

```bash
pip install sae-vis
```

```python
from sae_vis import FeatureVisualizer

vis = FeatureVisualizer(sae, model)
vis.visualize_feature(feature_idx=100, save_path="feature_100.html")
```

### Neuronpedia

Web interface for exploring published SAE features: [neuronpedia.org](https://neuronpedia.org)

- Browse max activating examples
- Search for features by concept
- Compare features across models

### SAEBench

Standardized evaluation for SAEs.

```bash
pip install saebench
```

```python
from saebench import evaluate_sae

results = evaluate_sae(
    sae=sae,
    model=model,
    evals=["absorption", "autointerp", "sparse_probing"]
)
```

### CircuitsVis

Interactive circuit visualization (attention patterns, logit lens).

```bash
pip install circuitsvis
```

```python
import circuitsvis as cv

# Attention pattern visualization
cv.attention.attention_heads(
    attention=attention_pattern,  # [n_heads, seq, seq]
    tokens=tokens,
)

# Logit lens
cv.logit_lens.logit_lens(
    residual_stack=residual_stack,  # [n_layers, seq, d_model]
    model=model,
)
```

## Tool Selection Guide

| Task | Recommended Tool |
|------|------------------|
| GPT-2, Pythia exploration | TransformerLens |
| Llama 2/3, Mistral (7B+) | nnsight |
| Quick prototyping | TransformerLens |
| Production-scale SAE training | SAELens |
| Feature exploration | Neuronpedia + SAE-Vis |
| Attention visualization | CircuitsVis |
| Model-agnostic code | nnterp |

## Common Setup Pattern

```python
import torch
from transformer_lens import HookedTransformer

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device=device,
    dtype=torch.float32,  # or torch.float16 for larger models
)
model.eval()

# Tokenize input
text = "The quick brown fox"
tokens = model.to_tokens(text)  # [1, seq_len]
print(f"Tokens: {model.to_str_tokens(tokens)}")

# Run with cache
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

print(f"Logits shape: {logits.shape}")
print(f"Available cache keys: {list(cache.keys())[:10]}...")
```
