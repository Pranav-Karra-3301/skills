# Compute Awareness for Mechanistic Interpretability

This reference covers GPU detection, memory estimation, and compute platform guidance.

## GPU Detection Patterns

### Basic PyTorch Detection

```python
import torch

def get_compute_info():
    """Detect available compute resources."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })

        # Current device memory
        info["current_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["current_cached_gb"] = torch.cuda.memory_reserved() / (1024**3)

    return info

# Usage
info = get_compute_info()
print(f"CUDA available: {info['cuda_available']}")
if info['devices']:
    print(f"GPU: {info['devices'][0]['name']} ({info['devices'][0]['total_memory_gb']:.1f}GB)")
```

### Environment Detection

```python
import os
import subprocess

def detect_compute_environment():
    """Detect what compute environment we're running in."""
    env = {"type": "unknown"}

    # Modal
    if os.environ.get("MODAL_TASK_ID"):
        env["type"] = "modal"
        env["task_id"] = os.environ.get("MODAL_TASK_ID")

    # Colab
    elif "COLAB_GPU" in os.environ or os.path.exists("/content"):
        env["type"] = "colab"
        env["has_gpu"] = "COLAB_GPU" in os.environ

    # RunPod
    elif os.environ.get("RUNPOD_POD_ID"):
        env["type"] = "runpod"

    # Vast.ai
    elif os.environ.get("VAST_CONTAINERLABEL"):
        env["type"] = "vast"

    # Lambda Labs
    elif os.path.exists("/etc/lambda"):
        env["type"] = "lambda"

    # SSH to remote (check hostname pattern or env)
    elif os.environ.get("SSH_CONNECTION"):
        env["type"] = "ssh_remote"
        try:
            result = subprocess.run(["hostname"], capture_output=True, text=True)
            env["hostname"] = result.stdout.strip()
        except:
            pass

    # Local
    else:
        env["type"] = "local"

    # Add GPU info
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return env
```

## Memory Estimation

### Model Memory

| Model | Parameters | FP32 Memory | FP16/BF16 Memory |
|-------|------------|-------------|------------------|
| GPT-2-small | 124M | ~0.5GB | ~0.25GB |
| GPT-2-medium | 355M | ~1.4GB | ~0.7GB |
| GPT-2-large | 774M | ~3GB | ~1.5GB |
| GPT-2-xl | 1.5B | ~6GB | ~3GB |
| Pythia-70M | 70M | ~0.3GB | ~0.15GB |
| Pythia-160M | 160M | ~0.6GB | ~0.3GB |
| Pythia-410M | 410M | ~1.6GB | ~0.8GB |
| Pythia-1B | 1B | ~4GB | ~2GB |
| Pythia-2.8B | 2.8B | ~11GB | ~5.5GB |
| Pythia-6.9B | 6.9B | ~28GB | ~14GB |
| Pythia-12B | 12B | ~48GB | ~24GB |
| Llama-2-7B | 7B | ~28GB | ~14GB |
| Llama-2-13B | 13B | ~52GB | ~26GB |
| Mistral-7B | 7B | ~28GB | ~14GB |

### Activation Cache Memory

```python
def estimate_cache_memory(
    model_name: str,
    seq_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 4,  # 4 for float32, 2 for float16
) -> dict:
    """Estimate memory needed for full activation cache."""

    # Model configs (d_model, n_layers, n_heads, d_head)
    configs = {
        "gpt2-small": (768, 12, 12, 64),
        "gpt2-medium": (1024, 24, 16, 64),
        "gpt2-large": (1280, 36, 20, 64),
        "gpt2-xl": (1600, 48, 25, 64),
        "pythia-70m": (512, 6, 8, 64),
        "pythia-160m": (768, 12, 12, 64),
        "pythia-410m": (1024, 24, 16, 64),
        "pythia-1b": (2048, 16, 8, 256),
        "pythia-2.8b": (2560, 32, 32, 80),
        "llama-2-7b": (4096, 32, 32, 128),
        "mistral-7b": (4096, 32, 32, 128),
    }

    if model_name not in configs:
        return {"error": f"Unknown model: {model_name}"}

    d_model, n_layers, n_heads, d_head = configs[model_name]

    # Per-layer cache sizes (rough estimates)
    # Residual stream: batch * seq * d_model
    resid_per_layer = batch_size * seq_length * d_model * dtype_bytes

    # Attention: Q, K, V, Z, pattern (pattern is n_heads * seq * seq)
    attn_per_layer = (
        4 * batch_size * seq_length * n_heads * d_head * dtype_bytes +  # Q, K, V, Z
        batch_size * n_heads * seq_length * seq_length * dtype_bytes     # Attention pattern
    )

    # MLP: pre, post
    mlp_per_layer = 2 * batch_size * seq_length * 4 * d_model * dtype_bytes

    # Total per layer
    per_layer_bytes = resid_per_layer + attn_per_layer + mlp_per_layer

    # Full model
    total_bytes = n_layers * per_layer_bytes

    return {
        "model": model_name,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "dtype": "float32" if dtype_bytes == 4 else "float16",
        "per_layer_mb": per_layer_bytes / (1024**2),
        "total_gb": total_bytes / (1024**3),
        "n_layers": n_layers,
    }

# Example
est = estimate_cache_memory("gpt2-small", seq_length=512, batch_size=1)
print(f"Full cache for {est['model']}: {est['total_gb']:.2f}GB")
```

### SAE Memory

```python
def estimate_sae_memory(
    d_model: int,
    expansion_factor: int,
    batch_size: int,
    seq_length: int,
    dtype_bytes: int = 4,
) -> dict:
    """Estimate memory for SAE training."""

    n_features = d_model * expansion_factor

    # SAE parameters: W_enc, W_dec, b_enc, b_dec
    param_bytes = (
        d_model * n_features * dtype_bytes +  # W_enc
        n_features * d_model * dtype_bytes +  # W_dec
        n_features * dtype_bytes +             # b_enc
        d_model * dtype_bytes                  # b_dec
    )

    # Activations during forward pass
    # Input: batch * seq * d_model
    # Hidden: batch * seq * n_features
    # Output: batch * seq * d_model
    activation_bytes = (
        batch_size * seq_length * d_model * dtype_bytes +      # Input
        batch_size * seq_length * n_features * dtype_bytes +   # Hidden
        batch_size * seq_length * d_model * dtype_bytes        # Output
    )

    # Gradients (roughly same as parameters + activations)
    gradient_bytes = param_bytes + activation_bytes

    # Optimizer state (Adam: 2x parameters for m and v)
    optimizer_bytes = 2 * param_bytes

    total_bytes = param_bytes + activation_bytes + gradient_bytes + optimizer_bytes

    return {
        "d_model": d_model,
        "n_features": n_features,
        "expansion_factor": expansion_factor,
        "parameters_gb": param_bytes / (1024**3),
        "activations_gb": activation_bytes / (1024**3),
        "total_training_gb": total_bytes / (1024**3),
    }
```

## GPU Selection Guide

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| GPT-2-small experiments | 8GB (RTX 3070) | 16GB (RTX 4080) |
| GPT-2-xl / Pythia-2.8B | 16GB (RTX 4080) | 24GB (RTX 3090/4090) |
| Pythia-6.9B / Llama-7B | 24GB (RTX 3090) | 40GB (A100-40GB) |
| SAE training (8x on 768) | 16GB | 24GB |
| SAE training (32x on 768) | 24GB | 40GB |
| SAE training (8x on 4096) | 40GB | 80GB (A100-80GB) |
| Full activation caching | Depends on seq length | Add 50% to model memory |

## Memory Optimization Strategies

### 1. Selective Caching

```python
# Only cache what you need
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "resid_post" in name  # Only residual streams
)

# Or specific layers
names_filter=lambda name: any(f"blocks.{i}" in name for i in [5, 6, 7])
```

### 2. Batch Size Reduction

```python
# If OOM, reduce batch size and accumulate
small_batch_results = []
for mini_batch in torch.split(tokens, batch_size=4):
    result = model(mini_batch)
    small_batch_results.append(result)
```

### 3. Gradient Checkpointing

```python
# For training, trade compute for memory
model.gradient_checkpointing_enable()
```

### 4. Mixed Precision

```python
# Use float16 for inference
model = model.half()  # or model.to(torch.float16)

# Note: Some operations may be numerically unstable
# Keep final computations in float32 if needed
with torch.cuda.amp.autocast():
    logits = model(tokens)
```

### 5. Offloading

```python
# For very large models
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = AutoModel.from_config(config)

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_path,
    device_map="auto",  # Automatically distribute across devices
    offload_folder="offload",  # Offload to disk if needed
)
```

## Cloud Platform Notes

### Modal

```python
import modal

app = modal.App("mech-interp")

# Define GPU requirements
@app.function(gpu="A100")  # or "T4", "A10G"
def run_experiment():
    import torch
    print(f"Running on {torch.cuda.get_device_name(0)}")
    # ... experiment code
```

### Colab

- **Free tier**: T4 (16GB) or older GPUs, limited runtime
- **Pro**: A100 (40GB) available, longer runtimes
- **Limitations**: Session timeouts, need to re-download models

```python
# Check Colab GPU
!nvidia-smi
```

### RunPod / Vast.ai / Lambda Labs

- Rent by the hour
- Check available GPUs and prices on their dashboards
- Use `docker` or `ssh` access
- Pre-install dependencies in custom images for faster startup

### SSH to Remote

```bash
# Check GPU on remote
ssh user@remote "nvidia-smi"

# Run with nohup for long experiments
ssh user@remote "cd project && nohup python train_sae.py > train.log 2>&1 &"

# Port forward for Jupyter
ssh -L 8888:localhost:8888 user@remote
```

## When to Ask About Compute

Ask the user about compute resources when:

1. **Loading large models (7B+)**: "What GPU do you have available?"
2. **Training SAEs**: "Do you have access to a GPU with 24GB+ VRAM?"
3. **Full activation caching**: "For sequences of length 512+, you'll need ~XGB. Is that available?"
4. **Unclear environment**: "Are you running locally, on Colab, or a cloud platform?"
5. **Long-running experiments**: "This will take several hours on an A100. What's your compute budget?"

### Good Phrasing

```
"This experiment needs approximately 16GB VRAM. Before we proceed:
- What GPU do you have access to?
- Are you running locally or on a cloud platform?

If you're on Colab, we can modify the code to work with smaller batches."
```

## Quick Diagnostics

```python
def diagnose_compute():
    """Print diagnostic info for debugging compute issues."""
    print("=== Compute Diagnostics ===")

    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

    # Environment
    import os
    print(f"\nEnvironment type: {detect_compute_environment()['type']}")

    # Python
    import sys
    print(f"Python: {sys.version}")

diagnose_compute()
```
