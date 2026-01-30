---
name: ml-research
description: |
  Comprehensive skill for ML/AI research experiments and finetuning. Use when:
  (1) Setting up new ML research project ("create ML project", "init experiment")
  (2) Finetuning models ("finetune LLM", "adapt pretrained model", "LoRA", "QLoRA")
  (3) Training from scratch ("train model", "run experiment")
  (4) Debugging ML issues ("model not converging", "loss exploding", "GPU OOM")
  (5) Setting up experiment tracking ("add W&B", "setup MLflow")
  (6) Optimizing GPU usage ("batch size tuning", "memory optimization")
  (7) Creating visualizations ("plot training curves", "confusion matrix")
  (8) Auditing ML code ("check reproducibility", "review experiment")
  Triggers: "ML", "machine learning", "deep learning", "training", "finetuning",
  "PyTorch", "TensorFlow", "experiment", "GPU", "CUDA", "model", "neural network",
  "W&B", "MLflow", "reproducibility", "learning rate", "checkpoint", "epoch"
---

# ML Research Skill

## Overview

This skill provides comprehensive support for ML/AI research experiments, finetuning, and training. It helps with:

- **System Detection**: GPU, CUDA, memory, framework versions
- **Project Setup**: Cookiecutter-style ML project structure with proper documentation
- **Experiment Tracking**: W&B, MLflow, TensorBoard configuration
- **Best Practices**: Reproducibility, data handling, training loops
- **Debugging**: Common mistakes, memory issues, convergence problems
- **Visualization**: Publication-quality plots, colorblind-friendly palettes

## Core Workflow

### Phase 1: System Detection

Before any ML work, detect the compute environment:

**1. GPU Detection**
```bash
# Check for NVIDIA GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"

# Check CUDA version
nvcc --version 2>/dev/null | grep "release" || echo "CUDA not found"

# Check cuDNN (if accessible)
cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 || echo "cuDNN version not directly accessible"
```

**2. Python Environment**
```bash
# Python version
python3 --version

# Virtual environment detection
echo $VIRTUAL_ENV $CONDA_DEFAULT_ENV

# Check for common ML frameworks
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null
python3 -c "import jax; print(f'JAX {jax.__version__}')" 2>/dev/null
```

**3. Memory Detection**
```bash
# System RAM
free -h 2>/dev/null || sysctl hw.memsize 2>/dev/null

# GPU memory (via torch if available)
python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" 2>/dev/null
```

**4. ML Tools Detection**
```bash
# Check for experiment tracking
python3 -c "import wandb; print(f'W&B {wandb.__version__}')" 2>/dev/null
python3 -c "import mlflow; print(f'MLflow {mlflow.__version__}')" 2>/dev/null

# Check for config management
python3 -c "import hydra; print(f'Hydra {hydra.__version__}')" 2>/dev/null
```

**Run comprehensive detection:**
```bash
python3 scripts/detect_system.py
```

See [references/gpu-management.md](references/gpu-management.md) for compute optimization.

### Phase 2: Task Understanding

Identify the ML task type:

| Task Type | Key Indicators | Critical Checks |
|-----------|---------------|-----------------|
| **Training from scratch** | New model, random init | Data size, compute budget |
| **Finetuning** | Pretrained model, adaptation | Base model selection, LR schedule |
| **Evaluation** | Metrics, benchmarking | No data leakage, proper splits |
| **Inference** | Deployment, serving | Batch size, latency requirements |

**Model Domain Detection:**
- **Vision**: Images, CNNs, ViT, augmentation
- **NLP/LLM**: Text, transformers, tokenization
- **Multimodal**: Vision + language, CLIP-style
- **Tabular**: Structured data, gradient boosting often better
- **RL**: Environment, policy, reward

**Scale Assessment:**
- **Local**: Single GPU, fits in memory
- **Multi-GPU**: DataParallel or DDP
- **Distributed**: Multiple nodes, FSDP/DeepSpeed
- **Cloud**: Managed services (SageMaker, Vertex AI)

### Phase 3: Gap Analysis

Check the project against ML best practices:

#### Reproducibility Checklist
- [ ] Random seeds set (Python, NumPy, PyTorch/TF)
- [ ] Deterministic mode enabled (if needed)
- [ ] Environment captured (requirements.txt, conda env)
- [ ] Data versioning (DVC, direct hashing)
- [ ] Code versioning (git commit tracked in experiments)

#### Data Handling Checklist
- [ ] Train/val/test splits defined before any preprocessing
- [ ] No data leakage between splits
- [ ] Data validation (schema, distributions)
- [ ] Preprocessing pipeline reproducible
- [ ] Augmentations only on training data

#### Training Checklist
- [ ] Experiment tracking configured (W&B, MLflow)
- [ ] Logging (not just print statements)
- [ ] Checkpointing enabled
- [ ] Early stopping configured
- [ ] Metrics defined and tracked

#### Documentation Checklist
- [ ] CLAUDE.md exists with project context
- [ ] AGENTS.md for multi-agent work
- [ ] README with setup instructions
- [ ] Experiment notes/logs

**Run validation:**
```bash
python3 scripts/validate_experiment.py
```

See [references/common-mistakes.md](references/common-mistakes.md) for detailed issues.

### Phase 4: Project Setup / Code Generation

#### Recommended Project Structure

When setting up an ML project, discuss with the user what structure fits their needs. A typical ML research project includes:
```
my_experiment/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   └── preprocessing.py # Data transforms
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py         # Model definitions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── losses.py        # Custom losses
│   └── utils/
│       ├── __init__.py
│       ├── logging.py       # Logging utilities
│       └── reproducibility.py # Seed setting
├── configs/
│   ├── config.yaml          # Main Hydra config
│   ├── model/
│   │   └── default.yaml
│   ├── data/
│   │   └── default.yaml
│   └── training/
│       └── default.yaml
├── scripts/
│   ├── train.py             # Main training entry
│   ├── evaluate.py          # Evaluation script
│   └── inference.py         # Inference script
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── conftest.py          # pytest fixtures
├── notebooks/
│   └── exploration.ipynb
├── data/                    # .gitignored
├── outputs/                 # .gitignored
├── experiments/             # .gitignored
├── CLAUDE.md
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
└── .env.example
```

See [references/project-structure.md](references/project-structure.md) for templates.

#### Configuration Generation

**Hydra Config Template:**
```yaml
# configs/config.yaml
defaults:
  - model: default
  - data: default
  - training: default
  - _self_

seed: 42
experiment_name: ${now:%Y-%m-%d_%H-%M-%S}

hydra:
  run:
    dir: outputs/${experiment_name}
  sweep:
    dir: outputs/sweeps/${experiment_name}
```

#### Experiment Tracking Setup

**W&B Integration:**
```python
import wandb

wandb.init(
    project="my-project",
    config=dict(cfg),  # Hydra config
    tags=["experiment-type"],
    notes="Description of this run"
)

# Log metrics
wandb.log({"loss": loss, "accuracy": acc})

# Log artifacts
wandb.save("model.pt")
```

**MLflow Integration:**
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # or remote URI
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_params(dict(cfg))
    mlflow.log_metrics({"loss": loss, "accuracy": acc})
    mlflow.log_artifact("model.pt")
```

See [references/experiment-tracking.md](references/experiment-tracking.md) for detailed setup.

### Phase 5: Validation & Verification

Before training, run sanity checks:

**1. Data Pipeline Verification**
```python
# Check data loader
batch = next(iter(train_loader))
print(f"Batch shape: {batch['x'].shape}")
print(f"Labels: {batch['y'].unique()}")

# Verify no leakage
train_ids = set(train_dataset.ids)
val_ids = set(val_dataset.ids)
assert len(train_ids & val_ids) == 0, "Data leakage detected!"
```

**2. Model Sanity Checks**
```python
# Verify forward pass
model.eval()
with torch.no_grad():
    output = model(batch['x'].to(device))
    print(f"Output shape: {output.shape}")

# Check gradient flow
model.train()
output = model(batch['x'].to(device))
loss = criterion(output, batch['y'].to(device))
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: No gradient for {name}")
```

**3. Reproducibility Test**
```python
# Run twice with same seed, verify identical results
results1 = train_one_step(seed=42)
results2 = train_one_step(seed=42)
assert torch.allclose(results1, results2), "Not reproducible!"
```

**4. Memory Estimation**
```python
# Estimate memory usage
def estimate_memory(model, batch_size, input_size):
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    grads = params  # Gradients same size as params
    optimizer = params * 2  # Adam has 2 states per param
    activations = batch_size * input_size * 4  # Rough estimate

    total_gb = (params + grads + optimizer + activations) / 1e9
    return total_gb
```

See [references/testing.md](references/testing.md) for comprehensive test examples.

## Language-Specific Setup

### PyTorch Project

**Essential imports:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For complete determinism (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Training loop template:**
```python
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0

    for batch in loader:
        x, y = batch['x'].to(device), batch['y'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

### TensorFlow/Keras Project

**Essential setup:**
```python
import tensorflow as tf
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### JAX/Flax Project

**Essential setup:**
```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

# JAX uses explicit PRNG
key = jax.random.PRNGKey(42)
```

## Finetuning Best Practices

See [references/finetuning.md](references/finetuning.md) for comprehensive guide.

### Quick Reference

| Aspect | Training from Scratch | Finetuning |
|--------|----------------------|------------|
| Learning Rate | 1e-3 to 1e-4 | 1e-5 to 1e-6 (10-100x smaller) |
| Epochs | Many (100+) | Few (3-10) |
| Data Size | Large | Can be small |
| Compute | High | Lower |

### LLM Finetuning

**LoRA Setup (PEFT):**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,              # Rank
    lora_alpha=32,     # Scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

**QLoRA (4-bit):**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

## GPU Management

See [references/gpu-management.md](references/gpu-management.md) for detailed optimization.

### Quick Memory Estimation

```python
def estimate_training_memory_gb(
    model_params_billions: float,
    batch_size: int,
    seq_length: int = 512,
    precision: str = "fp16"  # fp32, fp16, bf16
) -> float:
    """Rough estimate of GPU memory for training."""
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4

    # Model parameters
    param_memory = model_params_billions * 1e9 * bytes_per_param

    # Gradients (same size as params)
    grad_memory = param_memory

    # Optimizer states (Adam: 2x params in fp32)
    optimizer_memory = model_params_billions * 1e9 * 4 * 2

    # Activations (rough estimate)
    activation_memory = batch_size * seq_length * 4096 * bytes_per_param

    total_bytes = param_memory + grad_memory + optimizer_memory + activation_memory
    return total_bytes / 1e9
```

### Batch Size Tuning

```python
def find_optimal_batch_size(model, sample_input, device, start_batch=1):
    """Binary search for maximum batch size."""
    batch_size = start_batch

    while True:
        try:
            # Create batch
            batch = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
            batch = batch.to(device)

            # Forward + backward
            model.zero_grad()
            output = model(batch)
            loss = output.mean()
            loss.backward()

            # Success, try larger
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2

            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                optimal = batch_size // 2
                print(f"OOM at {batch_size}, optimal: {optimal}")
                return optimal
            raise
```

## Visualization

See [references/visualization.md](references/visualization.md) for complete guide.

### Publication-Quality Matplotlib

```python
import matplotlib.pyplot as plt

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
```

### Colorblind-Friendly Palettes

```python
# Recommended palettes
COLORBLIND_SAFE = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377']

# Or use built-in
plt.style.use('seaborn-v0_8-colorblind')
# Or: cmap = plt.cm.viridis / plt.cm.cividis
```

### Standard ML Plots

```python
def plot_training_curves(history, save_path=None):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curves')

    # Metric
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy Curves')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

## Common Tasks

### "Create a new ML project"
1. Run system detection (Phase 1)
2. Ask about project type (vision, NLP, tabular, etc.)
3. Discuss project structure with user (see [references/project-structure.md](references/project-structure.md))
4. Create necessary directories and files based on user's needs
5. Set up experiment tracking based on preference
6. Generate CLAUDE.md and AGENTS.md with project-specific context

### "Debug why my model isn't converging"
1. Check learning rate (too high? too low?)
2. Check data normalization
3. Check for NaN/Inf in gradients
4. Verify loss function matches task
5. Check batch size isn't too small
6. Look for data issues (labels correct? balanced?)

### "Set up W&B for my project"
1. Check if W&B is installed (`pip install wandb`)
2. Verify login status (`wandb login`)
3. Generate initialization code
4. Add logging to training loop
5. Set up sweep configuration if requested

### "Optimize GPU memory usage"
1. Run memory estimation
2. Enable mixed precision (AMP)
3. Add gradient accumulation
4. Enable gradient checkpointing if needed
5. Consider batch size reduction

### "Create publication figures"
1. Set publication rcParams
2. Use colorblind-friendly palette
3. Export as PDF (300+ DPI)
4. Verify text is readable at intended size

### "Review my experiment for common mistakes"
1. Run `scripts/validate_experiment.py`
2. Check reproducibility setup
3. Verify no data leakage
4. Check logging and checkpointing
5. Review hyperparameters against literature

## When to Alert the User

**Always notify for these issues:**

1. **GPU unavailable**
   > "No GPU detected. Training will run on CPU and be significantly slower."

2. **Memory constraints**
   > "Model (~X GB) may not fit in GPU (~Y GB). Consider reducing batch size or enabling gradient checkpointing."

3. **Missing reproducibility**
   > "No random seeds set. Results will not be reproducible. Add seed setting?"

4. **Data leakage detected**
   > "WARNING: Test data IDs found in training set. This will invalidate your results."

5. **Framework conflicts**
   > "Both PyTorch and TensorFlow detected. Which is the primary framework for this project?"

6. **Deterministic mode tradeoffs**
   > "Enabling deterministic mode for reproducibility. Note: This may slow training by ~10%."

7. **Checkpoint accumulation**
   > "Found 15GB of checkpoints in outputs/. Clean up old checkpoints?"

8. **Missing experiment tracking**
   > "No experiment tracking detected. Set up W&B or MLflow for proper logging?"

## Quality Checklist

Before considering ML setup complete:

- [ ] System detection run and documented
- [ ] Reproducibility seeds set
- [ ] Data splits verified (no leakage)
- [ ] Experiment tracking configured
- [ ] Checkpointing enabled
- [ ] Logging (not print) used throughout
- [ ] CLAUDE.md documents current experiment
- [ ] .gitignore excludes data, checkpoints, outputs
- [ ] Tests exist for data pipeline and model
- [ ] Memory usage estimated and feasible

See [references/checklists.md](references/checklists.md) for complete checklists.
