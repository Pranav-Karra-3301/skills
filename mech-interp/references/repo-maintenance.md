# Repository Maintenance for Mech Interp Projects

Templates and best practices for organizing mechanistic interpretability research projects.

## Recommended Project Structure

```
mech-interp-project/
├── CLAUDE.md              # Agent instructions (for AI assistants)
├── AGENTS.md              # Multi-agent coordination (if using multiple agents)
├── README.md              # Project overview and setup
├── pyproject.toml         # Dependencies and project config
│
├── notebooks/             # Exploration and analysis notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_circuit_analysis.ipynb
│   └── figures/           # Notebook-generated figures
│
├── src/
│   ├── __init__.py
│   ├── models/            # Model loading and configuration
│   │   ├── __init__.py
│   │   └── load.py        # Centralized model loading
│   │
│   ├── experiments/       # Experiment scripts
│   │   ├── __init__.py
│   │   ├── patching.py
│   │   └── circuit_discovery.py
│   │
│   ├── analysis/          # Analysis utilities
│   │   ├── __init__.py
│   │   ├── attribution.py
│   │   └── statistics.py
│   │
│   └── visualization/     # Plotting code
│       ├── __init__.py
│       ├── attention.py
│       └── features.py
│
├── data/
│   ├── activations/       # Cached activations
│   ├── results/           # Experiment outputs
│   └── datasets/          # Evaluation datasets
│
├── configs/               # Experiment configurations
│   ├── gpt2_small.yaml
│   └── sae_training.yaml
│
├── scripts/               # CLI scripts
│   ├── run_experiment.py
│   └── train_sae.py
│
└── tests/                 # Validation tests
    ├── test_patching.py
    └── test_attribution.py
```

## CLAUDE.md Template

```markdown
# Project: [Name]

## Overview

[One sentence describing the research goal]

## Active Research Questions

1. [Primary question you're investigating]
2. [Secondary question]
3. [Open problem you're working toward]

## Key Files

### Models and Data
- `src/models/load.py` - Model loading with standard configs
- `data/activations/` - Cached activations (git-ignored, regenerate with `scripts/cache_activations.py`)

### Main Experiments
- `notebooks/02_circuit_analysis.ipynb` - Primary analysis notebook
- `src/experiments/patching.py` - Activation patching utilities

### Results
- `data/results/` - Experiment outputs (JSON, CSVs)
- `notebooks/figures/` - Generated visualizations

## Compute Resources

- **GPU**: [e.g., "A100 via Modal", "Local RTX 3090", "Colab Pro"]
- **Memory constraints**: [e.g., "16GB VRAM limits batch size to 32"]
- **Activation caching**: [e.g., "Full cache for GPT-2-small uses ~2GB per 1k tokens"]

## Current Experiments

| Experiment | Status | Notes |
|------------|--------|-------|
| [Name] | In progress / Complete / Blocked | [Brief note] |

## Conventions

### Code Style
- Use `torch.no_grad()` for all inference
- Always specify device explicitly
- Use einops for tensor operations when clearer
- Type hints for function signatures

### Naming
- Hook points: Follow TransformerLens convention
- Experiments: `{date}_{description}` e.g., `2024_01_induction_patching`
- Cached activations: `{model}_{layer}_{dataset}_{n_tokens}.pt`

### Caching
- Cache activations to `data/activations/`
- Include metadata JSON alongside cached tensors
- Clear cache before final experiments (reproducibility)

## Common Commands

```bash
# Run experiments
python scripts/run_experiment.py --config configs/patching.yaml

# Train SAE
python scripts/train_sae.py --model gpt2-small --layer 8

# Generate figures
python -m src.visualization.generate_all

# Run tests
pytest tests/
```

## Important Notes

- [Any gotchas or warnings for future work]
- [Things that took time to figure out]
- [Decisions made and why]

## Related Work

- [Links to relevant repos, not papers]
- [Internal docs or notes]
```

## AGENTS.md Template

For projects using multiple AI agents or collaborators:

```markdown
# Multi-Agent Coordination

## Active Agents

| Agent | Role | Working On |
|-------|------|------------|
| Main | Experiment design and analysis | Circuit discovery |
| SAE | SAE training and feature analysis | Training run 3 |

## Shared Resources

### Cached Activations
- `data/activations/gpt2_resid_pre_L8.pt` - Ready for SAE training
- `data/activations/gpt2_full_cache.pt` - Full model cache (2GB)

### Results
- `data/results/patching_2024_01.json` - Patching experiment results

## Conventions

### File Ownership
- Notebooks: Single agent ownership, coordinate before editing others'
- `src/`: Shared, use git branches for major changes
- `data/`: Shared, append-only for results

### Communication
- Update this file when starting/finishing major tasks
- Note dependencies: "Blocked on X" or "Produces Y"

### Git Workflow
- Branch per experiment: `exp/{name}`
- Merge to main when validated
- Don't commit large activation files (use `.gitignore`)

## Current Blockers

- [ ] Need more GPU time for SAE training
- [ ] Waiting for [X] to complete before [Y]

## Completed Handoffs

| Date | From | To | What |
|------|------|-----|------|
| [Date] | [Agent] | [Agent] | [Description] |
```

## Configuration Management

### Experiment Config (YAML)

```yaml
# configs/experiment.yaml
experiment:
  name: "induction_head_patching"
  description: "Patch induction heads to measure causal effect"

model:
  name: "gpt2-small"
  device: "cuda"
  dtype: "float32"

data:
  dataset: "induction_sequences"
  n_samples: 1000
  seq_length: 50

patching:
  layers: [5, 6, 7, 8]
  heads: "all"
  positions: "all"
  metric: "logit_diff"

output:
  results_dir: "data/results/induction_patching"
  save_activations: false
  figures: true
```

### Loading Config

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

# Usage
config = load_config("configs/experiment.yaml")
model_name = config["model"]["name"]
```

## Code Organization Guidelines

### Separate Concerns

```python
# src/data/loading.py - Data loading ONLY
def load_dataset(name: str, n_samples: int) -> Dataset:
    ...

# src/models/load.py - Model loading ONLY
def load_model(name: str, device: str, dtype: torch.dtype) -> HookedTransformer:
    ...

# src/experiments/patching.py - Experiment logic ONLY
def run_patching_experiment(model, dataset, config) -> Results:
    ...

# src/analysis/metrics.py - Analysis ONLY
def compute_patching_metrics(results: Results) -> dict:
    ...
```

### Use Configs for Hyperparameters

```python
# BAD: Hardcoded values scattered in code
def train_sae():
    lr = 3e-4
    batch_size = 4096
    ...

# GOOD: Config-driven
def train_sae(config: dict):
    lr = config["training"]["lr"]
    batch_size = config["training"]["batch_size"]
    ...
```

### Version Control Best Practices

**.gitignore for mech interp projects:**

```gitignore
# Large files
data/activations/*.pt
data/activations/*.safetensors
*.ckpt
*.bin

# But keep metadata
!data/activations/*.json

# Notebooks outputs (optional - some prefer to track)
notebooks/.ipynb_checkpoints/

# Experiment outputs (keep structure, ignore large files)
data/results/*.pt
!data/results/*.json
!data/results/*.csv

# Model weights (download, don't commit)
models/
*.safetensors

# Python
__pycache__/
*.pyc
.pytest_cache/

# Environment
.env
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

### Notebook Best Practices

1. **Clear outputs before committing** (reduces git bloat)
2. **Number notebooks** for ordering: `01_exploration.ipynb`
3. **Extract reusable code** to `src/` modules
4. **Document at top**: What question does this notebook answer?
5. **Checkpoint results**: Save intermediate results to JSON/CSV

```python
# At top of notebook
"""
# Induction Head Analysis

**Question**: Which heads in GPT-2-small are induction heads?

**Method**: Pattern matching + patching validation

**Key findings**: [Updated after analysis]
"""
```

## Experiment Tracking

### Lightweight Logging

```python
import json
from datetime import datetime
from pathlib import Path

def log_experiment(name: str, config: dict, results: dict, notes: str = ""):
    """Log experiment results to JSON."""
    log_entry = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "notes": notes,
    }

    log_path = Path("data/results") / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    log_path.parent.mkdir(exist_ok=True)

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)

    print(f"Logged to {log_path}")
    return log_path
```

### Weights & Biases Integration

```python
import wandb

def run_experiment_with_wandb(config: dict):
    wandb.init(
        project="mech-interp",
        config=config,
        tags=["patching", config["model"]["name"]],
    )

    # Run experiment
    results = run_experiment(config)

    # Log results
    wandb.log(results)

    # Log artifacts
    wandb.log({"attention_heatmap": wandb.Image(fig)})

    wandb.finish()
```
