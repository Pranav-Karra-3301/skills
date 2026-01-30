# ML Project Structure

## Recommended Directory Layout

```
project_name/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset classes
│   │   ├── preprocessing.py      # Data transforms
│   │   └── augmentations.py      # Data augmentation
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── model.py              # Main model
│   │   ├── layers.py             # Custom layers
│   │   └── losses.py             # Custom loss functions
│   ├── training/                 # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── callbacks.py          # Training callbacks
│   │   └── schedulers.py         # LR schedulers
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── logging.py            # Logging setup
│       ├── reproducibility.py    # Seed setting
│       └── metrics.py            # Metric computation
│
├── configs/                      # Configuration files
│   ├── config.yaml               # Main config (Hydra)
│   ├── model/                    # Model configs
│   │   ├── default.yaml
│   │   └── large.yaml
│   ├── data/                     # Data configs
│   │   └── default.yaml
│   └── training/                 # Training configs
│       ├── default.yaml
│       └── finetune.yaml
│
├── scripts/                      # Entry point scripts
│   ├── train.py                  # Training entry
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Inference script
│   └── sweep.py                  # Hyperparameter sweep
│
├── tests/                        # Tests
│   ├── __init__.py
│   ├── conftest.py               # pytest fixtures
│   ├── test_data.py              # Data pipeline tests
│   ├── test_model.py             # Model tests
│   └── test_training.py          # Training tests
│
├── notebooks/                    # Jupyter notebooks
│   ├── exploration.ipynb         # Data exploration
│   └── analysis.ipynb            # Results analysis
│
├── data/                         # Data directory (.gitignored)
│   ├── raw/                      # Original data
│   ├── processed/                # Preprocessed data
│   └── splits/                   # Train/val/test splits
│
├── outputs/                      # Training outputs (.gitignored)
│   └── {experiment_name}/        # Per-experiment outputs
│       ├── checkpoints/          # Model checkpoints
│       ├── logs/                 # Training logs
│       └── predictions/          # Model predictions
│
├── experiments/                  # Experiment tracking (.gitignored)
│   └── mlruns/                   # MLflow tracking (if used)
│
├── CLAUDE.md                     # AI agent context
├── AGENTS.md                     # Multi-agent coordination
├── README.md                     # Project documentation
├── pyproject.toml                # Python project config
├── requirements.txt              # Dependencies (or use pyproject.toml)
├── .gitignore                    # Git ignore patterns
├── .env.example                  # Environment variable template
└── Makefile                      # Common commands (optional)
```

## CLAUDE.md Template for ML Projects

```markdown
# Project: [Name]

## Overview
[Brief description of the ML project, research question, or objective]

## Compute Environment
- **GPU**: [GPU model, e.g., "NVIDIA A100 40GB"]
- **CUDA**: [Version, e.g., "11.8"]
- **Framework**: [e.g., "PyTorch 2.1"]
- **Memory Constraints**: [Any known limitations]

## Architecture
- **Model**: [Model type/architecture]
- **Input**: [Input format, shape]
- **Output**: [Output format, expected shape]
- **Key Components**: [List main modules]

## Data
- **Source**: [Where data comes from]
- **Format**: [Data format]
- **Size**: [Approximate dataset size]
- **Splits**: [Train/val/test split ratios]
- **Preprocessing**: [Key preprocessing steps]

## Current Experiment
- **Goal**: [What we're trying to achieve]
- **Hypothesis**: [What we expect]
- **Status**: [In progress / Completed / Blocked]
- **Key Hyperparameters**:
  - Learning rate: [value]
  - Batch size: [value]
  - Epochs: [value]
- **Results**: [Current best metrics]

## Training
- **Entry Point**: `python scripts/train.py`
- **Config**: `configs/config.yaml`
- **Tracking**: [W&B project name / MLflow experiment]

## Key Files
- `src/models/model.py` - Main model definition
- `src/data/dataset.py` - Dataset implementation
- `src/training/trainer.py` - Training loop
- `configs/config.yaml` - Main configuration

## Known Issues / TODOs
- [ ] [Issue 1]
- [ ] [Issue 2]

## Reproducibility
- **Seed**: [Random seed used]
- **Deterministic**: [Yes/No]
- **Environment**: See `requirements.txt` or `pyproject.toml`
```

## AGENTS.md Template for ML Projects

```markdown
# Multi-Agent Coordination

## Agent Roles

### Research Agent
**Responsibility**: Literature review, experiment design
**Access**: Read-only to codebase, web access for papers
**Outputs**: Experiment proposals in `docs/proposals/`

### Training Agent
**Responsibility**: Running experiments, monitoring training
**Access**: Full access to `src/`, `scripts/`, `configs/`
**Outputs**: Trained models in `outputs/`, logs in W&B

### Analysis Agent
**Responsibility**: Results analysis, visualization
**Access**: Read access to `outputs/`, write to `notebooks/`
**Outputs**: Figures, analysis notebooks

## Communication Protocol

### Experiment Handoff
1. Research Agent creates proposal in `docs/proposals/{date}_{name}.md`
2. Training Agent implements and runs experiment
3. Training Agent updates experiment status in CLAUDE.md
4. Analysis Agent reviews results and updates proposal with findings

### Checkpoint Format
When saving intermediate work:
```yaml
checkpoint:
  agent: [agent name]
  timestamp: [ISO timestamp]
  status: [in_progress | completed | blocked]
  next_steps: [list of next actions]
  artifacts: [list of created files]
```

## Shared Resources

### Experiment Tracking
- W&B Project: [project-name]
- All agents should log with consistent tags

### File Ownership
| Directory | Primary Agent | Notes |
|-----------|---------------|-------|
| `src/` | Training | Core code |
| `configs/` | Training | Hyperparameters |
| `notebooks/` | Analysis | Exploration |
| `docs/` | Research | Proposals, notes |

## Conflict Resolution
1. Never overwrite another agent's in-progress work
2. Use git branches for parallel experiments
3. Coordinate via updates to CLAUDE.md
```

## .gitignore for ML Projects

```gitignore
# Data
data/
*.csv
*.parquet
*.json
!configs/*.json
*.pkl
*.npy
*.npz
*.h5
*.hdf5

# Model checkpoints
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
checkpoints/
outputs/
experiments/

# Logs and tracking
logs/
wandb/
mlruns/
*.log
tensorboard/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Secrets
.env.local
*.pem
*.key
credentials.json
```

## pyproject.toml Template

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project-name"
version = "0.1.0"
description = "ML research project"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
tracking = [
    "wandb>=0.15",
    "mlflow>=2.0",
]
nlp = [
    "transformers>=4.30",
    "tokenizers>=0.13",
    "datasets>=2.14",
    "peft>=0.5",
    "bitsandbytes>=0.41",
]
vision = [
    "torchvision>=0.15",
    "albumentations>=1.3",
    "opencv-python>=4.8",
]

[project.scripts]
train = "scripts.train:main"
evaluate = "scripts.evaluate:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.black]
line-length = 88
target-version = ["py39", "py310", "py311"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]  # Line length handled by black

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

## Makefile Template

```makefile
.PHONY: install train test lint clean

# Install dependencies
install:
	pip install -e ".[dev,tracking]"
	pre-commit install

# Training
train:
	python scripts/train.py

train-debug:
	python scripts/train.py training.epochs=1 data.debug=true

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Linting
lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

# Cleaning
clean:
	rm -rf outputs/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-checkpoints:
	find outputs/ -name "*.pt" -mtime +7 -delete
```
