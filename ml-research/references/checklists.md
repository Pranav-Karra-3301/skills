# ML Research Checklists

## New Project Checklist

### Environment Setup
- [ ] Python virtual environment created (venv/conda)
- [ ] GPU detected and working (`nvidia-smi`, `torch.cuda.is_available()`)
- [ ] CUDA and cuDNN versions compatible with framework
- [ ] Required packages installed with pinned versions
- [ ] `requirements.txt` or `pyproject.toml` created

### Project Structure
- [ ] Source code in `src/` with proper modules
- [ ] Configuration files in `configs/`
- [ ] Training scripts in `scripts/`
- [ ] Tests in `tests/`
- [ ] Data directory (gitignored)
- [ ] Outputs directory (gitignored)

### Documentation
- [ ] `README.md` with setup instructions
- [ ] `CLAUDE.md` with project context
- [ ] `.env.example` with required environment variables
- [ ] Docstrings for main functions/classes

### Version Control
- [ ] Git repository initialized
- [ ] `.gitignore` excludes data, checkpoints, logs, secrets
- [ ] Initial commit with project structure

### Experiment Tracking
- [ ] W&B or MLflow configured
- [ ] Project/experiment name defined
- [ ] Logging integrated into training code

---

## Pre-Training Checklist

### Data
- [ ] Data loaded and inspected
- [ ] Train/val/test splits created
- [ ] No data leakage between splits verified
- [ ] Data statistics computed (mean, std for normalization)
- [ ] Data loaders created and tested
- [ ] Batch shapes verified

### Model
- [ ] Model architecture defined
- [ ] Model can forward pass without error
- [ ] Output shapes match expected
- [ ] Parameter count reasonable for available GPU memory
- [ ] Weights initialized properly

### Training Setup
- [ ] Loss function appropriate for task
- [ ] Optimizer selected (AdamW usually good default)
- [ ] Learning rate set (start conservative: 1e-4)
- [ ] Learning rate scheduler configured
- [ ] Gradient clipping enabled if needed

### Reproducibility
- [ ] Random seeds set (random, numpy, torch)
- [ ] Data loader workers have seed
- [ ] Deterministic mode enabled if needed
- [ ] Config/hyperparameters logged

### Logging & Checkpointing
- [ ] Metrics logged to experiment tracker
- [ ] Training loss logged per step/epoch
- [ ] Validation metrics logged
- [ ] Checkpoints saved periodically
- [ ] Best model checkpoint saved
- [ ] Early stopping configured (optional)

### Memory & Performance
- [ ] Batch size fits in GPU memory
- [ ] Mixed precision enabled if beneficial
- [ ] Gradient accumulation configured if needed
- [ ] Data loading not bottleneck (enough workers)

### Validation
- [ ] Validation runs after each epoch
- [ ] Validation metrics match task objective
- [ ] Model in eval mode during validation
- [ ] No gradients computed during validation

---

## Pre-Finetuning Checklist

### Base Model
- [ ] Pretrained model loaded correctly
- [ ] Model architecture matches expectations
- [ ] Pretrained weights verified (not random)
- [ ] Model tokenizer/preprocessor loaded

### Finetuning Strategy
- [ ] Learning rate reduced (10-100x smaller than training)
- [ ] Layers to freeze/unfreeze decided
- [ ] LoRA/adapter config set (if using)
- [ ] Warmup steps configured

### Data
- [ ] Finetuning data formatted correctly
- [ ] Data compatible with model input format
- [ ] Tokenization/preprocessing matches pretraining
- [ ] No pretraining data leaked into finetuning (if applicable)

### Memory Optimization (for LLMs)
- [ ] Quantization applied if needed (4-bit, 8-bit)
- [ ] Gradient checkpointing enabled for large models
- [ ] LoRA rank set appropriately
- [ ] Effective batch size reasonable with accumulation

### Monitoring
- [ ] Watch for catastrophic forgetting
- [ ] Monitor both finetuning and original task performance
- [ ] Learning rate schedule appropriate for few epochs

---

## Pre-Publication Checklist

### Reproducibility
- [ ] All random seeds documented
- [ ] Full environment exported (`pip freeze`, conda env)
- [ ] All hyperparameters documented
- [ ] Data preprocessing steps documented
- [ ] Training command/script documented
- [ ] Hardware used documented (GPU, memory)

### Results
- [ ] Results averaged over multiple runs (3-5)
- [ ] Standard deviation/error bars reported
- [ ] Statistical significance tested if needed
- [ ] Baselines reproduced (not just copied from papers)
- [ ] Results match claims in text

### Code
- [ ] Code cleaned and commented
- [ ] Dead code removed
- [ ] Hardcoded paths removed
- [ ] Secrets/credentials removed
- [ ] License added
- [ ] README with usage instructions

### Figures
- [ ] Publication-quality (300+ DPI)
- [ ] Colorblind-friendly colors
- [ ] Legible fonts and labels
- [ ] Consistent styling across figures
- [ ] PDF format for vector graphics

### Paper
- [ ] All figures referenced in text
- [ ] All tables referenced in text
- [ ] Citations complete and correct
- [ ] No self-plagiarism
- [ ] Acknowledgments complete

---

## Debugging Checklist

### Model Not Learning
- [ ] Learning rate appropriate (try 10x smaller/larger)
- [ ] Loss function correct for task
- [ ] Labels encoded correctly
- [ ] No NaN/Inf in gradients
- [ ] Gradients flowing to all layers
- [ ] Data not all zeros/constant
- [ ] Model in train mode
- [ ] Optimizer.step() being called
- [ ] Gradients not cleared before step

### Loss Exploding
- [ ] Learning rate too high → reduce
- [ ] No gradient clipping → add clipping
- [ ] Data has outliers → normalize/clip
- [ ] Numerical instability → use mixed precision carefully

### Loss NaN
- [ ] Division by zero in loss
- [ ] Log of zero/negative
- [ ] Overflow in exponential
- [ ] Corrupted data
- [ ] Learning rate too high

### GPU Out of Memory
- [ ] Reduce batch size
- [ ] Enable gradient checkpointing
- [ ] Enable mixed precision
- [ ] Use gradient accumulation
- [ ] Clear cache between batches (`torch.cuda.empty_cache()`)
- [ ] Check for memory leaks (detach tensors)

### Training Too Slow
- [ ] GPU actually being used
- [ ] Data loading bottleneck (increase workers)
- [ ] Move data to GPU before loop
- [ ] Enable cuDNN benchmarking
- [ ] Use compiled models (torch.compile)

### Overfitting
- [ ] Add regularization (dropout, weight decay)
- [ ] Add data augmentation
- [ ] Reduce model size
- [ ] Add early stopping
- [ ] Get more training data

### Results Not Reproducible
- [ ] All seeds set (random, numpy, torch, cuda)
- [ ] Deterministic mode enabled
- [ ] Same hardware/drivers
- [ ] Same library versions
- [ ] Data loader shuffle seeded

---

## Gap Analysis Template

```markdown
## Project: [Name]
## Date: [Date]
## Analyst: [Name]

### Environment
| Check | Status | Notes |
|-------|--------|-------|
| GPU available | ✅/❌ | |
| CUDA compatible | ✅/❌ | |
| Framework installed | ✅/❌ | |
| Memory sufficient | ✅/❌ | |

### Reproducibility
| Check | Status | Notes |
|-------|--------|-------|
| Seeds set | ✅/❌ | |
| Environment exported | ✅/❌ | |
| Config versioned | ✅/❌ | |
| Data versioned | ✅/❌ | |

### Data
| Check | Status | Notes |
|-------|--------|-------|
| Splits defined | ✅/❌ | |
| No leakage verified | ✅/❌ | |
| Preprocessing documented | ✅/❌ | |
| Data validated | ✅/❌ | |

### Training
| Check | Status | Notes |
|-------|--------|-------|
| Experiment tracking | ✅/❌ | |
| Checkpointing | ✅/❌ | |
| Validation loop | ✅/❌ | |
| Early stopping | ✅/❌ | |
| Metrics logged | ✅/❌ | |

### Code Quality
| Check | Status | Notes |
|-------|--------|-------|
| Tests exist | ✅/❌ | |
| No hardcoded paths | ✅/❌ | |
| Documentation | ✅/❌ | |
| Git ignored properly | ✅/❌ | |

### Priority Fixes
1. [Highest priority issue]
2. [Second priority issue]
3. [Third priority issue]

### Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

---

## Quick Reference Card

### Essential Seeds
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### Essential Logging
```python
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "lr": optimizer.param_groups[0]["lr"],
    "epoch": epoch,
})
```

### Essential Checkpointing
```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, "checkpoint.pt")
```

### Essential Validation Loop
```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
        # compute metrics
model.train()
```
