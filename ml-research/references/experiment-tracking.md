# Experiment Tracking

## Overview

Experiment tracking is essential for reproducible ML research. This guide covers the two most popular tools: Weights & Biases (W&B) and MLflow.

## Comparison

| Feature | W&B | MLflow |
|---------|-----|--------|
| Setup Complexity | Simple (hosted) | Medium (self-hosted or Databricks) |
| Cost | Free tier, paid for teams | Open source, free |
| Offline Mode | Yes | Yes |
| UI Quality | Excellent | Good |
| Hyperparameter Sweeps | Built-in | Requires additional setup |
| Model Registry | Yes | Yes |
| Artifacts | Yes | Yes |
| Tables/Datasets | Yes | Limited |
| Collaboration | Excellent | Depends on hosting |

**Recommendation**:
- **W&B** for most projects - better UI, simpler setup, great free tier
- **MLflow** for enterprise, on-prem requirements, or when cost is a concern

## Weights & Biases (W&B)

### Installation and Setup

```bash
pip install wandb

# Login (one-time)
wandb login
```

### Basic Usage

```python
import wandb

# Initialize a run
wandb.init(
    project="my-project",          # Project name (creates if doesn't exist)
    name="experiment-001",          # Optional run name
    config={                        # Hyperparameters
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "model": "resnet50",
    },
    tags=["baseline", "vision"],    # Tags for filtering
    notes="First baseline run",     # Description
    group="ablation-study",         # Group related runs
)

# Access config (useful with Hydra)
config = wandb.config

# Training loop
for epoch in range(config.epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": scheduler.get_last_lr()[0],
    })

# Finish run
wandb.finish()
```

### Logging Different Types

```python
# Scalars (metrics)
wandb.log({"loss": 0.5, "accuracy": 0.95})

# Images
wandb.log({"samples": [wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(images)]})

# Confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_true,
    preds=y_pred,
    class_names=class_names
)})

# Histograms
wandb.log({"gradient_norm": wandb.Histogram(gradients)})

# Tables
table = wandb.Table(columns=["id", "prediction", "ground_truth"])
for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
    table.add_data(i, pred, gt)
wandb.log({"predictions": table})

# Model checkpoint
wandb.save("model.pt")

# Artifacts (for versioning)
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)
```

### Hyperparameter Sweeps

```yaml
# sweep.yaml
program: train.py
method: bayes  # grid, random, bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [16, 32, 64]
  optimizer:
    values: ["adam", "sgd"]
  weight_decay:
    distribution: uniform
    min: 0
    max: 0.1
early_terminate:
  type: hyperband
  min_iter: 5
```

```bash
# Create sweep
wandb sweep sweep.yaml

# Run agents (can run multiple in parallel)
wandb agent username/project/sweep_id
```

```python
# In train.py
import wandb

def train():
    wandb.init()
    config = wandb.config  # Gets parameters from sweep

    model = build_model(config)
    optimizer = get_optimizer(config)

    for epoch in range(100):
        loss = train_epoch(model, optimizer)
        val_acc = validate(model)
        wandb.log({"loss": loss, "val_accuracy": val_acc})

if __name__ == "__main__":
    train()
```

### Integration with Hydra

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to dict for W&B
    wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.get("tags", []),
    )

    # Training code...

    wandb.finish()

if __name__ == "__main__":
    main()
```

### Offline Mode

```python
# Set mode before init
import os
os.environ["WANDB_MODE"] = "offline"

# Or in init
wandb.init(mode="offline")

# Sync later
# wandb sync ./wandb/offline-run-*
```

## MLflow

### Installation and Setup

```bash
pip install mlflow

# Start tracking server (optional, for team use)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Basic Usage

```python
import mlflow
import mlflow.pytorch  # or mlflow.tensorflow

# Set tracking URI (optional, defaults to local)
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite
# mlflow.set_tracking_uri("http://localhost:5000")  # Remote server

# Set experiment
mlflow.set_experiment("my-experiment")

# Start run
with mlflow.start_run(run_name="experiment-001"):
    # Log parameters
    mlflow.log_params({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
    })

    # Training loop
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        # Log metrics (step parameter for x-axis)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts (any file)
    mlflow.log_artifact("config.yaml")

    # Log figures
    fig = plot_training_curves(history)
    mlflow.log_figure(fig, "training_curves.png")
```

### Model Registry

```python
# Register model
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="my-model"
)

# Load model from registry
model_uri = "models:/my-model/Production"
model = mlflow.pytorch.load_model(model_uri)

# Transition model stage
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)
```

### Autologging

```python
# PyTorch Lightning
mlflow.pytorch.autolog()

# Scikit-learn
mlflow.sklearn.autolog()

# XGBoost
mlflow.xgboost.autolog()

# Then train normally - metrics/params logged automatically
```

### Integration with Hydra

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run():
        # Log all config as params
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Training code...

if __name__ == "__main__":
    main()
```

## TensorBoard (Lightweight Alternative)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment-001")

for epoch in range(100):
    loss = train_epoch(model, train_loader)

    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Histograms
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Images
    writer.add_images("samples", images, epoch)

writer.close()

# View: tensorboard --logdir=runs
```

## Best Practices

### 1. Log Everything Reproducibility-Related

```python
wandb.config.update({
    # Training
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs,

    # Model
    "model_name": model.__class__.__name__,
    "num_params": sum(p.numel() for p in model.parameters()),

    # Data
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),

    # Environment
    "seed": seed,
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,

    # Git
    "git_commit": subprocess.getoutput("git rev-parse HEAD"),
})
```

### 2. Use Consistent Naming

```python
# Hierarchical metric names
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "lr": scheduler.get_last_lr()[0],
})
```

### 3. Save Checkpoints as Artifacts

```python
# Save best model
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), "best_model.pt")

    artifact = wandb.Artifact(
        f"model-{wandb.run.id}",
        type="model",
        metadata={"val_accuracy": val_acc, "epoch": epoch}
    )
    artifact.add_file("best_model.pt")
    wandb.log_artifact(artifact)
```

### 4. Use Tags for Organization

```python
wandb.init(
    tags=[
        "baseline",           # Experiment type
        "resnet50",           # Model
        "imagenet-subset",    # Dataset
        "lr-schedule-cosine", # Key hyperparameter
    ]
)
```

### 5. Group Related Runs

```python
# All runs in ablation study
wandb.init(group="lr-ablation-2024-01")

# Or use job_type
wandb.init(
    group="experiment-v2",
    job_type="train"  # train, eval, sweep
)
```
