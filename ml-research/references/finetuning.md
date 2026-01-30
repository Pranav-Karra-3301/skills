# Finetuning Best Practices

## Overview

Finetuning adapts a pretrained model to a new task or domain. This guide covers general finetuning, LLM-specific techniques, and common pitfalls.

## Training vs Finetuning

| Aspect | Training from Scratch | Finetuning |
|--------|----------------------|------------|
| **Learning Rate** | 1e-3 to 1e-4 | 1e-5 to 1e-6 (10-100x smaller) |
| **Epochs** | Many (50-500) | Few (1-10) |
| **Data Required** | Large (millions) | Can be small (hundreds-thousands) |
| **Compute** | High | Lower |
| **Risk** | Underfitting | Catastrophic forgetting |

## Learning Rate Selection

### General Guidelines

```python
# Typical learning rates by task
LEARNING_RATES = {
    # Vision finetuning
    "vision_last_layer": 1e-3,
    "vision_full_model": 1e-5,

    # NLP finetuning
    "bert_classification": 2e-5,
    "bert_ner": 3e-5,

    # LLM finetuning
    "llm_full": 1e-5,
    "llm_lora": 1e-4,
    "llm_qlora": 2e-4,
}
```

### Learning Rate Finder

```python
def find_learning_rate(model, train_loader, criterion, device, min_lr=1e-7, max_lr=10):
    """
    Find optimal learning rate using the LR range test.
    Look for the steepest descent in the loss curve.
    """
    import math

    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)

    lr_mult = (max_lr / min_lr) ** (1 / len(train_loader))
    lrs = []
    losses = []

    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())

        # Update LR
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult

        # Stop if loss explodes
        if loss.item() > 4 * min(losses):
            break

    # Plot and find optimal LR (steepest descent)
    return lrs, losses
```

### Differential Learning Rates

```python
def get_layer_lr_groups(model, base_lr, decay=0.9):
    """
    Apply different learning rates to different layers.
    Earlier layers get smaller LR (more pretrained knowledge).
    """
    parameters = []
    prev_group = None

    for i, (name, param) in enumerate(model.named_parameters()):
        layer_num = i // 4  # Rough grouping

        if prev_group is None or layer_num != prev_group:
            parameters.append({
                "params": [param],
                "lr": base_lr * (decay ** (len(list(model.named_parameters())) - i) // 4)
            })
            prev_group = layer_num
        else:
            parameters[-1]["params"].append(param)

    return parameters

# Usage
optimizer = torch.optim.AdamW(
    get_layer_lr_groups(model, base_lr=1e-4, decay=0.9)
)
```

## Finetuning Strategies

### Strategy 1: Feature Extraction (Freeze Base)

```python
# Freeze all pretrained layers
for param in model.base.parameters():
    param.requires_grad = False

# Only train the head
model.head = nn.Linear(hidden_size, num_classes)
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
```

**When to use**: Small dataset, very different task, quick baseline

### Strategy 2: Gradual Unfreezing

```python
def unfreeze_layers(model, num_layers):
    """Unfreeze last N layers."""
    layers = list(model.children())

    # Freeze all
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Unfreeze last N
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

# Epoch 1-3: Only head
unfreeze_layers(model, 1)
train_epochs(3)

# Epoch 4-6: Last few layers
unfreeze_layers(model, 4)
train_epochs(3)

# Epoch 7+: Full model
for param in model.parameters():
    param.requires_grad = True
train_epochs(10)
```

**When to use**: Limited data, want to preserve pretrained features

### Strategy 3: Full Finetuning

```python
# All parameters trainable
for param in model.parameters():
    param.requires_grad = True

# Use small learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# With learning rate warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
```

**When to use**: Sufficient data, task similar to pretraining

## LLM Finetuning

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank (8-64 typical)
    lora_alpha=32,             # Scaling factor (usually 2x rank)
    target_modules=[           # Which modules to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,         # Dropout for regularization
    bias="none",               # Don't train biases
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Trainable: 0.1% of parameters

# Training is normal after this
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

### QLoRA (Quantized LoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # Normalized float 4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
```

### Supervised Finetuning (SFT)

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",  # Column with formatted text
    max_seq_length=2048,
    packing=True,  # Pack multiple examples per sequence
)

trainer.train()
```

### Data Format for SFT

```python
# Chat format (recommended)
def format_chat(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Instruction format
def format_instruction(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
    }

dataset = dataset.map(format_chat)
```

## Checkpointing

### Save Best and Last

```python
class CheckpointManager:
    def __init__(self, save_dir, keep_last_k=3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.best_metric = float("-inf")
        self.checkpoints = []

    def save(self, model, optimizer, epoch, metric, is_best=False):
        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        }

        path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, path)
        self.checkpoints.append(path)

        # Keep only last K
        while len(self.checkpoints) > self.keep_last_k:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists() and "best" not in str(old_ckpt):
                old_ckpt.unlink()

        # Save best
        if metric > self.best_metric:
            self.best_metric = metric
            best_path = self.save_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            return True
        return False

    def load_best(self, model, optimizer=None):
        path = self.save_dir / "best_model.pt"
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["metric"]
```

### PEFT Model Saving

```python
# Save only LoRA weights (small)
model.save_pretrained("lora-adapter")

# Load adapter onto base model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "lora-adapter")

# Merge for inference (optional, no PEFT needed)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged-model")
```

## Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
        elif self._is_improvement(metric):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, metric):
        if self.mode == "min":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

# Usage
early_stopping = EarlyStopping(patience=5, mode="min")

for epoch in range(max_epochs):
    val_loss = validate(model)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Common Pitfalls

### 1. Learning Rate Too High

**Symptom**: Loss spikes, NaN values, performance degrades
```python
# Fix: Start with very low LR
lr = 1e-6  # Then gradually increase if stable
```

### 2. Catastrophic Forgetting

**Symptom**: Performance on pretrained tasks drops significantly
```python
# Fix 1: Lower learning rate
# Fix 2: Use LoRA or other parameter-efficient methods
# Fix 3: Mix in pretrained data
```

### 3. Overfitting to Small Dataset

**Symptom**: Train loss keeps dropping, val loss increases
```python
# Fix 1: More regularization
optimizer = AdamW(model.parameters(), weight_decay=0.1)

# Fix 2: Dropout
lora_config = LoraConfig(lora_dropout=0.1, ...)

# Fix 3: Data augmentation
# Fix 4: Early stopping
```

### 4. Gradient Accumulation Mistakes

```python
# WRONG: Gradients not normalized
for i, batch in enumerate(loader):
    loss = model(batch).loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# CORRECT: Normalize by accumulation steps
for i, batch in enumerate(loader):
    loss = model(batch).loss / accumulation_steps  # Normalize!
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Not Handling Padding Correctly

```python
# WRONG: Loss includes padding tokens
loss = criterion(output, labels)

# CORRECT: Mask padding in loss
loss = criterion(output, labels)
mask = (labels != pad_token_id).float()
loss = (loss * mask).sum() / mask.sum()

# Or use ignore_index
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
```
