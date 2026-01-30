# Common ML Mistakes

## Data Leakage

### 1. Preprocessing Before Split

**Wrong:**
```python
# WRONG: Normalizing before split leaks test statistics
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # Fits on ALL data
X_train, X_test = train_test_split(X_normalized)
```

**Correct:**
```python
# CORRECT: Split first, then fit only on training
X_train, X_test = train_test_split(X)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit only on train
X_test = scaler.transform(X_test)        # Transform test with train stats
```

### 2. Feature Engineering Before Split

**Wrong:**
```python
# WRONG: Using target to create features before split
df['target_mean'] = df.groupby('category')['target'].transform('mean')
train, test = train_test_split(df)
```

**Correct:**
```python
# CORRECT: Compute on training only
train, test = train_test_split(df)
category_means = train.groupby('category')['target'].mean()
train['target_mean'] = train['category'].map(category_means)
test['target_mean'] = test['category'].map(category_means)  # May have NaN
```

### 3. Cross-Validation with Time Series

**Wrong:**
```python
# WRONG: Random split for time series
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)  # Randomly mixes future and past
```

**Correct:**
```python
# CORRECT: Time-based split
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)  # Always trains on past, validates on future
```

### 4. Data Augmentation Leak

**Wrong:**
```python
# WRONG: Augment before split (augmented versions of same image in train and test)
augmented_data = augment_dataset(data)
train, test = train_test_split(augmented_data)
```

**Correct:**
```python
# CORRECT: Split first, augment only training
train, test = train_test_split(data)
train = augment_dataset(train)  # Only augment training
```

## Reproducibility Failures

### 1. Missing Seeds

**Wrong:**
```python
# WRONG: No seeds set
model = train(data)  # Different results every run
```

**Correct:**
```python
# CORRECT: Set ALL relevant seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For complete determinism (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
model = train(data)
```

### 2. Non-Deterministic Operations

**Wrong:**
```python
# Operations with non-deterministic behavior by default
torch.nn.functional.interpolate(x, scale_factor=2)  # May be non-deterministic
torch.index_select(x, 0, indices)  # Non-deterministic backward
```

**Correct:**
```python
# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)

# Or handle specific operations
# Set environment variable for CUDA operations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### 3. Data Loader Randomness

**Wrong:**
```python
# WRONG: Random but not reproducible
loader = DataLoader(dataset, shuffle=True)
```

**Correct:**
```python
# CORRECT: Reproducible shuffling
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)
```

## Memory Issues

### 1. Keeping Computation Graph

**Wrong:**
```python
# WRONG: Accumulating history keeps entire graph
losses = []
for batch in loader:
    loss = model(batch).loss
    losses.append(loss)  # Keeps computation graph!
```

**Correct:**
```python
# CORRECT: Detach or use .item()
losses = []
for batch in loader:
    loss = model(batch).loss
    losses.append(loss.item())  # Just the number

# Or for tensor operations
losses.append(loss.detach())
```

### 2. Gradients During Inference

**Wrong:**
```python
# WRONG: Computing gradients during inference
def evaluate(model, loader):
    predictions = []
    for batch in loader:
        output = model(batch)  # Still computing gradients
        predictions.append(output)
```

**Correct:**
```python
# CORRECT: Disable gradients
def evaluate(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():  # No gradient computation
        for batch in loader:
            output = model(batch)
            predictions.append(output)
```

### 3. Not Clearing CUDA Cache

**Wrong:**
```python
# Running out of GPU memory between experiments
model1 = train_model_1()
model2 = train_model_2()  # OOM!
```

**Correct:**
```python
# Clear memory between experiments
model1 = train_model_1()
del model1
torch.cuda.empty_cache()
gc.collect()
model2 = train_model_2()
```

### 4. Large Validation Batches

**Wrong:**
```python
# WRONG: Same batch size for validation (may OOM)
train_loader = DataLoader(train_data, batch_size=8)
val_loader = DataLoader(val_data, batch_size=8)  # Could use larger
```

**Correct:**
```python
# CORRECT: Larger batches for validation (no gradients)
train_loader = DataLoader(train_data, batch_size=8)
val_loader = DataLoader(val_data, batch_size=32)  # 4x larger is often safe
```

## Training Issues

### 1. Forgetting model.train() / model.eval()

**Wrong:**
```python
# WRONG: Not setting mode
def train_epoch(model, loader):
    for batch in loader:
        # Dropout and BatchNorm may behave wrong
        loss = compute_loss(model(batch))

def validate(model, loader):
    for batch in loader:
        # Dropout still active, BatchNorm using batch stats
        output = model(batch)
```

**Correct:**
```python
def train_epoch(model, loader):
    model.train()  # Enable dropout, use batch stats for BN
    for batch in loader:
        loss = compute_loss(model(batch))

def validate(model, loader):
    model.eval()  # Disable dropout, use running stats for BN
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
```

### 2. Incorrect Loss Reduction

**Wrong:**
```python
# WRONG with gradient accumulation
criterion = nn.CrossEntropyLoss(reduction='mean')

for i, batch in enumerate(loader):
    loss = criterion(model(batch['x']), batch['y'])
    loss.backward()

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
# Problem: loss is already averaged per batch, but you want average over accum_steps
```

**Correct:**
```python
# CORRECT: Scale loss for gradient accumulation
criterion = nn.CrossEntropyLoss(reduction='mean')

for i, batch in enumerate(loader):
    loss = criterion(model(batch['x']), batch['y']) / accum_steps
    loss.backward()

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Learning Rate Not Matching Batch Size

**Wrong:**
```python
# WRONG: Changing batch size without adjusting LR
# Original: batch_size=32, lr=0.001
# New setup: batch_size=128, lr=0.001  # LR should be 4x
```

**Correct:**
```python
# CORRECT: Linear scaling rule
# When increasing batch size by k, increase LR by k
original_lr = 0.001
original_batch_size = 32
new_batch_size = 128

new_lr = original_lr * (new_batch_size / original_batch_size)
# new_lr = 0.004
```

### 4. Zero Gradients at Wrong Time

**Wrong:**
```python
# WRONG: zero_grad after backward (accumulates gradients)
for batch in loader:
    loss = model(batch).loss
    loss.backward()
    optimizer.zero_grad()  # Clears the gradients you just computed!
    optimizer.step()
```

**Correct:**
```python
# CORRECT: zero_grad before forward
for batch in loader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

## Silent Failures

### 1. NaN/Inf Loss Not Detected

**Wrong:**
```python
# WRONG: NaN goes undetected
for epoch in range(100):
    loss = train_epoch(model)
    print(f"Loss: {loss}")  # Might print "nan" and continue training garbage
```

**Correct:**
```python
# CORRECT: Check for invalid values
for epoch in range(100):
    loss = train_epoch(model)

    if not torch.isfinite(torch.tensor(loss)):
        raise ValueError(f"Loss became {loss} at epoch {epoch}")

    # Or enable anomaly detection during debugging
    # with torch.autograd.detect_anomaly():
    #     loss.backward()
```

### 2. Wrong Tensor Device

**Wrong:**
```python
# WRONG: Silently runs on CPU (slow)
model = Model()
data = data.cuda()
output = model(data)  # Model on CPU, data copied back silently
```

**Correct:**
```python
# CORRECT: Explicitly manage devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
data = data.to(device)
output = model(data)
```

### 3. Incorrect Broadcasting

**Wrong:**
```python
# WRONG: Shapes broadcast unexpectedly
labels = torch.tensor([0, 1, 2])  # Shape: (3,)
predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # Shape: (2, 2)

# This doesn't error but gives wrong results
loss = F.cross_entropy(predictions, labels)  # Shape mismatch!
```

**Correct:**
```python
# CORRECT: Verify shapes
assert predictions.shape[0] == labels.shape[0], \
    f"Batch size mismatch: {predictions.shape[0]} vs {labels.shape[0]}"

loss = F.cross_entropy(predictions, labels)
```

### 4. Frozen Parameters Silently Ignored

**Wrong:**
```python
# WRONG: Optimizer includes frozen params (no error, but wastes memory)
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = Adam(model.parameters(), lr=1e-3)  # Includes frozen params
```

**Correct:**
```python
# CORRECT: Only include trainable parameters
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Or verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {trainable}/{total} parameters ({100*trainable/total:.1f}%)")
```

## Overfitting Indicators

### Signs of Overfitting

1. **Train loss decreasing, val loss increasing**
2. **Train accuracy >> val accuracy** (e.g., 99% vs 70%)
3. **Model memorizes training examples** (perfect training performance)
4. **Performance degrades on new data**

### Fixes

```python
# 1. More data / Data augmentation
transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2),
])

# 2. Regularization
optimizer = AdamW(model.parameters(), weight_decay=0.01)

# 3. Dropout
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.Dropout(0.5),  # Add dropout
    nn.Linear(50, 10),
)

# 4. Early stopping
# (see finetuning.md)

# 5. Simpler model
# Reduce layers, hidden size, etc.

# 6. Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Gradient Issues

### Exploding Gradients

**Detection:**
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")  # If >> 1000, likely exploding
```

**Fix:**
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Vanishing Gradients

**Detection:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm < 1e-7:
            print(f"WARNING: Near-zero gradient for {name}: {grad_norm}")
```

**Fixes:**
```python
# 1. Use residual connections
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.block(x)  # Skip connection

# 2. Use proper initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# 3. Use batch/layer normalization
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.BatchNorm1d(50),  # Helps gradient flow
    nn.ReLU(),
)

# 4. Use LSTM/GRU instead of vanilla RNN
rnn = nn.LSTM(input_size, hidden_size)  # Has gates to control gradient flow
```
