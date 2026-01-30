# GPU Management and Optimization

## Memory Estimation

### Quick Estimation Formula

For training with Adam optimizer:
```
Memory (GB) ≈ (model_params × 16) / 1e9 + activation_memory

Where:
- 16 bytes per parameter = 4 (param) + 4 (grad) + 8 (Adam states: m + v)
- activation_memory depends on batch size and model architecture
```

### Detailed Estimation

```python
def estimate_training_memory(
    model,
    batch_size: int,
    seq_length: int = 512,
    precision: str = "fp32",  # fp32, fp16, bf16
    optimizer: str = "adam",  # adam, sgd, adamw
) -> dict:
    """
    Estimate GPU memory for training.

    Returns dict with breakdown of memory usage.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Bytes per element based on precision
    bytes_per_param = 4 if precision == "fp32" else 2

    # Model parameters
    param_memory = trainable_params * bytes_per_param

    # Gradients (same size as params)
    grad_memory = trainable_params * bytes_per_param

    # Optimizer states
    if optimizer in ["adam", "adamw"]:
        # Adam stores m (momentum) and v (variance) in fp32
        optimizer_memory = trainable_params * 4 * 2  # 2 states, always fp32
    elif optimizer == "sgd":
        optimizer_memory = trainable_params * 4  # Only momentum if enabled
    else:
        optimizer_memory = 0

    # Activation memory (rough estimate)
    # This varies greatly by architecture
    # For transformers: roughly batch_size * seq_len * hidden_dim * num_layers * 2
    hidden_dim = 768  # Default, adjust based on model
    num_layers = 12   # Default, adjust based on model

    if hasattr(model, 'config'):
        hidden_dim = getattr(model.config, 'hidden_size', hidden_dim)
        num_layers = getattr(model.config, 'num_hidden_layers', num_layers)

    activation_memory = batch_size * seq_length * hidden_dim * num_layers * bytes_per_param * 2

    total = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        "total_gb": total / 1e9,
        "params_gb": param_memory / 1e9,
        "grads_gb": grad_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
```

### Model Size Reference

| Model Size | Params | FP32 Size | FP16 Size | 4-bit Size |
|------------|--------|-----------|-----------|------------|
| Small | 125M | 500 MB | 250 MB | 63 MB |
| Base | 350M | 1.4 GB | 700 MB | 175 MB |
| Large | 774M | 3.1 GB | 1.6 GB | 400 MB |
| 1.3B | 1.3B | 5.2 GB | 2.6 GB | 650 MB |
| 7B | 7B | 28 GB | 14 GB | 3.5 GB |
| 13B | 13B | 52 GB | 26 GB | 6.5 GB |
| 70B | 70B | 280 GB | 140 GB | 35 GB |

**Note**: Training requires ~4x model size for FP32 (params + grads + optimizer), ~3x for mixed precision.

## Batch Size Tuning

### Binary Search for Max Batch Size

```python
def find_max_batch_size(
    model,
    create_batch_fn,
    device,
    min_batch=1,
    max_batch=512,
):
    """
    Binary search for maximum batch size that fits in GPU memory.

    Args:
        model: PyTorch model
        create_batch_fn: Function that takes batch_size and returns a batch
        device: torch.device
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try

    Returns:
        Maximum viable batch size
    """
    model.train()

    def try_batch_size(batch_size):
        try:
            torch.cuda.empty_cache()

            batch = create_batch_fn(batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            output = model(**batch)
            loss = output.loss if hasattr(output, 'loss') else output.mean()

            # Backward
            loss.backward()

            # Clear
            model.zero_grad()
            torch.cuda.empty_cache()

            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise

    # Binary search
    low, high = min_batch, max_batch
    max_working = min_batch

    while low <= high:
        mid = (low + high) // 2

        if try_batch_size(mid):
            max_working = mid
            low = mid + 1
            print(f"  Batch size {mid}: OK")
        else:
            high = mid - 1
            print(f"  Batch size {mid}: OOM")

    # Use 90% of max for safety margin
    recommended = int(max_working * 0.9)
    print(f"\nMax batch size: {max_working}")
    print(f"Recommended (with margin): {recommended}")

    return recommended
```

### Gradient Accumulation

When batch size is limited by memory:

```python
def train_with_accumulation(
    model,
    loader,
    optimizer,
    accumulation_steps=4,
    max_grad_norm=1.0,
):
    """
    Train with gradient accumulation.

    Effective batch size = batch_size * accumulation_steps
    """
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale loss

        # Backward pass
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

## Mixed Precision Training

### PyTorch Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, loader, optimizer, criterion, device):
    """
    Training loop with automatic mixed precision.
    """
    scaler = GradScaler()
    model.train()

    for batch in loader:
        x, y = batch['x'].to(device), batch['y'].to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            output = model(x)
            loss = criterion(output, y)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        scaler.step(optimizer)
        scaler.update()

    return loss.item()
```

### BFloat16 (for Ampere+ GPUs)

```python
# BF16 has same range as FP32, better for training stability
torch.set_default_dtype(torch.bfloat16)

# Or use in autocast
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```

## Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass.

```python
# For custom models
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # Recompute activations during backward
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# For Hugging Face transformers
model.gradient_checkpointing_enable()
```

**Memory savings**: ~50-70% reduction in activation memory
**Compute cost**: ~30% slower training

## Multi-GPU Training

### DataParallel (Simple, Single-Node)

```python
# Simple but not recommended for serious training
model = nn.DataParallel(model)
output = model(input)  # Automatically splits batch across GPUs
```

**Issues**: GPU 0 bottleneck, high memory on GPU 0, slower than DDP.

### DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size, model, dataset):
    setup(rank, world_size)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        for batch in loader:
            # Training code
            pass

    dist.destroy_process_group()

# Launch with:
# torchrun --nproc_per_node=4 train.py
```

### FSDP (For Very Large Models)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Wrap model with FSDP
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
)
```

## Monitoring

### GPU Monitoring Commands

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU memory by process
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Continuous logging
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 1 > gpu_log.csv
```

### PyTorch Memory Tracking

```python
def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"GPU Memory: {allocated:.2f}GB allocated, "
              f"{reserved:.2f}GB reserved, "
              f"{max_allocated:.2f}GB max allocated")

# Memory snapshot for debugging
def get_memory_snapshot():
    """Get detailed memory snapshot."""
    snapshot = torch.cuda.memory_snapshot()
    return snapshot

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

### Memory Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    with record_function("forward"):
        output = model(input)
    with record_function("backward"):
        output.loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

## DeepSpeed Integration

For very large models and efficient distributed training.

```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="deepspeed_config.json",
)

for batch in loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### ZeRO Stages

| Stage | What's Sharded | Memory Reduction |
|-------|---------------|------------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~N (linear with GPUs) |

## Quick Reference

### Memory Reduction Techniques (by Impact)

| Technique | Memory Reduction | Speed Impact |
|-----------|-----------------|--------------|
| FP16/BF16 | ~50% | 0-20% faster |
| Gradient checkpointing | ~50-70% | 20-30% slower |
| Smaller batch + accumulation | Variable | ~Same |
| ZeRO-2 | ~8x (distributed) | Slight overhead |
| 8-bit optimizers | ~50% optimizer mem | Minimal |
| Offloading (CPU/NVMe) | Large | Much slower |

### Common GPU Memory Sizes

| GPU | Memory | Good For |
|-----|--------|----------|
| RTX 3080 | 10 GB | 7B inference (4-bit), small training |
| RTX 3090 | 24 GB | 7B fine-tuning, 13B inference |
| RTX 4090 | 24 GB | Same as 3090, faster |
| A100 40GB | 40 GB | 13B fine-tuning, 70B inference |
| A100 80GB | 80 GB | 70B fine-tuning |
| H100 80GB | 80 GB | Same as A100, much faster |
