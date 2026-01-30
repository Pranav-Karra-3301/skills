# ML Testing Guide

## Overview

Testing ML code is crucial for catching bugs that may silently degrade model performance. This guide covers pytest setup, ML-specific tests, and property-based testing.

## Pytest Setup

### conftest.py

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
import random

@pytest.fixture(scope="session")
def device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
def set_seed():
    """Set seeds for reproducibility in tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing."""
    batch_size = 4
    seq_length = 32
    vocab_size = 1000

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)).to(device),
        "attention_mask": torch.ones(batch_size, seq_length).to(device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_length)).to(device),
    }

@pytest.fixture
def sample_images(device):
    """Create sample images for vision tests."""
    batch_size = 4
    channels = 3
    height, width = 224, 224

    return torch.randn(batch_size, channels, height, width).to(device)

@pytest.fixture
def sample_tabular():
    """Create sample tabular data."""
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short -x
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    integration: marks integration tests
```

## Model Tests

### Test Output Shapes

```python
# tests/test_model.py
import pytest
import torch
from src.models import MyModel

class TestModel:
    @pytest.fixture
    def model(self, device):
        return MyModel(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
        ).to(device)

    def test_output_shape(self, model, sample_batch, device):
        """Test that model output has expected shape."""
        output = model(**sample_batch)

        batch_size = sample_batch["input_ids"].shape[0]
        seq_length = sample_batch["input_ids"].shape[1]

        assert output.logits.shape == (batch_size, seq_length, 1000)

    def test_output_shape_different_batch_sizes(self, model, device):
        """Test with various batch sizes."""
        for batch_size in [1, 4, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
            attention_mask = torch.ones_like(input_ids)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            assert output.logits.shape[0] == batch_size

    def test_output_shape_different_seq_lengths(self, model, device):
        """Test with various sequence lengths."""
        for seq_length in [16, 64, 128]:
            input_ids = torch.randint(0, 1000, (4, seq_length)).to(device)
            attention_mask = torch.ones_like(input_ids)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            assert output.logits.shape[1] == seq_length
```

### Test Gradient Flow

```python
def test_gradient_flow(self, model, sample_batch, device):
    """Test that gradients flow to all trainable parameters."""
    model.train()

    output = model(**sample_batch)
    loss = output.loss if hasattr(output, 'loss') else output.logits.mean()
    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

def test_no_gradient_for_frozen_params(self, model, sample_batch, device):
    """Test that frozen parameters don't receive gradients."""
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    output = model(**sample_batch)
    loss = output.loss if hasattr(output, 'loss') else output.logits.mean()
    loss.backward()

    for name, param in model.encoder.named_parameters():
        assert param.grad is None or (param.grad == 0).all(), \
            f"Frozen param {name} received gradients"
```

### Test Determinism

```python
def test_deterministic_forward(self, model, sample_batch, device):
    """Test that forward pass is deterministic."""
    model.eval()

    with torch.no_grad():
        output1 = model(**sample_batch).logits
        output2 = model(**sample_batch).logits

    assert torch.allclose(output1, output2), "Forward pass is not deterministic"

def test_deterministic_training_step(self, device):
    """Test that training step is deterministic with same seed."""
    def train_step(seed):
        torch.manual_seed(seed)
        model = MyModel(vocab_size=1000, hidden_size=256, num_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        input_ids = torch.randint(0, 1000, (4, 32)).to(device)
        labels = torch.randint(0, 1000, (4, 32)).to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()

        return output.loss.item(), list(model.parameters())[0].sum().item()

    loss1, param1 = train_step(42)
    loss2, param2 = train_step(42)

    assert loss1 == loss2, "Loss not deterministic"
    assert param1 == param2, "Parameters not deterministic after step"
```

### Test Save/Load

```python
def test_save_load_checkpoint(self, model, device, tmp_path):
    """Test that model can be saved and loaded correctly."""
    # Save
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Load into new model
    loaded_model = MyModel(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
    ).to(device)
    loaded_model.load_state_dict(torch.load(checkpoint_path))

    # Compare parameters
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(),
        loaded_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2), f"Mismatch in {name1}"

def test_save_load_produces_same_output(self, model, sample_batch, device, tmp_path):
    """Test that loaded model produces same output."""
    model.eval()

    # Original output
    with torch.no_grad():
        original_output = model(**sample_batch).logits

    # Save and load
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded_model = MyModel(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
    ).to(device)
    loaded_model.load_state_dict(torch.load(checkpoint_path))
    loaded_model.eval()

    # Loaded output
    with torch.no_grad():
        loaded_output = loaded_model(**sample_batch).logits

    assert torch.allclose(original_output, loaded_output)
```

## Data Pipeline Tests

### Test No Data Leakage

```python
# tests/test_data.py
import pytest
from src.data import create_data_splits

class TestDataPipeline:
    def test_no_overlap_between_splits(self, dataset):
        """Test that train/val/test splits don't overlap."""
        train, val, test = create_data_splits(dataset)

        train_ids = set(train.ids)
        val_ids = set(val.ids)
        test_ids = set(test.ids)

        assert len(train_ids & val_ids) == 0, "Train/val overlap!"
        assert len(train_ids & test_ids) == 0, "Train/test overlap!"
        assert len(val_ids & test_ids) == 0, "Val/test overlap!"

    def test_splits_cover_all_data(self, dataset):
        """Test that all data is in some split."""
        train, val, test = create_data_splits(dataset)

        total = len(train) + len(val) + len(test)
        assert total == len(dataset), "Some data missing from splits"
```

### Test Data Loader

```python
def test_dataloader_batch_size(self, train_loader):
    """Test that data loader produces correct batch sizes."""
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[0] == train_loader.batch_size

def test_dataloader_reproducibility(self, dataset):
    """Test that data loader is reproducible with same seed."""
    def get_first_batch(seed):
        torch.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        return next(iter(loader))

    batch1 = get_first_batch(42)
    batch2 = get_first_batch(42)

    assert torch.equal(batch1["input_ids"], batch2["input_ids"])

def test_dataloader_no_nans(self, train_loader):
    """Test that data loader doesn't produce NaN values."""
    for batch in train_loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                assert not torch.isnan(value).any(), f"NaN in {key}"
        break  # Just check first batch
```

### Test Preprocessing

```python
def test_preprocessing_reversible(self, preprocessor):
    """Test that preprocessing can be reversed (if applicable)."""
    original = torch.randn(10, 5)
    processed = preprocessor.transform(original)
    recovered = preprocessor.inverse_transform(processed)

    assert torch.allclose(original, recovered, atol=1e-5)

def test_preprocessing_fitted_on_train_only(self, train_data, test_data):
    """Test that preprocessing stats come from train only."""
    preprocessor = StandardScaler()
    preprocessor.fit(train_data)

    train_mean = train_data.mean(axis=0)
    test_mean = test_data.mean(axis=0)

    # Preprocessor should use train mean, not combined mean
    assert np.allclose(preprocessor.mean_, train_mean)
    assert not np.allclose(preprocessor.mean_, (train_mean + test_mean) / 2)
```

## Property-Based Testing with Hypothesis

```python
# tests/test_properties.py
from hypothesis import given, strategies as st, settings
import torch

class TestModelProperties:
    @given(batch_size=st.integers(min_value=1, max_value=32))
    @settings(max_examples=10)
    def test_batch_size_invariance(self, model, batch_size, device):
        """Test that model handles any batch size."""
        input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
        attention_mask = torch.ones_like(input_ids)

        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.logits.shape[0] == batch_size
        assert not torch.isnan(output.logits).any()

    @given(seq_length=st.integers(min_value=1, max_value=512))
    @settings(max_examples=10)
    def test_sequence_length_invariance(self, model, seq_length, device):
        """Test that model handles any sequence length."""
        input_ids = torch.randint(0, 1000, (4, seq_length)).to(device)
        attention_mask = torch.ones_like(input_ids)

        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.logits.shape[1] == seq_length

    @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=10, max_size=100))
    def test_loss_always_positive(self, values):
        """Test that loss is always non-negative."""
        predictions = torch.tensor(values).unsqueeze(0)
        targets = torch.zeros_like(predictions)

        loss = F.mse_loss(predictions, targets)

        assert loss >= 0
```

## Integration Tests

```python
# tests/test_integration.py
import pytest

@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipeline:
    def test_full_training_loop(self, tmp_path):
        """Test complete training pipeline."""
        # Setup
        model = create_model()
        train_loader = create_train_loader()
        val_loader = create_val_loader()
        optimizer = torch.optim.Adam(model.parameters())

        # Train for a few steps
        initial_loss = None
        for epoch in range(2):
            for batch in train_loader:
                loss = train_step(model, batch, optimizer)
                if initial_loss is None:
                    initial_loss = loss
                break  # Just one batch per epoch

        # Verify loss decreased
        final_loss = loss
        assert final_loss < initial_loss, "Loss didn't decrease"

        # Verify checkpoint saves
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(model, optimizer, checkpoint_path)
        assert checkpoint_path.exists()

        # Verify can resume
        model2 = create_model()
        optimizer2 = torch.optim.Adam(model2.parameters())
        load_checkpoint(model2, optimizer2, checkpoint_path)

    def test_overfitting_small_dataset(self, device):
        """Test that model can overfit a tiny dataset (sanity check)."""
        # Create tiny dataset
        X = torch.randn(10, 5).to(device)
        y = torch.randint(0, 2, (10,)).to(device)

        model = SimpleModel(input_dim=5, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Train many epochs
        for _ in range(100):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Should achieve very low loss on this tiny dataset
        assert loss.item() < 0.1, "Model can't overfit tiny dataset"

        # Should achieve high accuracy
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        assert accuracy > 0.9, "Model can't overfit tiny dataset"
```

## Markers for Selective Testing

```python
@pytest.mark.gpu
def test_gpu_functionality(self, model, device):
    """Test that requires GPU."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    # GPU-specific tests

@pytest.mark.slow
def test_long_running_training(self):
    """Test that takes a long time."""
    # Full training run

@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_different_precisions(self, model, precision, device):
    """Test model with different precisions."""
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported")
    # Test with precision
```

### Running Tests

```bash
# Run all tests
pytest

# Run without slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestModel::test_output_shape
```
