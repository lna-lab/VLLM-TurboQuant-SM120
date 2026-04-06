"""Shared test fixtures for trinity-turbo."""

import pytest
import torch


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def head_dim():
    return 128


@pytest.fixture
def num_outliers():
    return 8


@pytest.fixture
def random_kv(device, head_dim):
    """Generate random KV vectors mimicking Trinity's output."""
    torch.manual_seed(42)
    batch_size = 32
    num_kv_heads = 2  # per-GPU shard (8 total / TP=4)
    return torch.randn(batch_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
