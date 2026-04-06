"""Tests for bit-packing utilities."""

import torch
import pytest

from trinity_turbo.quant.packing import pack_indices, unpack_indices, packed_size


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_roundtrip(bits):
    """Pack then unpack should recover original indices."""
    num_elements = 120  # Trinity's normal_dim (128 - 8 outliers)
    max_val = (1 << bits) - 1
    torch.manual_seed(42)
    indices = torch.randint(0, max_val + 1, (4, num_elements), dtype=torch.uint8)

    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, num_elements)

    torch.testing.assert_close(unpacked, indices)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_packed_size(bits):
    num_elements = 120
    expected = (num_elements * bits + 7) // 8
    assert packed_size(num_elements, bits) == expected


@pytest.mark.parametrize("bits,num_elements", [
    (4, 1), (4, 2), (4, 127),
    (3, 1), (3, 7), (3, 8), (3, 9), (3, 120),
    (2, 1), (2, 3), (2, 4), (2, 5), (2, 120),
])
def test_roundtrip_edge_cases(bits, num_elements):
    """Test various element counts including non-aligned."""
    max_val = (1 << bits) - 1
    indices = torch.randint(0, max_val + 1, (num_elements,), dtype=torch.uint8)

    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, num_elements)

    torch.testing.assert_close(unpacked, indices)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_batch_dims(bits):
    """Packing should work with arbitrary batch dimensions."""
    num_elements = 120
    max_val = (1 << bits) - 1
    indices = torch.randint(0, max_val + 1, (2, 8, num_elements), dtype=torch.uint8)

    packed = pack_indices(indices, bits)
    unpacked = unpack_indices(packed, bits, num_elements)

    torch.testing.assert_close(unpacked, indices)
