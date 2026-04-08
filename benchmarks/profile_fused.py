"""Compare old vs fused compress+rotation overhead."""
import math
import torch
from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_fused_compress import fused_compress_to_slot
from trinity_turbo.kernels.triton_fused_rotation import (
    triton_apply_rotation, triton_apply_inverse_rotation,
)
from trinity_turbo.quant.rotation import apply_rotation, apply_inverse_rotation
from trinity_turbo.quant.turboquant import QuantState

DEVICE = "cuda"
ITERS = 100
WARMUP = 10

state = QuantState.create(bits=4, head_dim=128, num_outliers=8, device=DEVICE)

# Test data (decode: 1 token × 2 kv heads)
key = torch.randn(1, 2, 128, device=DEVICE, dtype=torch.bfloat16)
q_normal = torch.randn(1, 16, 120, device=DEVICE, dtype=torch.float32)

def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    print(f"  {label}: {ms:.4f} ms")
    return ms

print("=== Compress: old vs fused ===")
t_old = bench(lambda: compress_to_slot(key, state), "PyTorch compress_to_slot")
t_new = bench(lambda: fused_compress_to_slot(key, state), "Triton fused_compress")
print(f"  Speedup: {t_old/t_new:.1f}×")

# Verify correctness
old_slot = compress_to_slot(key, state)
new_slot = fused_compress_to_slot(key, state)
match = (old_slot == new_slot).float().mean().item()
print(f"  Byte match: {match*100:.1f}%")

print("\n=== Rotation: old vs fused ===")
t_rot_old = bench(lambda: apply_rotation(q_normal, state.sign_flips), "PyTorch apply_rotation")
t_rot_new = bench(lambda: triton_apply_rotation(q_normal, state.sign_flips), "Triton fused_rotation")
print(f"  Speedup: {t_rot_old/t_rot_new:.1f}×")

# Verify correctness
ref_rot = apply_rotation(q_normal, state.sign_flips)
new_rot = triton_apply_rotation(q_normal, state.sign_flips)
cos_sim = torch.nn.functional.cosine_similarity(
    ref_rot.reshape(-1), new_rot.reshape(-1), dim=0
).item()
print(f"  cos_sim: {cos_sim:.6f}")

print("\n=== Inverse Rotation: old vs fused ===")
t_inv_old = bench(lambda: apply_inverse_rotation(q_normal, state.sign_flips), "PyTorch inv_rotation")
t_inv_new = bench(lambda: triton_apply_inverse_rotation(q_normal, state.sign_flips), "Triton fused_inv_rot")
print(f"  Speedup: {t_inv_old/t_inv_new:.1f}×")

ref_inv = apply_inverse_rotation(q_normal, state.sign_flips)
new_inv = triton_apply_inverse_rotation(q_normal, state.sign_flips)
cos_inv = torch.nn.functional.cosine_similarity(
    ref_inv.reshape(-1), new_inv.reshape(-1), dim=0
).item()
print(f"  cos_sim: {cos_inv:.6f}")

# Total per-step saving
print(f"\n=== Per decode step (60 layers) ===")
old_total = (t_old + t_rot_old + t_inv_old) * 60
new_total = (t_new + t_rot_new + t_inv_new) * 60
print(f"  Old: {old_total:.1f} ms")
print(f"  New: {new_total:.1f} ms")
print(f"  Saved: {old_total - new_total:.1f} ms ({(1-new_total/old_total)*100:.0f}%)")
