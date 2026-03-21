import torch
import sys

# Patch for torchao / unsloth compatibility
for i in range(1, 8):
    attr = f"int{i}"
    if not hasattr(torch, attr): setattr(torch, attr, torch.int8)

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Has int1: {hasattr(torch, 'int1')}")
print(f"Has int2: {hasattr(torch, 'int2')}")
print(f"Has int4: {hasattr(torch, 'int4')}")
if hasattr(torch, 'int1'):
    print(f"torch.int1: {torch.int1}")
if hasattr(torch, 'int2'):
    print(f"torch.int2: {torch.int2}")
if hasattr(torch, 'int4'):
    print(f"torch.int4: {torch.int4}")

try:
    import torchao
    print("Torchao imported successfully")
except Exception as e:
    print(f"Torchao import failed: {e}")
