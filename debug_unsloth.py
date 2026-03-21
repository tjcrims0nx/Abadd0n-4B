import traceback
import sys

print("Python version:", sys.version)
try:
    import torch
    # Patch for torchao / unsloth compatibility
    for i in range(1, 8):
        attr = f"int{i}"
        if not hasattr(torch, attr): setattr(torch, attr, torch.int8)
    import torch._inductor.config
    print("Torch version:", torch.__version__)
    print("Torch Inductor Config loaded")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception:
    print("Failed to import torch")
    traceback.print_exc()

print("\n--- Importing Unsloth ---")
try:
    import pre_unsloth

    pre_unsloth.before_import()
    import unsloth
    print("Unsloth imported successfully")
except Exception:
    traceback.print_exc()
