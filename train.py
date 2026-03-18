"""
Abadd0n — Character-Level Pre-Training Script
═════════════════════════════════════════════
Trains the custom Abadd0n architecture (GQA + SwiGLU + RMSNorm + RoPE)
from scratch on raw text data (data.txt).

This is the *foundational pre-training* step — run this BEFORE QLoRA fine-tuning
if you want to build a model from scratch rather than fine-tuning a pretrained one.

Usage:
    python train.py [--epochs 1000] [--lr 1e-3] [--data data.txt]
"""

import argparse
import time
import torch
import os
from llm import Abadd0n, STOI, ITOS, VOCAB_SIZE, block_size

WEIGHTS_PATH = "model_weights.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Data loading & encoding
# ─────────────────────────────────────────────
def load_text(path: str) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Corpus: {len(text):,} characters, vocab={VOCAB_SIZE}")
    data = torch.tensor([STOI.get(c, STOI[" "]) for c in text], dtype=torch.long)
    return data


def get_batch(data: torch.Tensor, batch_size: int):
    if len(data) <= block_size:
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)
        return x.to(DEVICE), y.to(DEVICE)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i : i + block_size]     for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(model: Abadd0n, data_path: str, epochs: int, lr: float, batch_size: int):
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return

    data  = load_text(data_path)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n  Device   : {DEVICE}")
    print(f"  Params   : {model.param_count()}")
    print(f"  Train    : {len(train_data):,} tokens")
    print(f"  Val      : {len(val_data):,} tokens")
    print(f"  Epochs   : {epochs}   LR: {lr}   Batch: {batch_size}")
    print()

    model.to(DEVICE).train()
    t0 = time.time()

    for step in range(epochs):
        xb, yb     = get_batch(train_data, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == epochs - 1:
            # Validation loss
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, batch_size)
                _, val_loss = model(xv, yv)
            model.train()
            elapsed = time.time() - t0
            print(f"  step {step:>5} | train_loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | {elapsed:.1f}s")

    # Save weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\n  ✓ Weights saved to {WEIGHTS_PATH}")
    print(f"  Total time: {(time.time()-t0)/60:.2f} min")


# ─────────────────────────────────────────────
# Quick inference test
# ─────────────────────────────────────────────
def run_inference(model: Abadd0n, prompt: str = "Abadd0n", max_tokens: int = 100):
    model.eval().to(DEVICE)
    idx = torch.tensor(
        [[STOI.get(c, STOI[" "]) for c in prompt]],
        dtype=torch.long, device=DEVICE
    )
    with torch.no_grad():
        out_ids = model.generate(
            idx,
            max_new_tokens    = max_tokens,
            temperature       = 0.8,
            top_k             = 40,
            top_p             = 0.9,
            repetition_penalty= 1.1,
        )
    generated = "".join([ITOS.get(i, "?") for i in out_ids[0].tolist()])
    return generated


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abadd0n Character-Level Pre-Training")
    parser.add_argument("--data",   type=str,   default="data.txt",  help="Path to training text")
    parser.add_argument("--epochs", type=int,   default=1000,        help="Training steps")
    parser.add_argument("--lr",     type=float, default=1e-3,        help="Learning rate")
    parser.add_argument("--batch",  type=int,   default=16,          help="Batch size")
    parser.add_argument("--resume", action="store_true",             help="Resume from saved weights")
    args = parser.parse_args()

    print("=" * 55)
    print("  Abadd0n — Character-Level Pre-Training")
    print("=" * 55)

    model = Abadd0n(vocab=VOCAB_SIZE)

    if args.resume and os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        print(f"  Resumed from {WEIGHTS_PATH}")

    train(model, args.data, args.epochs, args.lr, args.batch)

    # Quick test after training
    sample = run_inference(model, prompt="Hello")
    print(f"\n  Sample output:\n  {repr(sample)}")
