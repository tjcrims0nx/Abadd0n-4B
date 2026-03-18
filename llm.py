"""
Abadd0n — Custom LLM Architecture
Mimicking Qwen3 / Llama-3 design patterns:
  - RMSNorm          (instead of LayerNorm)
  - RoPE             (Rotary Positional Embedding)
  - SwiGLU           (Gated feed-forward, Qwen3 style)
  - GQA              (Grouped Query Attention)
  - Pre-norm residual connections

Designed to train on 4 GB VRAM via QLoRA / Unsloth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ─────────────────────────────────────────────
# Hyperparameters  (small enough for 4 GB VRAM)
# ─────────────────────────────────────────────
n_embd      = 512        # Hidden dimension
n_head      = 8          # Query heads
n_kv_head   = 2          # Key/Value heads (GQA: n_head / n_kv_head = 4 groups)
n_layer     = 8          # Transformer depth
block_size  = 512        # Context window (token count)
dropout     = 0.0        # Disabled during inference; set 0.1 for scratch training
ffn_mult    = 8 / 3      # SwiGLU expansion factor (~2.67× for parameter parity)
vocab_size  = 32000      # Set at runtime by tokenizer; default matches BPE tokenizers

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _get_ffn_dim(n_embd: int, mult: float = ffn_mult) -> int:
    """Compute SwiGLU hidden dim, rounded to nearest multiple of 256."""
    raw = int(n_embd * mult)
    return (raw + 255) // 256 * 256


# ─────────────────────────────────────────────
# 1. RMSNorm  (Qwen3 / Llama-3 use this instead of LayerNorm)
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ─────────────────────────────────────────────
# 2. RoPE — Rotary Positional Embeddings
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """Pre-compute cos/sin tables for RoPE; applied per-head per query/key."""

    def __init__(self, dim: int, max_seq_len: int = block_size, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (T, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (T, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # (1,1,T,dim)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        self._cached_len = seq_len

    def forward(self, seq_len: int):
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ─────────────────────────────────────────────
# 3. Grouped Query Attention (GQA)
# ─────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    GQA: n_head query heads share n_kv_head key/value heads.
    For 4 GB VRAM: n_head=8, n_kv_head=2  →  4× KV cache reduction.
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int):
        super().__init__()
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.n_groups  = n_head // n_kv_head
        self.head_dim  = n_embd // n_head

        # Separate projections — key/value are smaller (n_kv_head × head_dim)
        self.q_proj = nn.Linear(n_embd, n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd,                    bias=False)

        self.rope  = RotaryEmbedding(self.head_dim)
        self.drop  = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        H, KV, D = self.n_head, self.n_kv_head, self.head_dim

        # Project & reshape  →  (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, H,  D).transpose(1, 2)   # (B, H,  T, D)
        k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)   # (B, KV, T, D)
        v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)   # (B, KV, T, D)

        # Apply RoPE to Q and K
        cos, sin = self.rope(T)
        cos = cos[:, :, :T, :D]
        sin = sin[:, :, :T, :D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match Q heads  (repeat_interleave for GQA)
        k = k.repeat_interleave(self.n_groups, dim=1)   # (B, H, T, D)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale           # (B, H, T, T)
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            else:
                causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
                att    = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            out = att @ v                                            # (B, H, T, D)

        # Re-shape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C)
        return self.o_proj(out)


# ─────────────────────────────────────────────
# 4. SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    """
    SwiGLU: output = SiLU(gate) ⊙ up  →  down
    Matches Qwen3 / Llama-3 FFN design.
    gate_proj and up_proj run in parallel; their product feeds down_proj.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        hidden = _get_ffn_dim(n_embd)
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─────────────────────────────────────────────
# 5. Transformer Block  (pre-norm, like Qwen3)
# ─────────────────────────────────────────────
class Abadd0nBlock(nn.Module):
    """Single transformer block with GQA + SwiGLU + RMSNorm (pre-norm)."""

    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = GroupedQueryAttention(n_embd, n_head, n_kv_head)
        self.norm2 = RMSNorm(n_embd)
        self.ffn   = SwiGLUFFN(n_embd)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), mask)   # Pre-norm residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 6. Full Abadd0n Model
# ─────────────────────────────────────────────
class Abadd0n(nn.Module):
    """
    Abadd0n — Qwen3-style decoder-only transformer.
    Architecture highlights:
      - GQA (8Q / 2KV heads)
      - SwiGLU FFN
      - RoPE positional embeddings
      - RMSNorm (pre-norm)
      - No positional embedding table (RoPE handles position)

    Designed for 4 GB VRAM QLoRA fine-tuning via Unsloth.
    """

    def __init__(self, vocab: int = vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, n_embd)
        self.embed_drop   = nn.Dropout(dropout)
        self.layers       = nn.ModuleList([Abadd0nBlock() for _ in range(n_layer)])
        self.norm         = RMSNorm(n_embd)
        self.lm_head      = nn.Linear(n_embd, vocab, bias=False)

        # Weight tying — ties embedding and lm_head weights (saves VRAM)
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= block_size, f"Sequence length {T} exceeds block_size {block_size}"

        x = self.embed_drop(self.embed_tokens(idx))   # (B, T, C)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)                       # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ── Generation ──────────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with nucleus (top-p) sampling,
        temperature scaling, and repetition penalty.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # Last token logits

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

        return idx

    def param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{total / 1e6:.1f}M parameters"

    def generate_text(self, text: str, max_new_chars: int = 200) -> str:
        """Generate text from input string (character-level)."""
        idx = torch.tensor(
            [[STOI.get(c, STOI.get(" ", 0)) for c in text]],
            dtype=torch.long
        )
        idx = idx.to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        with torch.no_grad():
            out_ids = self.generate(idx, max_new_tokens=max_new_chars)
        return "".join([ITOS.get(i, "?") for i in out_ids[0].tolist()])


# ─────────────────────────────────────────────
# Legacy character-level alias (kept for train.py compatibility)
# ─────────────────────────────────────────────
CHARS      = sorted(list(set(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?'\"-()[]{}<>:;/@#$%^&*+_=\n\\|~`\r")))
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}

# LocalLLM is retained for backwards compatibility with train.py / main.py
LocalLLM = Abadd0n
