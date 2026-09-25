"""MiniMax H3 text encoder: Qwen3-VL-32B (50-layer) NVFP4 AWQ + vision tower.

Turns a prompt (and optional memory / keyframe images) into the [L, 5120] f32
embedding stream the DiT consumes, matching Comfy PR #15224 / MiniMaxH3Tokenizer:

- Checkpoint: text tower + ``visual.*`` in one safetensors
  (``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors``). Text linears are NVFP4
  (must ``from_blocked`` swizzled ``weight_scale``); vision is bf16 full.
- NOT chat-templated. Pure text = raw token ids. With images (Multishot /
  FL2VA presentation)::

      "<Picture 1>: " <|vision_start|> <image tokens> <|vision_end|> …
      <prompt>

- Conditioning = unnormalized hidden state after layer 50 (no final norm).
- Vision: Qwen3-VL-32B tower (27 blocks, deepstack at [8,16,24], spatial merge
  2, patch 16). Deepstack features inject into LM layers 0..2 at visual token
  positions. Interleaved MRoPE when images are present (rope_dims [24,20,20]).
- Optional ``minimax_token_tags`` [L] i64: 0 = vision block (incl. start/end),
  1 = text — DiT modality tags if a consumer wants them.

Streaming: vision runs first (bf16, ~1.2 GB weights), then text layers one at a
time so the 15.7 GB file fits a 16 GB card.

Usage::

    python scripts/h3_text_encode.py "prompt" out.safetensors
    python scripts/h3_text_encode.py "prompt" out.safetensors --image a.png --image b.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent


def _checkpoints_root() -> Path:
    """External weights tree (Gemmy) or ROOT/checkpoints (standalone H3 repo)."""
    import os
    env = (os.environ.get("GEMMY_H3_CHECKPOINTS") or "").strip()
    if env:
        return Path(env)
    return ROOT / "checkpoints"


CKPT = (
    _checkpoints_root()
    / "text_encoders"
    / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
)
TOK_DIR = _checkpoints_root() / "text_encoders" / "qwen25_tokenizer"

CFG = dict(
    hidden=5120,
    intermediate=25600,
    num_layers=50,
    num_heads=64,
    num_kv_heads=8,
    head_dim=128,
    vocab=151936,
    rope_theta=5_000_000.0,
    rms_eps=1e-6,
    # Qwen3-VL-32B MRoPE (inherited from Qwen3VL_8BConfig)
    rope_dims=(24, 20, 20),
    interleaved_mrope=True,
)

VISION = dict(
    hidden_size=1152,
    intermediate_size=4304,
    depth=27,
    num_heads=16,
    patch_size=16,
    temporal_patch_size=2,
    in_channels=3,
    spatial_merge_size=2,
    num_position_embeddings=2304,
    out_hidden_size=5120,
    deepstack_visual_indexes=(8, 16, 24),
)

VISION_START = 151652
VISION_END = 151653
PAD_ID = 151643
QWEN_IMAGE_MEAN = (0.5, 0.5, 0.5)
QWEN_IMAGE_STD = (0.5, 0.5, 0.5)

# FP4 E2M1: 1 sign, 2 exp (bias 1), 1 mantissa. Matches comfy_kitchen eager dequant.
_FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_table() -> torch.Tensor:
    t = torch.zeros(16, dtype=torch.float32)
    t[:8] = _FP4_MAG
    t[8:] = -_FP4_MAG
    return t


def unpack_fp4_weight(u8: torch.Tensor) -> torch.Tensor:
    """[out, in/2] U8 packed FP4 -> [out, in] nibble values (hi nibble = even index)."""
    u8 = u8.to(torch.uint8)
    lo = u8 & 0x0F
    hi = (u8 >> 4) & 0x0F
    return torch.stack([hi, lo], dim=-1).reshape(u8.shape[0], -1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Unswizzle cuBLAS D-block scaling layout back to plain [num_rows, num_cols]."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if blocked_matrix.dim() == 1:
        blocked_matrix = blocked_matrix.view(padded_rows, padded_cols)
    elif blocked_matrix.shape != (padded_rows, padded_cols):
        if blocked_matrix.shape[0] <= padded_rows and blocked_matrix.shape[1] <= padded_cols:
            padded = blocked_matrix.new_zeros((padded_rows, padded_cols))
            padded[: blocked_matrix.shape[0], : blocked_matrix.shape[1]] = blocked_matrix
            blocked_matrix = padded
        else:
            raise ValueError(
                f"blocked scale shape {tuple(blocked_matrix.shape)} incompatible with "
                f"target ({num_rows}, {num_cols}) / padded ({padded_rows}, {padded_cols})"
            )

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def dequant_nvfp4(u8: torch.Tensor, block_scale: torch.Tensor, tensor_scale: torch.Tensor) -> torch.Tensor:
    """Full-precision reconstruction of an NVFP4 weight (Comfy full_precision path)."""
    device = u8.device
    nib = unpack_fp4_weight(u8)
    table = fp4_table().to(device=device, dtype=torch.float32)
    w = table[nib.to(torch.long)]
    out_f, in_f = w.shape
    blocks_per_row = in_f // 16
    bs = from_blocked(block_scale.to(device), num_rows=out_f, num_cols=blocks_per_row).to(torch.float32)
    ts = tensor_scale.to(device=device, dtype=torch.float32).reshape(-1)[0]
    w = w.view(out_f, blocks_per_row, 16) * bs.unsqueeze(-1) * ts
    return w.view(out_f, in_f)


def load_tensor(f: safe_open, name: str) -> torch.Tensor:
    return f.get_tensor(name)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_freqs(seq_len: int, device, theta: float, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """1D RoPE for pure-text (no images)."""
    n = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (n / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def mrope_freqs(
    position_ids: torch.Tensor,
    device,
    theta: float,
    head_dim: int,
    rope_dims: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL interleaved MRoPE.

    ``position_ids`` is [3, L] (T/H/W). Returns cos/sin shaped [L, head_dim]
    matching the pure-text ``apply_rope`` convention.
    """
    # freqs per axis: [3, L, head_dim/2]
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))  # [head_dim/2]
    inv = inv_freq[None, :, None].expand(3, -1, 1)  # [3, D/2, 1]
    pos = position_ids.to(device=device, dtype=torch.float32)[:, None, :]  # [3, 1, L]
    freqs = (inv.float() @ pos.float()).transpose(1, 2)  # [3, L, D/2]

    # Interleaved: start from T, then H/W replace every 3rd dim in their sections.
    freqs_inter = freqs[0].clone()
    for axis_idx, offset in ((1, 1), (2, 2)):
        length = int(rope_dims[axis_idx]) * 3
        idx = slice(offset, length, 3)
        freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
    emb = torch.cat((freqs_inter, freqs_inter), dim=-1)  # [L, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Comfy apply_rope: cos full [L, dim], sin split into halves.

    x = x*cos; first half += x[half:]*nsin; second half += x[:half]*sin_half.
    """
    split = xq.shape[-1] // 2
    cos = cos.to(xq.dtype).unsqueeze(1)
    sin = sin.to(xq.dtype).unsqueeze(1)
    sin_half = sin[..., :split]
    nsin = -sin[..., split:]
    q = xq * cos
    q[..., :split].addcmul_(xq[..., split:], nsin)
    q[..., split:].addcmul_(xq[..., :split], sin_half)
    k = xk * cos
    k[..., :split].addcmul_(xk[..., split:], nsin)
    k[..., split:].addcmul_(xk[..., :split], sin_half)
    return q, k


def linear_dequant(x: torch.Tensor, f: safe_open, prefix: str, compute: torch.dtype, device) -> torch.Tensor:
    """Dequantize one nvfp4 linear and apply it (input AWQ pre_quant_scale first)."""
    pqs = None
    if f"{prefix}.pre_quant_scale" in f.keys():
        pqs = f.get_tensor(f"{prefix}.pre_quant_scale").to(device)
    if pqs is not None:
        x = x * pqs.to(x.dtype)
    w = dequant_nvfp4(
        load_tensor(f, f"{prefix}.weight").to(device),
        load_tensor(f, f"{prefix}.weight_scale").to(device),
        load_tensor(f, f"{prefix}.weight_scale_2").to(device),
    ).to(compute)
    return F.linear(x, w)


def embed_lookup(f: safe_open, ids: torch.Tensor, device) -> torch.Tensor:
    """int8 per-row scale embedding -> [L, hidden] f32."""
    w = load_tensor(f, "model.embed_tokens.weight").to(device)
    s = load_tensor(f, "model.embed_tokens.weight_scale").to(device).float()
    wf = (w.to(torch.float32) * s).to(torch.float32)
    return F.embedding(ids.to(device), wf)


def tokenize(prompt: str) -> list[int]:
    from transformers import AutoTokenizer

    if not TOK_DIR.exists():
        raise SystemExit(f"tokenizer assets not found at {TOK_DIR}")
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR), add_prefix_space=False)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        ids = [PAD_ID]
    return ids


# ---------------------------------------------------------------------------
# Image load / Qwen3-VL preprocess
# ---------------------------------------------------------------------------


def load_image_hwc01(path: Path) -> torch.Tensor:
    """Load image as float [H,W,3] in 0..1 (RGB)."""
    from PIL import Image
    import numpy as np

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def process_qwen3vl_image(
    image_hwc: torch.Tensor,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single image [H,W,C] 0..1 → (flatten_patches [N, C*t*p*p], grid_thw [1,3]).

    Mirrors Comfy ``process_qwen2vl_images`` with mean/std 0.5 (Qwen3-VL).
    """
    if image_hwc.dim() != 3 or image_hwc.shape[-1] != 3:
        raise ValueError(f"expected [H,W,3], got {tuple(image_hwc.shape)}")
    height, width, _ = image_hwc.shape
    img = image_hwc.permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    if device is not None:
        img = img.to(device)

    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    img = F.interpolate(img, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    img = (img - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    # Temporal pad: repeat frame to fill temporal_patch_size=2 (image path).
    pixel = img.squeeze(0).unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)  # [2,3,H,W]
    patches = pixel.reshape(
        1,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], device=img.device, dtype=torch.long)
    return flatten, grid_thw


# ---------------------------------------------------------------------------
# Vision tower (Qwen3-VL / Qwen35Vision + DeepStack), weight-streamed
# ---------------------------------------------------------------------------


def _vlinear(f: safe_open, prefix: str, x: torch.Tensor, compute: torch.dtype, device) -> torch.Tensor:
    w = load_tensor(f, f"{prefix}.weight").to(device=device, dtype=compute)
    bname = f"{prefix}.bias"
    b = load_tensor(f, bname).to(device=device, dtype=compute) if bname in f.keys() else None
    return F.linear(x, w, b)


def _vlayernorm(f: safe_open, prefix: str, x: torch.Tensor, compute: torch.dtype, device, eps: float = 1e-6):
    w = load_tensor(f, f"{prefix}.weight").to(device=device, dtype=compute)
    b = load_tensor(f, f"{prefix}.bias").to(device=device, dtype=compute)
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_vision(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _vision_rot_pos_emb(grid_thw: torch.Tensor, spatial_merge_size: int, head_dim: int, device) -> torch.Tensor:
    """Per-patch rotary freqs flattened to [N_tokens, head_dim]."""
    dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    merge_size = spatial_merge_size
    grid_list = grid_thw.tolist()
    max_hw = max(max(h, w) for _, h, w in grid_list)
    seq = torch.arange(max_hw, device=device, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)  # [max_hw, dim/2]

    total = sum(int(t * h * w) for t, h, w in grid_list)
    pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
    offset = 0
    for num_frames, height, width in grid_list:
        num_frames, height, width = int(num_frames), int(height), int(width)
        merged_h, merged_w = height // merge_size, width // merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)
        n = coords.shape[0]
        pos_ids[offset : offset + n] = coords
        offset += n
    # freq_table[pos] -> [N, 2, dim/2] -> flatten last two -> [N, dim]
    emb = freq_table[pos_ids].flatten(1)
    return emb


def _vision_fast_pos_embed(
    f: safe_open,
    grid_thw: torch.Tensor,
    *,
    compute: torch.dtype,
    device,
    num_position_embeddings: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Bilinear interpolate learned 2D pos embed onto each image grid (Comfy)."""
    pos_w = load_tensor(f, "visual.pos_embed.weight").to(device=device, dtype=compute)  # [2304, 1152]
    num_grid_per_side = int(num_position_embeddings**0.5)
    grid_list = grid_thw.tolist()
    grid_ts = [int(r[0]) for r in grid_list]
    grid_hs = [int(r[1]) for r in grid_list]
    grid_ws = [int(r[2]) for r in grid_list]

    idx_list: list[list[int]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]
    for t, h, w in grid_list:
        h, w = int(h), int(w)
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, device=device)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, device=device)
        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clamp(max=num_grid_per_side - 1)
        w_ceil = (w_idxs.int() + 1).clamp(max=num_grid_per_side - 1)
        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()
        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side
        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for j in range(4):
            idx_list[j].extend(indices[j].tolist())
            weight_list[j].extend(weights[j].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=compute, device=device)
    pos_embeds = pos_w[idx_tensor] * weight_tensor[:, :, None]
    patch_pos = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    parts = patch_pos.split([h * w for h, w in zip(grid_hs, grid_ws)])
    out = []
    merge = spatial_merge_size
    for pe, t, h, w in zip(parts, grid_ts, grid_hs, grid_ws):
        pe = pe.repeat(t, 1)
        pe = (
            pe.view(t, h // merge, merge, w // merge, merge, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        out.append(pe)
    return torch.cat(out, dim=0)


def _vision_attn(
    f: safe_open,
    prefix: str,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    compute: torch.dtype,
    device,
    num_heads: int,
) -> torch.Tensor:
    seq = x.shape[0]
    head_dim = VISION["hidden_size"] // num_heads
    qkv = _vlinear(f, f"{prefix}.attn.qkv", x, compute, device)
    q, k, v = qkv.reshape(seq, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
    # cos/sin: [seq, 1, head_dim] after expand
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    q, k = _apply_rotary_vision(q, k, cos_e, sin_e)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outs = []
    q_s = torch.split(q, lengths, dim=0)
    k_s = torch.split(k, lengths, dim=0)
    v_s = torch.split(v, lengths, dim=0)
    for qi, ki, vi in zip(q_s, k_s, v_s):
        # [1, heads, L, dim]
        qb = qi.transpose(0, 1).unsqueeze(0)
        kb = ki.transpose(0, 1).unsqueeze(0)
        vb = vi.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
        outs.append(o.squeeze(0).transpose(0, 1))  # [L, heads, dim]
    attn = torch.cat(outs, dim=0).reshape(seq, -1)
    return _vlinear(f, f"{prefix}.attn.proj", attn, compute, device)


def _vision_mlp(f: safe_open, prefix: str, x: torch.Tensor, compute: torch.dtype, device) -> torch.Tensor:
    h = _vlinear(f, f"{prefix}.mlp.linear_fc1", x, compute, device)
    h = F.gelu(h, approximate="tanh")
    return _vlinear(f, f"{prefix}.mlp.linear_fc2", h, compute, device)


def _vision_merger(f: safe_open, prefix: str, x: torch.Tensor, compute: torch.dtype, device, merge_dim: int) -> torch.Tensor:
    # Main merger: LayerNorm on hidden, then view to merge_dim.
    x = _vlayernorm(f, f"{prefix}.norm", x, compute, device)
    x = x.reshape(-1, merge_dim)
    x = _vlinear(f, f"{prefix}.linear_fc1", x, compute, device)
    x = F.gelu(x)
    return _vlinear(f, f"{prefix}.linear_fc2", x, compute, device)


def _deepstack_merger(f: safe_open, idx: int, x: torch.Tensor, compute: torch.dtype, device, merge_dim: int) -> torch.Tensor:
    # DeepStack: postshuffle LN (norm after spatial merge view).
    prefix = f"visual.deepstack_merger_list.{idx}"
    x = x.reshape(-1, merge_dim)
    w = load_tensor(f, f"{prefix}.norm.weight").to(device=device, dtype=compute)
    b = load_tensor(f, f"{prefix}.norm.bias").to(device=device, dtype=compute)
    x = F.layer_norm(x, (merge_dim,), weight=w, bias=b, eps=1e-6)
    x = _vlinear(f, f"{prefix}.linear_fc1", x, compute, device)
    x = F.gelu(x)
    return _vlinear(f, f"{prefix}.linear_fc2", x, compute, device)


def run_vision_tower(
    f: safe_open,
    flatten_patches: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    device,
    compute: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Vision forward → (merged [N_merged, 5120], deepstack list of 3)."""
    hs = VISION["hidden_size"]
    heads = VISION["num_heads"]
    head_dim = hs // heads
    merge = VISION["spatial_merge_size"]
    merge_dim = hs * (merge**2)
    ds_idx = list(VISION["deepstack_visual_indexes"])

    x = flatten_patches.to(device=device, dtype=compute)
    # patch_embed: Conv3d weight [1152, 3, 2, 16, 16]
    w = load_tensor(f, "visual.patch_embed.proj.weight").to(device=device, dtype=compute)
    b = load_tensor(f, "visual.patch_embed.proj.bias").to(device=device, dtype=compute)
    x = x.view(-1, VISION["in_channels"], VISION["temporal_patch_size"], VISION["patch_size"], VISION["patch_size"])
    x = F.conv3d(x, w, b, stride=(VISION["temporal_patch_size"], VISION["patch_size"], VISION["patch_size"]))
    x = x.view(-1, hs)

    pos = _vision_fast_pos_embed(
        f,
        grid_thw.to(device),
        compute=compute,
        device=device,
        num_position_embeddings=VISION["num_position_embeddings"],
        spatial_merge_size=merge,
    )
    x = x + pos

    rot = _vision_rot_pos_emb(grid_thw.to(device), merge, head_dim, device)  # [N, head_dim]
    emb = torch.cat((rot, rot), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # cu_seqlens over pre-merge tokens
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu = F.pad(cu.to(device), (1, 0), value=0)

    deepstack: list[torch.Tensor] = []
    for layer in range(VISION["depth"]):
        p = f"visual.blocks.{layer}"
        x = x + _vision_attn(f, p, _vlayernorm(f, f"{p}.norm1", x, compute, device), cu, cos, sin, compute=compute, device=device, num_heads=heads)
        x = x + _vision_mlp(f, p, _vlayernorm(f, f"{p}.norm2", x, compute, device), compute, device)
        if layer in ds_idx:
            deepstack.append(_deepstack_merger(f, ds_idx.index(layer), x, compute, device, merge_dim))

    merged = _vision_merger(f, "visual.merger", x, compute, device, merge_dim)
    return merged, deepstack


# ---------------------------------------------------------------------------
# MiniMax presentation: Picture labels + vision blocks + MRoPE bookkeeping
# ---------------------------------------------------------------------------


def build_minimax_sequence(
    prompt: str,
    images: Sequence[torch.Tensor],
    f: safe_open,
    *,
    device,
    compute: torch.dtype,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Build input embeds + optional MRoPE / deepstack / token tags.

    Returns:
      embeds [L, 5120] compute dtype on device
      position_ids [3, L] or None (pure text)
      visual_pos_masks [L] bool or None
      deepstack list (len 3) or []
      token_tags [L] long (0=vision block incl start/end, 1=text)
    """
    # Token plan: list of either int token-id or ("img", image_index)
    plan: list = []
    for i, _img in enumerate(images):
        plan.extend(tokenize(f"<Picture {i + 1}>: "))
        plan.append(VISION_START)
        plan.append(("img", i))
        plan.append(VISION_END)
    plan.extend(tokenize(prompt) if prompt else [])
    if not plan:
        plan = [PAD_ID]

    # Run vision per image, collect merged + deepstack + grid
    img_merged: list[torch.Tensor] = []
    img_deep: list[list[torch.Tensor]] = []
    img_grid: list[torch.Tensor] = []
    for img in images:
        flat, grid = process_qwen3vl_image(img, device=device)
        merged, ds = run_vision_tower(f, flat, grid, device=device, compute=compute)
        img_merged.append(merged)
        img_deep.append(ds)
        img_grid.append(grid)
        print(f"[text] vision image → merged {tuple(merged.shape)} grid={grid.tolist()}", flush=True)

    # Materialize embeds; record image spans for MRoPE / deepstack / tags
    pieces: list[torch.Tensor] = []
    embeds_info = []  # {type, index, size, grid, deepstack}
    token_tag_parts: list[torch.Tensor] = []
    pos = 0
    for item in plan:
        if isinstance(item, tuple) and item[0] == "img":
            mi = item[1]
            m = img_merged[mi].to(device=device, dtype=compute)
            n = m.shape[0]
            pieces.append(m)
            embeds_info.append(
                {
                    "type": "image",
                    "index": pos,
                    "size": n,
                    "grid": img_grid[mi],
                    "deepstack": img_deep[mi],
                }
            )
            token_tag_parts.append(torch.zeros(n, dtype=torch.long, device=device))
            pos += n
        else:
            tid = int(item)
            e = embed_lookup(f, torch.tensor([tid], dtype=torch.long), device).to(compute)
            pieces.append(e)
            # vision_start / vision_end are VIDEO tags in MiniMax
            tag = 0 if tid in (VISION_START, VISION_END) else 1
            token_tag_parts.append(torch.tensor([tag], dtype=torch.long, device=device))
            pos += 1

    embeds = torch.cat(pieces, dim=0)  # [L, 5120]
    tags = torch.cat(token_tag_parts, dim=0)

    if not embeds_info:
        return embeds, None, None, [], tags

    # Widen VIDEO tags over whole vision block (start + image + end) — already
    # tagged start/end as 0; image tokens too. Good.

    L = embeds.shape[0]
    position_ids, visual_pos_masks, deepstack = build_image_inputs(embeds_info, L, device)
    return embeds, position_ids, visual_pos_masks, deepstack, tags


def build_image_inputs(embeds_info, seq_len: int, device):
    """Comfy Qwen3VL.build_image_inputs + qwen2vl_mrope_position_ids."""
    images = sorted([e for e in embeds_info if e.get("type") == "image"], key=lambda e: e["index"])
    if not images:
        return None, None, None

    position_ids = None
    offset = 0
    visual_pos_masks = torch.zeros(seq_len, dtype=torch.bool, device=device)
    deepstack = None
    for e in images:
        start = int(e["index"])
        end = int(e["size"]) + start
        visual_pos_masks[start:end] = True
        grid = e["grid"]
        ds = e["deepstack"]
        if deepstack is None:
            deepstack = [d for d in ds]
        else:
            deepstack = [torch.cat([deepstack[i], ds[i]], dim=0) for i in range(len(ds))]

        # MRoPE ids (Comfy qwen2vl_mrope_position_ids)
        if position_ids is None:
            position_ids = torch.zeros((3, seq_len), device=device, dtype=torch.float32)
            position_ids[:, :start] = torch.arange(0, start, device=device, dtype=torch.float32)
        len_max = int(grid.max().item()) // 2
        start_next = len_max + start
        position_ids[:, end:] = torch.arange(
            start_next + offset,
            start_next + (seq_len - end) + offset,
            device=device,
            dtype=torch.float32,
        )
        position_ids[0, start:end] = float(start + offset)
        max_d = int(grid[0][1].item()) // 2
        h_ids = (
            torch.arange(start + offset, start + max_d + offset, device=device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, math.ceil((end - start) / max(max_d, 1)))
            .flatten(0)[: end - start]
        )
        position_ids[1, start:end] = h_ids
        max_d = int(grid[0][2].item()) // 2
        w_ids = (
            torch.arange(start + offset, start + max_d + offset, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(math.ceil((end - start) / max(max_d, 1)), 1)
            .flatten(0)[: end - start]
        )
        position_ids[2, start:end] = w_ids
        offset += len_max - (end - start)

    return position_ids, visual_pos_masks, deepstack


# ---------------------------------------------------------------------------
# LM encode
# ---------------------------------------------------------------------------


def encode(
    prompt: str,
    *,
    images: Optional[Sequence[Path | str | torch.Tensor]] = None,
    device: str = "cuda",
    quiet: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the tower; returns (text_embeds [L,5120] f32 CPU, token_tags [L] i64 CPU)."""
    compute = torch.bfloat16
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if not quiet:
        print(f"[text] device {dev} | compute {compute}", flush=True)
    t0 = time.time()

    # Normalize images to list of [H,W,3] 0..1 tensors
    img_tensors: list[torch.Tensor] = []
    if images:
        for im in images:
            if isinstance(im, torch.Tensor):
                t = im.detach().cpu().float()
                if t.dim() == 4 and t.shape[0] == 1:
                    t = t[0]
                if t.dim() == 3 and t.shape[0] == 3:  # CHW
                    t = t.permute(1, 2, 0)
                if t.max() > 1.5:
                    t = t / 255.0
                img_tensors.append(t.contiguous())
            else:
                img_tensors.append(load_image_hwc01(Path(im)))

    if not CKPT.exists():
        raise SystemExit(f"text encoder checkpoint not found: {CKPT}")

    with safe_open(str(CKPT), framework="pt") as f:
        if img_tensors:
            if not quiet:
                print(f"[text] multimodal: {len(img_tensors)} image(s) + prompt", flush=True)
            x, position_ids, visual_pos_masks, deepstack, tags = build_minimax_sequence(
                prompt, img_tensors, f, device=dev, compute=compute
            )
            L = x.shape[0]
            if not quiet:
                print(f"[text] embeds {tuple(x.shape)} ({time.time()-t0:.1f}s)", flush=True)
            cos, sin = mrope_freqs(
                position_ids,
                dev,
                CFG["rope_theta"],
                CFG["head_dim"],
                CFG["rope_dims"],
            )
        else:
            ids = tokenize(prompt)
            L = len(ids)
            if not quiet:
                print(f"[text] prompt: {prompt!r}", flush=True)
                print(f"[text] tokens L={L}: {ids[:12]}{'...' if L > 12 else ''}", flush=True)
            ids_t = torch.tensor(ids, dtype=torch.long)
            x = embed_lookup(f, ids_t, dev).to(compute)
            tags = torch.ones(L, dtype=torch.long, device=dev)
            position_ids = None
            visual_pos_masks = None
            deepstack = []
            if not quiet:
                print(f"[text] embed {tuple(x.shape)} ({time.time()-t0:.1f}s)", flush=True)
            cos, sin = rope_freqs(L, dev, CFG["rope_theta"], CFG["head_dim"])

        for i in range(CFG["num_layers"]):
            p = f"model.layers.{i}."
            t1 = time.time()
            residual = x
            w_norm = load_tensor(f, f"{p}input_layernorm.weight").to(dev).to(compute)
            h = rms_norm(x, w_norm, CFG["rms_eps"])
            q = linear_dequant(h, f, f"{p}self_attn.q_proj", compute, dev)
            k = linear_dequant(h, f, f"{p}self_attn.k_proj", compute, dev)
            v = linear_dequant(h, f, f"{p}self_attn.v_proj", compute, dev)
            S = q.shape[0]
            q = q.view(S, CFG["num_heads"], CFG["head_dim"])
            k = k.view(S, CFG["num_kv_heads"], CFG["head_dim"])
            v = v.view(S, CFG["num_kv_heads"], CFG["head_dim"])
            wq = load_tensor(f, f"{p}self_attn.q_norm.weight").to(dev).to(compute)
            wk = load_tensor(f, f"{p}self_attn.k_norm.weight").to(dev).to(compute)
            q = rms_norm(q, wq, CFG["rms_eps"])
            k = rms_norm(k, wk, CFG["rms_eps"])
            q, k = apply_rope(q, k, cos, sin)
            rep = CFG["num_heads"] // CFG["num_kv_heads"]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            qb = q.permute(1, 0, 2).unsqueeze(0)
            kb = k.permute(1, 0, 2).unsqueeze(0)
            vb = v.permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
            out = out.squeeze(0).permute(1, 0, 2).reshape(S, CFG["num_heads"] * CFG["head_dim"])
            attn_out = linear_dequant(out, f, f"{p}self_attn.o_proj", compute, dev)
            x = residual + attn_out

            residual = x
            w_norm2 = load_tensor(f, f"{p}post_attention_layernorm.weight").to(dev).to(compute)
            h = rms_norm(x, w_norm2, CFG["rms_eps"])
            gate = linear_dequant(h, f, f"{p}mlp.gate_proj", compute, dev)
            up = linear_dequant(h, f, f"{p}mlp.up_proj", compute, dev)
            x = residual + linear_dequant(F.silu(gate) * up, f, f"{p}mlp.down_proj", compute, dev)

            # DeepStack inject into first len(deepstack) LM layers at visual positions.
            if deepstack and visual_pos_masks is not None and i < len(deepstack):
                ds = deepstack[i].to(device=dev, dtype=x.dtype)
                # x[mask] is [N_vis, hidden]; deepstack is [N_vis, hidden]
                x = x.clone()
                x[visual_pos_masks] = x[visual_pos_masks] + ds

            if not quiet and ((i + 1) % 10 == 0 or i + 1 == CFG["num_layers"]):
                print(
                    f"[text] layer {i+1:>2}/{CFG['num_layers']} "
                    f"({time.time()-t1:.2f}s, {time.time()-t0:.1f}s total)",
                    flush=True,
                )

    return x.float().cpu(), tags.cpu()


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Back-compat: `script "prompt" out.safetensors` still works.
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) >= 2 and not raw[0].startswith("-") and not raw[1].startswith("-"):
        # positional form
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("prompt")
        ap.add_argument("out", type=Path)
        ap.add_argument("--image", action="append", default=[], dest="images", help="memory/keyframe image (repeatable)")
        ap.add_argument("--device", default="cuda")
        ap.add_argument("--quiet", action="store_true")
        args = ap.parse_args(raw)
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("prompt")
        ap.add_argument("out", type=Path)
        ap.add_argument("--image", action="append", default=[], dest="images", help="memory/keyframe image (repeatable)")
        ap.add_argument("--device", default="cuda")
        ap.add_argument("--quiet", action="store_true")
        args = ap.parse_args(raw)

    emb, tags = encode(args.prompt, images=args.images or None, device=args.device, quiet=args.quiet)
    payload = {
        "text_embeds": emb.contiguous(),
        "minimax_token_tags": tags.to(torch.int64).contiguous(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(args.out))
    print(
        f"[text] saved {args.out} (embeds {tuple(emb.shape)}, tags {tuple(tags.shape)}, "
        f"mean {emb.mean():.5f}, std {emb.std():.5f}, vision_tokens={(tags == 0).sum().item()})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
