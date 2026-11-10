"""Vectorized PyTorch rolling primitives and depthwise convolution operators.

Provides zero-Python-loop implementations of technical primitives (EMA, linear decay,
rolling max/min, rolling zscore, rolling rank, rolling correlation, argmax/argmin)
with finite-value safety clamping and multi-GPU device key normalization.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F

# Thread-safe pre-computed convolution weight cache
_WEIGHT_CACHE: Dict[Tuple[str, int, int, torch.dtype, str], torch.Tensor] = {}


def _norm_device_str(device: torch.device | str) -> str:
    """Normalize device representations (e.g., 'cuda:0' vs 'cuda') to canonical strings."""
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        return f"cuda:{idx}"
    return "cpu"


def _get_ema_weights(span: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    dev_str = _norm_device_str(device)
    key = ("ema", span, w, dtype, dev_str)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        alpha = 2.0 / (span + 1.0)
        weights = torch.tensor([(1.0 - alpha) ** i for i in range(w - 1, -1, -1)], dtype=dtype, device=device)
        cached = (weights / weights.sum()).view(1, 1, w)
        _WEIGHT_CACHE[key] = cached
    return cached


def _get_decay_weights(window: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    dev_str = _norm_device_str(device)
    key = ("decay", window, 0, dtype, dev_str)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        weights = torch.arange(1, window + 1, dtype=dtype, device=device)
        cached = (weights / weights.sum()).view(1, 1, window)
        _WEIGHT_CACHE[key] = cached
    return cached


def _flatten_for_conv(x: torch.Tensor) -> Tuple[torch.Tensor, int, Tuple[int, ...], int]:
    """Reshapes [T], [B, T], or [B, T, F] to (N, 1, T) along the time axis."""
    orig_shape = x.shape
    if x.ndim == 1:
        return x.reshape(1, 1, -1), x.shape[0], orig_shape, 1
    if x.ndim == 2:
        return x.reshape(x.shape[0], 1, -1), x.shape[-1], orig_shape, 2
    if x.ndim == 3:
        B, T, F_dim = x.shape
        flat = x.permute(0, 2, 1).reshape(B * F_dim, 1, T)
        return flat, T, orig_shape, 3
    flat = x.reshape(-1, 1, x.shape[-1])
    return flat, x.shape[-1], orig_shape, x.ndim


def _restore_from_conv(conv: torch.Tensor, orig_shape: Tuple[int, ...], ndim: int) -> torch.Tensor:
    if ndim == 3:
        B, T, F_dim = orig_shape
        return conv.reshape(B, F_dim, T).permute(0, 2, 1)
    return conv.reshape(orig_shape)


def div_safe_tensor(l: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe division preserving denominator sign and clamping output to [-1e4, 1e4]."""
    l_c = torch.nan_to_num(torch.clamp(l, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    r_c = torch.nan_to_num(torch.clamp(r, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    r_abs = torch.abs(r_c)
    denom_sign = torch.sign(r_c)
    denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
    safe_denom = denom_sign * torch.clamp(r_abs, min=eps)
    res = torch.where(r_abs > eps, l_c / safe_denom, torch.zeros_like(l_c))
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_delta(x: torch.Tensor, period: int = 5) -> torch.Tensor:
    """Computes x[t] - x[t - period], zero-padded for prefix t < period."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    res = torch.zeros_like(clean_x)
    if clean_x.ndim == 3:
        T = clean_x.shape[1]
        if T > period:
            res[:, period:, :] = clean_x[:, period:, :] - clean_x[:, :-period, :]
    else:
        T = clean_x.shape[-1]
        if T > period:
            res[..., period:] = clean_x[..., period:] - clean_x[..., :-period]
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_delay(x: torch.Tensor, period: int = 1) -> torch.Tensor:
    """Computes lagged series x[t - period] with edge-padded prefix."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    res = torch.zeros_like(clean_x)
    if clean_x.ndim == 3:
        T = clean_x.shape[1]
        if T == 0:
            return res
        if T > period:
            res[:, period:, :] = clean_x[:, :-period, :]
            res[:, :period, :] = clean_x[:, 0:1, :]
        else:
            res[...] = clean_x[:, 0:1, :]
    else:
        T = clean_x.shape[-1]
        if T == 0:
            return res
        if T > period:
            res[..., period:] = clean_x[..., :-period]
            res[..., :period] = clean_x[..., 0:1]
        else:
            res[...] = clean_x[..., 0:1]
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_ema(x: torch.Tensor, span: int = 10) -> torch.Tensor:
    """Exponential moving average via 1D depthwise convolution with zero Python loops."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    w = min(T, max(span * 3, 20))
    weights = _get_ema_weights(span, w, clean_x.dtype, clean_x.device)
    padded = F.pad(flat_x, (w - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    res = _restore_from_conv(conv, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_max(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling maximum via 1D max pooling with finite bound padding."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=-1e12, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = F.max_pool1d(padded, kernel_size=window, stride=1)
    res = _restore_from_conv(pooled, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_min(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling minimum via negated 1D max pooling."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=1e12, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(-clean_x)
    if T == 0:
        return clean_x.clone()
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = -F.max_pool1d(padded, kernel_size=window, stride=1)
    res = _restore_from_conv(pooled, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_rank_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Sliding-window percentile rank over 20 periods via PyTorch unfold."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.full_like(clean_x, 0.5)
    prefix_len = min(T, window)
    for i in range(prefix_len):
        hist = clean_x[..., : i + 1]
        target = clean_x[..., i : i + 1]
        less = (hist < target).sum(dim=-1, keepdim=True).to(clean_x.dtype)
        eq = (hist == target).sum(dim=-1, keepdim=True).to(clean_x.dtype)
        res[..., i : i + 1] = (less + 0.5 * eq) / float(i + 1)
    if T > window:
        windows = clean_x.unfold(dimension=-1, size=window, step=1)
        target = clean_x[..., window - 1 :].unsqueeze(-1)
        less = (windows < target).sum(dim=-1).to(clean_x.dtype)
        eq = (windows == target).sum(dim=-1).to(clean_x.dtype)
        res[..., window - 1 :] = (less + 0.5 * eq) / float(window)
    res = torch.nan_to_num(res, nan=0.5)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


def ts_zscore_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling z-score clamped to [-4.0, 4.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.zeros_like(clean_x)
    prefix_len = min(T, window)
    for i in range(1, prefix_len):
        hist = clean_x[..., : i + 1]
        mean = hist.mean(dim=-1, keepdim=True)
        std = hist.std(dim=-1, unbiased=True, keepdim=True)
        target = clean_x[..., i : i + 1]
        res[..., i : i + 1] = (target - mean) / (std + 1e-6)
    if T >= window:
        windows = clean_x.unfold(dimension=-1, size=window, step=1)
        mean = windows.mean(dim=-1)
        std = windows.std(dim=-1, unbiased=True)
        target = clean_x[..., window - 1 :]
        res[..., window - 1 :] = (target - mean) / (std + 1e-6)
    res = torch.nan_to_num(torch.clamp(res, min=-4.0, max=4.0), nan=0.0)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


def ts_decay_linear(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Linear time-decay weighted moving average via 1D convolution."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    weights = _get_decay_weights(window, clean_x.dtype, clean_x.device)
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    res = _restore_from_conv(conv, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_corr_20(x: torch.Tensor, y: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling Pearson correlation clamped to [-1.0, 1.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    clean_y = torch.nan_to_num(torch.clamp(y, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
        clean_y = clean_y.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.zeros_like(clean_x)
    prefix_len = min(T, window)
    for i in range(2, prefix_len):
        hx = clean_x[..., : i + 1]
        hy = clean_y[..., : i + 1]
        mx = hx.mean(dim=-1, keepdim=True)
        my = hy.mean(dim=-1, keepdim=True)
        cov = ((hx - mx) * (hy - my)).sum(dim=-1, keepdim=True) / float(i)
        sx = hx.std(dim=-1, unbiased=True, keepdim=True)
        sy = hy.std(dim=-1, unbiased=True, keepdim=True)
        res[..., i : i + 1] = cov / (sx * sy + 1e-6)
    if T >= window:
        wx = clean_x.unfold(dimension=-1, size=window, step=1)
        wy = clean_y.unfold(dimension=-1, size=window, step=1)
        mx = wx.mean(dim=-1, keepdim=True)
        my = wy.mean(dim=-1, keepdim=True)
        cov = ((wx - mx) * (wy - my)).sum(dim=-1) / float(window - 1)
        sx = wx.std(dim=-1, unbiased=True)
        sy = wy.std(dim=-1, unbiased=True)
        res[..., window - 1 :] = cov / (sx * sy + 1e-6)
    res = torch.nan_to_num(torch.clamp(res, min=-1.0, max=1.0), nan=0.0)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


def ts_std(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling standard deviation via unfold."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T <= 1:
        return torch.zeros_like(clean_x)
    w = min(T, window)
    s = flat.squeeze(1)
    prefix = s[:, :1].repeat(1, w - 1)
    padded = torch.cat([prefix, s], dim=-1)
    unfolded = padded.unfold(dimension=-1, size=w, step=1)
    std = unfolded.std(dim=-1, unbiased=True)
    res = _restore_from_conv(std.unsqueeze(1), orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)


def ts_median(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Vectorized rolling median via unfold."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    w = min(T, window)
    s = flat.squeeze(1)
    prefix = s[:, :1].repeat(1, w - 1)
    padded = torch.cat([prefix, s], dim=-1)
    unfolded = padded.unfold(dimension=-1, size=w, step=1)
    med = unfolded.median(dim=-1).values
    res = _restore_from_conv(med.unsqueeze(1), orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)


def ts_argmax(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Normalized index of rolling maximum in [0.0, 1.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return torch.zeros_like(clean_x)
    w = min(T, window)
    if w <= 1:
        return torch.full_like(clean_x, 0.5)
    s = flat.squeeze(1)
    prefix = s[:, :1].repeat(1, w - 1)
    padded = torch.cat([prefix, s], dim=-1)
    unfolded = padded.unfold(dimension=-1, size=w, step=1)
    am = unfolded.argmax(dim=-1).to(dtype=flat.dtype) / float(w - 1)
    res = _restore_from_conv(am.unsqueeze(1), orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=0.0, max=1.0), nan=0.5)


def ts_argmin(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Normalized index of rolling minimum in [0.0, 1.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return torch.zeros_like(clean_x)
    w = min(T, window)
    if w <= 1:
        return torch.full_like(clean_x, 0.5)
    s = flat.squeeze(1)
    prefix = s[:, :1].repeat(1, w - 1)
    padded = torch.cat([prefix, s], dim=-1)
    unfolded = padded.unfold(dimension=-1, size=w, step=1)
    am = unfolded.argmin(dim=-1).to(dtype=flat.dtype) / float(w - 1)
    res = _restore_from_conv(am.unsqueeze(1), orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=0.0, max=1.0), nan=0.5)
