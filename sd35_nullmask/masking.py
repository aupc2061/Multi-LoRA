from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependent
    torch = None  # type: ignore[assignment]


@dataclass
class MaskBuildResult:
    scores_by_adapter: dict[str, Any]
    binary_masks_by_adapter: dict[str, Any]
    ownership_ratio_by_adapter: dict[str, float]
    unowned_ratio: float
    patch_owner_index: Any


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("torch is required for SD3.5 null-mask tensor operations.")
    return torch


def reduce_attention_to_patch_scores(attention: Any, token_indices: list[int]) -> Any:
    _torch = _require_torch()
    if attention.ndim < 2:
        raise ValueError("attention tensor must have at least 2 dimensions")
    if not token_indices:
        raise ValueError("token_indices cannot be empty")
    token_axis = attention.ndim - 1
    patch_axis = attention.ndim - 2
    selected = attention.index_select(token_axis, _torch.tensor(token_indices, device=attention.device))
    reduced = selected.mean(dim=token_axis)
    if reduced.ndim > 1:
        dims = tuple(range(reduced.ndim - 1))
        reduced = reduced.mean(dim=dims)
    if patch_axis != reduced.ndim - 1:
        reduced = reduced.reshape(-1)
    return reduced.float()


def build_exclusive_binary_masks(
    scores_by_adapter: dict[str, Any],
    *,
    confidence_threshold: float = 0.0,
) -> MaskBuildResult:
    _require_torch()
    if not scores_by_adapter:
        raise ValueError("scores_by_adapter cannot be empty")
    adapter_ids = list(scores_by_adapter.keys())
    first = scores_by_adapter[adapter_ids[0]]
    if first.ndim != 1:
        raise ValueError("Each patch score tensor must be 1D")
    num_patches = int(first.shape[0])
    stacked = []
    for adapter_id in adapter_ids:
        scores = scores_by_adapter[adapter_id].float().reshape(-1)
        if scores.shape[0] != num_patches:
            raise ValueError("All patch score tensors must have the same length")
        stacked.append(scores)
    score_matrix = torch.stack(stacked, dim=0)
    best_scores, owner_idx = score_matrix.max(dim=0)
    # Bug 9: use strict < so patches with score == threshold are still owned (not unowned).
    # With the default threshold=0.0 and <=, ~70% of patches would be zeroed by smooth_and_binarize
    # and then incorrectly marked unowned, leaving most of the image with no LoRA contribution.
    low_confidence = best_scores < float(confidence_threshold)
    owner_idx = owner_idx.clone()
    owner_idx[low_confidence] = -1

    masks: dict[str, torch.Tensor] = {}
    ownership_ratio: dict[str, float] = {}
    for idx, adapter_id in enumerate(adapter_ids):
        mask = owner_idx == idx
        masks[adapter_id] = mask
        ownership_ratio[adapter_id] = float(mask.float().mean().item())
    unowned_ratio = float((owner_idx == -1).float().mean().item())
    return MaskBuildResult(
        scores_by_adapter={key: value.float().reshape(-1) for key, value in scores_by_adapter.items()},
        binary_masks_by_adapter=masks,
        ownership_ratio_by_adapter=ownership_ratio,
        unowned_ratio=unowned_ratio,
        patch_owner_index=owner_idx,
    )


def summarize_mask_stats(result: MaskBuildResult) -> dict[str, Any]:
    return {
        "ownership_ratio_by_adapter": result.ownership_ratio_by_adapter,
        "unowned_ratio": result.unowned_ratio,
        "num_patches": int(result.patch_owner_index.numel()),
    }
