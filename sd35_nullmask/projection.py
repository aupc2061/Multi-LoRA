from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependent
    torch = None  # type: ignore[assignment]


@dataclass
class StructuralBasis:
    vectors: Any
    singular_values: Any
    centered: bool


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("torch is required for SD3.5 null-projection tensor operations.")
    return torch


def _flatten_hidden_states(hidden_states: Any) -> Any:
    if hidden_states.ndim == 2:
        return hidden_states
    if hidden_states.ndim == 3:
        # Bug 10: reshape(-1, C) would mix batch items if B>1, corrupting the SVD basis.
        # Use only the first sample — SVD captures per-sample structural directions.
        return hidden_states[0]  # [N, C]
    raise ValueError("hidden_states must be rank-2 or rank-3")


def compute_structural_basis(hidden_states: Any, *, rank: int = 1, center: bool = True) -> StructuralBasis | None:
    _require_torch()
    if rank <= 0:
        return None
    matrix = _flatten_hidden_states(hidden_states).float()
    if matrix.shape[0] == 0:
        return None
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, vh.shape[0])
    return StructuralBasis(vectors=vh[:kept].transpose(0, 1), singular_values=singular_values[:kept], centered=center)


def project_delta_to_nullspace(delta: Any, basis: StructuralBasis | None) -> Any:
    _require_torch()
    if basis is None or basis.vectors.numel() == 0:
        return delta
    flat = delta.reshape(-1, delta.shape[-1]).float()
    # Ensure basis vectors are on the same device/dtype as delta (Bug 7 companion fix)
    v = basis.vectors.to(device=flat.device, dtype=flat.dtype)
    projection = flat @ v @ v.transpose(0, 1)
    out = flat - projection
    return out.reshape_as(delta).to(dtype=delta.dtype, device=delta.device)
