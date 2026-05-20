from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SD35NullMaskConfig
from .inventory import InventoryAdapter
from .masking import MaskBuildResult
from .projection import StructuralBasis
from .prompting import ResolvedTrigger


class SD35BackendError(RuntimeError):
    pass


@dataclass
class SD35RuntimeInfo:
    model_name: str
    device: str
    dtype: str
    transformer_block_count: int | None


class SD35PipelineBackend:
    def __init__(self, config: SD35NullMaskConfig) -> None:
        self.config = config
        self.pipeline: Any | None = None

    def require_runtime(self) -> None:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:  # pragma: no cover - runtime dependent
            raise SD35BackendError(
                "SD3.5 runtime dependencies are not installed. "
                "Install a diffusers/transformers/torch stack with Stable Diffusion 3.5 support."
            ) from exc

    def load_pipeline(self) -> SD35RuntimeInfo:
        self.require_runtime()
        import torch
        from diffusers import StableDiffusion3Pipeline

        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
        pipeline = StableDiffusion3Pipeline.from_pretrained(self.config.model_name, torch_dtype=dtype)
        pipeline = pipeline.to(self.config.device)
        self.pipeline = pipeline
        block_count = self._count_transformer_blocks(pipeline)
        return SD35RuntimeInfo(
            model_name=self.config.model_name,
            device=self.config.device,
            dtype=self.config.dtype,
            transformer_block_count=block_count,
        )

    def _count_transformer_blocks(self, pipeline: Any) -> int | None:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            return None
        if hasattr(transformer, "transformer_blocks"):
            return int(len(transformer.transformer_blocks))
        if hasattr(transformer, "blocks"):
            return int(len(transformer.blocks))
        return None

    def load_adapters(self, adapters: list[InventoryAdapter]) -> None:
        if self.pipeline is None:
            raise SD35BackendError("Pipeline must be loaded before loading adapters.")
        for adapter in adapters:
            if not adapter.local_files:
                raise FileNotFoundError(f"Local adapter file missing for {adapter.adapter_id}")
            weight_name = Path(adapter.local_files[0]).name
            self.pipeline.load_lora_weights(adapter.expected_local_dir, weight_name=weight_name, adapter_name=adapter.adapter_id)

    def run_preflight(self, adapters: list[InventoryAdapter], resolved_triggers: list[ResolvedTrigger]) -> dict[str, Any]:
        runtime = self.load_pipeline()
        self.load_adapters(adapters)
        return {
            "runtime": runtime.__dict__,
            "adapters": [
                {
                    "adapter_id": adapter.adapter_id,
                    "repo_id": adapter.repo_id,
                    "local_file": adapter.local_files[0] if adapter.local_files else None,
                    "trigger_phrase": resolved.phrase,
                }
                for adapter, resolved in zip(adapters, resolved_triggers)
            ],
        }

    def run_method(
        self,
        *,
        method: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        intervention_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise SD35BackendError(
            f"Method '{method}' is scaffolded but not executed in this environment yet. "
            "The backend exists to host SD3.5-specific intervention hooks once the runtime is installed and validated."
        )

    @staticmethod
    def save_json(path: str | Path, payload: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_intervention_payload(
    *,
    method: str,
    mask_result: MaskBuildResult | None = None,
    bases_by_adapter: dict[str, StructuralBasis | None] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"method": method}
    if mask_result is not None:
        payload["mask_stats"] = {
            "ownership_ratio_by_adapter": mask_result.ownership_ratio_by_adapter,
            "unowned_ratio": mask_result.unowned_ratio,
        }
    if bases_by_adapter is not None:
        payload["structural_basis"] = {
            adapter_id: (
                {
                    "rank": int(basis.vectors.shape[1]),
                    "singular_values": [float(value) for value in basis.singular_values.tolist()],
                }
                if basis is not None
                else None
            )
            for adapter_id, basis in bases_by_adapter.items()
        }
    return payload
