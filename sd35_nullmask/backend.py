from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

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
        self._loaded_adapter_ids: list[str] = []
        self._loaded_triggers: dict[str, str] = {}

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

        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = _dtype_map.get(self.config.dtype, torch.bfloat16)
        pipeline = StableDiffusion3Pipeline.from_pretrained(self.config.model_name, dtype=dtype)
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

    def load_adapters(self, adapters: list[InventoryAdapter], resolved_triggers: list[ResolvedTrigger] | None = None) -> None:
        if self.pipeline is None:
            raise SD35BackendError("Pipeline must be loaded before loading adapters.")
        for adapter in adapters:
            if not adapter.local_files:
                raise FileNotFoundError(f"Local adapter file missing for {adapter.adapter_id}")
            weight_name = Path(adapter.local_files[0]).name
            self.pipeline.load_lora_weights(adapter.expected_local_dir, weight_name=weight_name, adapter_name=adapter.adapter_id)
        self._loaded_adapter_ids = [adapter.adapter_id for adapter in adapters]
        if resolved_triggers is not None:
            self._loaded_triggers = {trigger.adapter_id: trigger.phrase for trigger in resolved_triggers}

    def run_preflight(self, adapters: list[InventoryAdapter], resolved_triggers: list[ResolvedTrigger]) -> dict[str, Any]:
        runtime = self.load_pipeline()
        self.load_adapters(adapters, resolved_triggers)
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
        if self.pipeline is None:
            raise SD35BackendError("Pipeline must be loaded before running a method.")
        if not self._loaded_adapter_ids:
            raise SD35BackendError("Adapters must be loaded before running a method.")
        from .inference import SD35NullMaskEngine

        trigger_list = [self._loaded_triggers.get(aid, "") for aid in self._loaded_adapter_ids]
        engine = SD35NullMaskEngine(
            pipeline=self.pipeline,
            config=self.config,
            adapter_ids=self._loaded_adapter_ids,
            triggers=trigger_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        image = engine.generate(method=method, seed=seed)
        return {"image": image, "method": method, "seed": seed}

    def run_all_methods(
        self,
        *,
        methods: list[str],
        seeds: list[int],
        prompt: str,
        negative_prompt: str = "",
        pair_id: str,
        out_root: Path | str,
        config: SD35NullMaskConfig | None = None,
    ) -> list[dict[str, Any]]:
        if self.pipeline is None:
            raise SD35BackendError("Pipeline must be loaded before running methods.")
        if not self._loaded_adapter_ids:
            raise SD35BackendError("Adapters must be loaded before running methods.")
        from .inference import SD35NullMaskEngine

        cfg = config if config is not None else self.config
        pair_dir = Path(out_root) / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        trigger_list = [self._loaded_triggers.get(aid, "") for aid in self._loaded_adapter_ids]
        engine = SD35NullMaskEngine(
            pipeline=self.pipeline,
            config=cfg,
            adapter_ids=self._loaded_adapter_ids,
            triggers=trigger_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        records: list[dict[str, Any]] = []
        total_images = len(methods) * len(seeds)
        outer_bar = (
            _tqdm(total=total_images, desc=f"  {pair_id}", unit="img", ncols=100, position=0)
            if _TQDM_AVAILABLE
            else None
        )
        for method in methods:
            for seed in seeds:
                if outer_bar is not None:
                    outer_bar.set_description(f"  {pair_id}  [{method}] seed={seed}")
                print(f"\n→ generating  method={method}  seed={seed}", flush=True)
                image = engine.generate(method=method, seed=seed)
                img_path = pair_dir / f"{method}_seed{seed}.png"
                image.save(str(img_path))
                print(f"  saved → {img_path}", flush=True)
                records.append({"method": method, "seed": seed, "path": str(img_path)})
                if outer_bar is not None:
                    outer_bar.update(1)
        if outer_bar is not None:
            outer_bar.close()
        return records

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
