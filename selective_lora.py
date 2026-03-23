from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]

from circuit_ap.attribution import build_lora_profile_from_positive_nodes


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sorted_top_keys(score_map: dict[str, float], top_k: int) -> list[str]:
    return [key for key, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def load_lora_profile(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        profile_path = path / "lora_profile.json"
        support_path = path / "support_ap.json"
        if profile_path.exists():
            return json.loads(profile_path.read_text(encoding="utf-8"))
        if support_path.exists():
            payload = json.loads(support_path.read_text(encoding="utf-8"))
            return build_lora_profile_from_positive_nodes(payload["lora_id"], payload.get("positive_nodes", []))
        raise FileNotFoundError(f"No lora_profile.json or support_ap.json found in {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "module_scores" in payload and "module_step_scores" in payload:
        return payload
    if "positive_nodes" in payload:
        return build_lora_profile_from_positive_nodes(payload["lora_id"], payload.get("positive_nodes", []))
    raise ValueError(f"Unsupported profile payload: {path}")


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    max_value = max(score_map.values())
    if max_value <= 1e-8:
        return {key: 0.0 for key in score_map}
    return {key: float(value / max_value) for key, value in score_map.items()}


def build_profile_mask(
    profile: dict[str, Any],
    *,
    top_modules: int,
    top_steps: int,
    support_mode: str,
) -> dict[str, Any]:
    module_scores = {str(key): float(value) for key, value in profile.get("module_scores", {}).items()}
    step_scores = {str(key): float(value) for key, value in profile.get("step_scores", {}).items()}
    node_scores = {str(key): float(value) for key, value in profile.get("module_step_scores", {}).items()}
    selected_modules = set(_sorted_top_keys(module_scores, top_modules))
    selected_steps = set(_sorted_top_keys(step_scores, top_steps))
    active_nodes: dict[str, float] = {}
    for node, score in node_scores.items():
        module_path, step_index = node.rsplit("@", 1)
        in_module = module_path in selected_modules
        in_step = step_index in selected_steps
        if support_mode == "intersection":
            keep = in_module and in_step
        else:
            keep = in_module or in_step
        if keep:
            active_nodes[node] = score
    return {
        "selected_modules": sorted(selected_modules),
        "selected_steps": sorted(int(step) for step in selected_steps),
        "active_nodes": active_nodes,
        "module_scores": module_scores,
        "step_scores": step_scores,
        "normalized_node_scores": normalize_score_map(active_nodes),
    }


def build_selective_policy(
    lora_ids: Sequence[str],
    profiles: dict[str, dict[str, Any]],
    *,
    lora_weights: dict[str, float],
    denoise_steps: int,
    top_modules: int,
    top_steps: int,
    support_mode: str,
) -> dict[str, Any]:
    masks = {
        lora_id: build_profile_mask(
            profiles[lora_id],
            top_modules=top_modules,
            top_steps=top_steps,
            support_mode=support_mode,
        )
        for lora_id in lora_ids
    }

    timestep_schedule: list[list[str]] = []
    for step_index in range(denoise_steps):
        active_loras = [
            lora_id
            for lora_id in lora_ids
            if step_index in set(masks[lora_id]["selected_steps"])
        ]
        timestep_schedule.append(active_loras)

    module_only_owners: dict[str, str] = {}
    module_candidates = sorted({module for lora_id in lora_ids for module in masks[lora_id]["selected_modules"]})
    for module_path in module_candidates:
        contenders = [
            (
                lora_id,
                lora_weights.get(lora_id, 1.0) * float(masks[lora_id]["module_scores"].get(module_path, 0.0)),
            )
            for lora_id in lora_ids
            if module_path in set(masks[lora_id]["selected_modules"])
        ]
        if not contenders:
            continue
        module_only_owners[module_path] = max(contenders, key=lambda item: item[1])[0]

    module_only_assignments = {
        step_index: {module_path: [owner] for module_path, owner in module_only_owners.items()}
        for step_index in range(denoise_steps)
    }

    module_step_assignments: dict[int, dict[str, list[str]]] = {step_index: {} for step_index in range(denoise_steps)}
    for step_index in range(denoise_steps):
        per_module_scores: dict[str, list[tuple[str, float]]] = {}
        for lora_id in lora_ids:
            normalized = masks[lora_id]["normalized_node_scores"]
            for node, score in normalized.items():
                module_path, step_part = node.rsplit("@", 1)
                if int(step_part) != step_index:
                    continue
                per_module_scores.setdefault(module_path, []).append((lora_id, lora_weights.get(lora_id, 1.0) * score))
        for module_path, contenders in per_module_scores.items():
            if not contenders:
                continue
            owner = max(contenders, key=lambda item: item[1])[0]
            module_step_assignments[step_index][module_path] = [owner]

    return {
        "lora_ids": list(lora_ids),
        "lora_weights": dict(lora_weights),
        "support_mode": support_mode,
        "top_modules": top_modules,
        "top_steps": top_steps,
        "masks": masks,
        "timestep_schedule": timestep_schedule,
        "module_only_assignments": module_only_assignments,
        "module_step_assignments": module_step_assignments,
    }


class SelectiveStepCallback:
    def __init__(self, controller: "SelectiveLoRAController") -> None:
        self.controller = controller

    def __call__(self, pipeline: Any, step_index: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        self.controller.current_step = step_index + 1
        pipeline.disable_lora()
        return {}


@dataclass
class SelectiveLoRAController:
    pipeline: Any
    assignments_by_step: dict[int, dict[str, list[str]]]
    modules: dict[str, Any]

    def __post_init__(self) -> None:
        self.current_step = 0
        self._handles: list[Any] = []
        for path, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._build_pre_hook(path)))
            self._handles.append(module.register_forward_hook(self._build_post_hook()))

    def _build_pre_hook(self, path: str):
        def _hook(_module: Any, _inputs: Any) -> None:
            adapters = self.assignments_by_step.get(self.current_step, {}).get(path)
            if not adapters:
                self.pipeline.disable_lora()
                return
            self.pipeline.enable_lora()
            self.pipeline.set_adapters(adapters)

        return _hook

    def _build_post_hook(self):
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            self.pipeline.disable_lora()
            return output

        return _hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def discover_target_modules(unet: Any, module_paths: Sequence[str]) -> dict[str, Any]:
    wanted = set(module_paths)
    return {path: module for path, module in unet.named_modules() if path in wanted}


def run_selective_generation(
    *,
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    lora_ids: Sequence[str],
    seed: int,
    device: str,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    module_assignments_by_step: dict[int, dict[str, list[str]]],
) -> tuple[Any, float]:
    if torch is None:
        raise ImportError("torch is required for selective generation.")
    target_paths = sorted({path for per_step in module_assignments_by_step.values() for path in per_step})
    modules = discover_target_modules(pipeline.unet, target_paths)
    controller = SelectiveLoRAController(pipeline=pipeline, assignments_by_step=module_assignments_by_step, modules=modules)
    callback = SelectiveStepCallback(controller)
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline.disable_lora()
    start = time.perf_counter()
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=denoise_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="pil",
            return_dict=False,
            cross_attention_kwargs={"scale": lora_scale},
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    finally:
        controller.close()
        pipeline.disable_lora()
    return result, float(time.perf_counter() - start)
