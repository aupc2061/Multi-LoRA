from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]

from circuit_utils import infer_prompt_for_lora
from utils import get_prompt


EPS = 1e-8


def _cpu_clone(value: Any) -> Any:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_clone(item) for item in value]
    return value


def _to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if isinstance(value, torch.Tensor):
        out = value.to(device=device)
        if dtype is not None and torch.is_floating_point(out):
            out = out.to(dtype=dtype)
        return out
    if isinstance(value, dict):
        return {key: _to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device, dtype) for item in value]
    return value


def _tensor_mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _tensor_mean(values)
    return float(sum((float(v) - mean) ** 2 for v in values) / (len(values) - 1))


def directional_recovery(patched_noise: torch.Tensor, corr_noise: torch.Tensor, direction: torch.Tensor, eps: float = EPS) -> float:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    patched = patched_noise.detach().float()
    corr = corr_noise.detach().float()
    d_t = direction.detach().float()
    numer = torch.sum((patched - corr) * d_t).item()
    denom = torch.sum(d_t * d_t).item() + eps
    return float(numer / denom)


def activation_direction_recovery(
    patched_activation: torch.Tensor,
    corr_activation: torch.Tensor,
    target_delta: torch.Tensor,
    eps: float = EPS,
) -> float:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    patched = patched_activation.detach().float()
    corr = corr_activation.detach().float()
    delta = target_delta.detach().float()
    numer = torch.sum((patched - corr) * delta).item()
    denom = torch.sum(delta * delta).item() + eps
    return float(numer / denom)


def apply_classifier_free_guidance(
    noise_pred: torch.Tensor,
    *,
    do_classifier_free_guidance: bool,
    guidance_scale: float,
    guidance_rescale: float,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if not do_classifier_free_guidance:
        return noise_pred

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    if guidance_rescale > 0.0:
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = guided.std(dim=list(range(1, guided.ndim)), keepdim=True)
        rescaled = guided * (std_text / std_cfg.clamp_min(EPS))
        guided = guidance_rescale * rescaled + (1 - guidance_rescale) * guided
    return guided


def discover_candidate_modules(unet: Any, class_names: Sequence[str]) -> dict[str, Any]:
    allowed = {name.strip() for name in class_names if name.strip()}
    out: dict[str, Any] = {}
    for path, module in unet.named_modules():
        if not path:
            continue
        if module.__class__.__name__ not in allowed:
            continue
        if not (path.startswith("down_blocks") or path.startswith("mid_block") or path.startswith("up_blocks")):
            continue
        out[path] = module
    if not out:
        raise ValueError(f"No candidate modules found for classes: {sorted(allowed)}")
    return out


@dataclass
class StepTrace:
    step_index: int
    timestep: int
    latents: torch.Tensor
    latent_model_input: torch.Tensor
    prompt_embeds: torch.Tensor
    timestep_cond: torch.Tensor | None
    added_cond_kwargs: dict[str, Any] | None
    cross_attention_kwargs: dict[str, Any] | None
    guidance_scale: float
    guidance_rescale: float
    do_classifier_free_guidance: bool
    scheduler_state: Any | None = None
    noise_pred: torch.Tensor | None = None
    activations: dict[str, torch.Tensor] | None = None
    execution_order: list[str] | None = None


class StepTraceObserver:
    def __init__(self, modules: dict[str, Any], target_steps: set[int] | None = None) -> None:
        self.modules = modules
        self.target_steps = target_steps
        self._module_lookup = {id(module): path for path, module in modules.items()}
        self._handles = [module.register_forward_hook(self._hook) for module in modules.values()]
        self.current_step: int | None = None
        self._records: dict[int, StepTrace] = {}

    @property
    def records(self) -> dict[int, StepTrace]:
        return self._records

    def start_step(
        self,
        *,
        step_index: int,
        timestep: Any,
        latent_model_input: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_cond: torch.Tensor | None,
        added_cond_kwargs: dict[str, Any] | None,
        cross_attention_kwargs: dict[str, Any] | None,
        guidance_scale: float,
        guidance_rescale: float,
        do_classifier_free_guidance: bool,
        scheduler_state: Any | None = None,
    ) -> None:
        if self.target_steps is not None and step_index not in self.target_steps:
            self.current_step = None
            return
        step_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        self.current_step = step_index
        self._records[step_index] = StepTrace(
            step_index=int(step_index),
            timestep=step_value,
            latents=_cpu_clone(latents),
            latent_model_input=_cpu_clone(latent_model_input),
            prompt_embeds=_cpu_clone(prompt_embeds),
            timestep_cond=_cpu_clone(timestep_cond),
            added_cond_kwargs=_cpu_clone(added_cond_kwargs),
            cross_attention_kwargs=dict(cross_attention_kwargs) if cross_attention_kwargs is not None else None,
            guidance_scale=float(guidance_scale),
            guidance_rescale=float(guidance_rescale),
            do_classifier_free_guidance=bool(do_classifier_free_guidance),
            scheduler_state=scheduler_state,
            activations={},
            execution_order=[],
        )

    def end_step(self, *, step_index: int, timestep: Any, noise_pred: torch.Tensor) -> None:
        if self.current_step is None:
            return
        record = self._records.get(step_index)
        if record is not None:
            record.noise_pred = _cpu_clone(noise_pred)
        self.current_step = None

    def _hook(self, module: Any, _inputs: Any, output: Any) -> Any:
        if self.current_step is None:
            return output
        record = self._records.get(self.current_step)
        if record is None or record.activations is None or record.execution_order is None:
            return output
        path = self._module_lookup[id(module)]
        tensor = output[0] if isinstance(output, tuple) else output
        record.activations[path] = _cpu_clone(tensor)
        record.execution_order.append(path)
        return output

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class FullRunPatchObserver:
    def __init__(self, modules: dict[str, Any], patches_by_step: dict[int, dict[str, torch.Tensor]]) -> None:
        self.modules = modules
        self.patches_by_step = patches_by_step
        self.current_step: int | None = None
        self._module_lookup = {id(module): path for path, module in modules.items()}
        self._handles = [module.register_forward_hook(self._hook) for module in modules.values()]

    def start_step(self, *, step_index: int, **_: Any) -> None:
        self.current_step = int(step_index)

    def end_step(self, **_: Any) -> None:
        self.current_step = None

    def _hook(self, module: Any, _inputs: Any, output: Any) -> Any:
        if self.current_step is None:
            return output
        patch = self.patches_by_step.get(self.current_step, {}).get(self._module_lookup[id(module)])
        if patch is None:
            return output
        tensor = output[0] if isinstance(output, tuple) else output
        patched = patch.to(device=tensor.device, dtype=tensor.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def run_pipeline_with_observer(
    pipeline: Any,
    *,
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    latents: torch.Tensor,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    observer: Any | None,
    with_lora: bool,
    output_type: str = "latent",
) -> Any:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if with_lora:
        pipeline.enable_lora()
        pipeline.set_adapters([lora_id])
    else:
        pipeline.disable_lora()
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=denoise_steps,
        guidance_scale=cfg_scale,
        latents=latents.clone(),
        output_type=output_type,
        return_dict=False,
        cross_attention_kwargs={"scale": lora_scale},
        step_observer=observer,
    )


def sample_initial_latents(
    pipeline: Any,
    *,
    seed: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return pipeline.prepare_latents(
        1,
        pipeline.unet.config.in_channels,
        height,
        width,
        dtype,
        device,
        generator,
        None,
    )


def collect_step_pair(
    pipeline: Any,
    *,
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    latents: torch.Tensor,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    modules: dict[str, Any],
    step_index: int,
) -> tuple[StepTrace, StepTrace]:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    clean_observer = StepTraceObserver(modules, target_steps={step_index})
    corr_observer = StepTraceObserver(modules, target_steps={step_index})
    try:
        run_pipeline_with_observer(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_id=lora_id,
            latents=latents,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            observer=clean_observer,
            with_lora=True,
        )
        run_pipeline_with_observer(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_id=lora_id,
            latents=latents,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            observer=corr_observer,
            with_lora=False,
        )
    finally:
        clean_records = dict(clean_observer.records)
        corr_records = dict(corr_observer.records)
        clean_observer.close()
        corr_observer.close()

    if step_index not in clean_records or step_index not in corr_records:
        raise RuntimeError(f"Failed to capture step {step_index}")
    return clean_records[step_index], corr_records[step_index]


def run_unet_step_with_patches(
    pipeline: Any,
    *,
    step_trace: StepTrace,
    modules: dict[str, Any],
    patch_map: dict[str, torch.Tensor] | None = None,
    capture_paths: Iterable[str] = (),
    latents_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    patch_map = patch_map or {}
    capture_set = set(capture_paths)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    module_lookup = {path: module for path, module in modules.items()}
    device = pipeline._execution_device
    unet_dtype = next(pipeline.unet.parameters()).dtype

    def build_hook(path: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            final_tensor = tensor
            if path in patch_map:
                final_tensor = patch_map[path].to(device=tensor.device, dtype=tensor.dtype)
            if path in capture_set:
                captured[path] = final_tensor.detach().float().cpu()
            if path in patch_map:
                if isinstance(output, tuple):
                    return (final_tensor, *output[1:])
                return final_tensor
            return output

        return _hook

    for path in set(patch_map) | capture_set:
        if path not in module_lookup:
            continue
        handles.append(module_lookup[path].register_forward_hook(build_hook(path)))

    try:
        if latents_override is None:
            latent_model_input = _to_device(step_trace.latent_model_input, device=device, dtype=unet_dtype)
        else:
            latents = _to_device(latents_override, device=device, dtype=unet_dtype)
            if step_trace.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = pipeline.scheduler.scale_model_input(
                latent_model_input,
                torch.tensor(step_trace.timestep, device=device),
            )
        prompt_embeds = _to_device(step_trace.prompt_embeds, device=device, dtype=unet_dtype)
        timestep_cond = _to_device(step_trace.timestep_cond, device=device, dtype=unet_dtype)
        added_cond_kwargs = _to_device(step_trace.added_cond_kwargs, device=device, dtype=unet_dtype)
        raw_noise_pred = pipeline.unet(
            latent_model_input,
            torch.tensor(step_trace.timestep, device=device),
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=step_trace.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        guided = apply_classifier_free_guidance(
            raw_noise_pred,
            do_classifier_free_guidance=step_trace.do_classifier_free_guidance,
            guidance_scale=step_trace.guidance_scale,
            guidance_rescale=step_trace.guidance_rescale,
        )
        return guided.detach().float().cpu(), captured
    finally:
        for handle in handles:
            handle.remove()


def build_example_specs(prompt: str, num_examples: int, seed_start: int) -> list[dict[str, Any]]:
    return [{"prompt": prompt, "seed": seed_start + idx} for idx in range(num_examples)]


def split_examples(examples: list[dict[str, Any]], validation_split: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], []
    val_count = int(math.ceil(len(examples) * validation_split))
    val_count = min(max(val_count, 1), len(examples) - 1) if len(examples) > 1 else 0
    if val_count <= 0:
        return examples, []
    return examples[:-val_count], examples[-val_count:]


def node_key(module_path: str, step_index: int) -> str:
    return f"{module_path}@{step_index}"


def robust_node_stats(values_by_key: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, values in values_by_key.items():
        out[key] = {
            "median_score": _median(values),
            "mean_score": _tensor_mean(values),
            "variance": _variance(values),
            "num_observations": len(values),
        }
    return out


def build_lora_profile_from_positive_nodes(
    lora_id: str,
    positive_nodes: Sequence[dict[str, Any]],
    *,
    topk_modules: int = 12,
    topk_steps: int = 12,
) -> dict[str, Any]:
    module_scores: dict[str, float] = {}
    step_scores: dict[str, float] = {}
    module_step_scores: dict[str, float] = {}
    for row in positive_nodes:
        module_path = str(row["module_path"])
        step_index = int(row["step_index"])
        score = float(row["median_score"])
        module_scores[module_path] = module_scores.get(module_path, 0.0) + score
        step_scores[str(step_index)] = step_scores.get(str(step_index), 0.0) + score
        module_step_scores[node_key(module_path, step_index)] = score

    top_modules = [
        {"module_path": module_path, "score": score}
        for module_path, score in sorted(module_scores.items(), key=lambda item: item[1], reverse=True)[:topk_modules]
    ]
    top_steps = [
        {"step_index": int(step_index), "score": score}
        for step_index, score in sorted(step_scores.items(), key=lambda item: item[1], reverse=True)[:topk_steps]
    ]
    support_density = float(len(module_step_scores) / max(len(module_scores) * max(len(step_scores), 1), 1))
    return {
        "lora_id": lora_id,
        "module_scores": module_scores,
        "step_scores": step_scores,
        "module_step_scores": module_step_scores,
        "top_modules": top_modules,
        "top_steps": top_steps,
        "support_density": support_density,
        "num_positive_nodes": len(positive_nodes),
    }


def cache_validation_pairs(
    pipeline: Any,
    *,
    examples: Sequence[dict[str, Any]],
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    modules: dict[str, Any],
    step_indices: Sequence[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    device = pipeline._execution_device
    dtype = next(pipeline.unet.parameters()).dtype
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    for example_idx, example in enumerate(examples):
        latents = sample_initial_latents(
            pipeline,
            seed=int(example["seed"]),
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        for step_index in step_indices:
            clean_trace, corr_trace = collect_step_pair(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=lora_id,
                latents=latents,
                height=height,
                width=width,
                denoise_steps=denoise_steps,
                cfg_scale=cfg_scale,
                lora_scale=lora_scale,
                modules=modules,
                step_index=step_index,
            )
            direction = clean_trace.noise_pred - corr_trace.noise_pred
            cache[(example_idx, step_index)] = {
                "clean": clean_trace,
                "corr": corr_trace,
                "direction": direction,
                "direction_norm": float(torch.sum(direction.float() * direction.float()).item()),
                "seed": int(example["seed"]),
            }
    return cache


def evaluate_node_subset(
    pipeline: Any,
    *,
    modules: dict[str, Any],
    validation_cache: dict[tuple[int, int], dict[str, Any]],
    node_subset: Sequence[str],
) -> float:
    nodes_by_step: dict[int, set[str]] = {}
    for key in node_subset:
        module_path, step_part = key.rsplit("@", 1)
        nodes_by_step.setdefault(int(step_part), set()).add(module_path)

    values: list[float] = []
    for (_, step_index), payload in validation_cache.items():
        clean_trace: StepTrace = payload["clean"]
        corr_trace: StepTrace = payload["corr"]
        direction: torch.Tensor = payload["direction"]
        if payload["direction_norm"] <= EPS:
            continue
        patch_map = {
            module_path: clean_trace.activations[module_path]
            for module_path in nodes_by_step.get(step_index, set())
            if clean_trace.activations is not None and module_path in clean_trace.activations
        }
        if patch_map:
            patched_noise, _ = run_unet_step_with_patches(
                pipeline,
                step_trace=corr_trace,
                modules=modules,
                patch_map=patch_map,
            )
            values.append(directional_recovery(patched_noise, corr_trace.noise_pred, direction))
        else:
            values.append(0.0)
    return _tensor_mean(values)


def sample_random_node_controls(
    all_node_keys: Sequence[str],
    subset_size: int,
    *,
    num_trials: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    universe = list(all_node_keys)
    if subset_size <= 0 or subset_size > len(universe):
        return []
    return [rng.sample(universe, subset_size) for _ in range(num_trials)]


def rows_by_node_key(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["node"]: row for row in rows}


def group_nodes_by_step(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["step_index"]), []).append(row)
    for step_rows in grouped.values():
        step_rows.sort(key=lambda item: float(item["median_score"]), reverse=True)
    return grouped


def build_cross_step_edge_frontier(
    positive_nodes: Sequence[dict[str, Any]],
    *,
    source_topk: int,
    target_topk: int,
    max_step_delta: int,
    denoise_steps: int,
) -> list[dict[str, Any]]:
    grouped = group_nodes_by_step(positive_nodes)
    frontier: list[dict[str, Any]] = []
    for step_index in range(denoise_steps):
        source_rows = grouped.get(step_index, [])[:source_topk]
        if not source_rows:
            continue
        for step_delta in range(1, max_step_delta + 1):
            target_step = step_index + step_delta
            if target_step >= denoise_steps:
                continue
            target_rows = grouped.get(target_step, [])[:target_topk]
            if not target_rows:
                continue
            frontier.append(
                {
                    "source_step": step_index,
                    "target_step": target_step,
                    "source_paths": [row["module_path"] for row in source_rows],
                    "target_paths": [row["module_path"] for row in target_rows],
                }
            )
    return frontier


def build_same_step_diag_frontier(
    positive_nodes: Sequence[dict[str, Any]],
    *,
    topk: int,
) -> dict[int, list[str]]:
    grouped = group_nodes_by_step(positive_nodes)
    return {
        step_index: [row["module_path"] for row in rows[:topk]]
        for step_index, rows in grouped.items()
        if len(rows[:topk]) >= 2
    }


def advance_latents_with_scheduler(
    pipeline: Any,
    *,
    step_trace: StepTrace,
    noise_pred: torch.Tensor,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if step_trace.scheduler_state is None:
        raise ValueError("Step trace is missing scheduler_state required for cross-step transport.")
    scheduler = copy.deepcopy(step_trace.scheduler_state)
    device = pipeline._execution_device
    unet_dtype = next(pipeline.unet.parameters()).dtype
    latents = _to_device(step_trace.latents, device=device, dtype=unet_dtype)
    timestep = torch.tensor(step_trace.timestep, device=device)
    step_noise = _to_device(noise_pred, device=device, dtype=unet_dtype)
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, 0.0)
    next_latents = scheduler.step(
        step_noise,
        timestep,
        latents,
        **extra_step_kwargs,
        return_dict=False,
    )[0]
    return next_latents.detach().float().cpu()


def build_semantic_ablation_patches(
    validation_cache: dict[tuple[int, int], dict[str, Any]],
    *,
    example_idx: int,
    node_subset: Sequence[str],
    use_corrupted: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    patches_by_step: dict[int, dict[str, torch.Tensor]] = {}
    for key in node_subset:
        module_path, step_part = key.rsplit("@", 1)
        step_index = int(step_part)
        payload = validation_cache.get((example_idx, step_index))
        if payload is None:
            continue
        trace: StepTrace = payload["corr"] if use_corrupted else payload["clean"]
        if trace.activations is None or module_path not in trace.activations:
            continue
        patches_by_step.setdefault(step_index, {})[module_path] = trace.activations[module_path]
    return patches_by_step


def run_semantic_ablation(
    pipeline: Any,
    *,
    modules: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    latents: torch.Tensor,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    patches_by_step: dict[int, dict[str, torch.Tensor]],
) -> Any:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    observer = FullRunPatchObserver(modules, patches_by_step)
    try:
        result = run_pipeline_with_observer(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_id=lora_id,
            latents=latents,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            observer=observer,
            with_lora=True,
            output_type="pil",
        )
    finally:
        observer.close()
    return result[0][0]


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_same_step_diagnostic_edges(
    pipeline: Any,
    *,
    args: Any,
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    train_examples: Sequence[dict[str, Any]],
    modules: dict[str, Any],
    execution_orders: dict[int, list[str]],
    candidate_paths_by_step: dict[int, list[str]],
    positive_lookup: dict[str, dict[str, Any]],
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, Any]]:
    edge_scores: dict[str, list[float]] = {}
    for example in train_examples:
        latents = sample_initial_latents(
            pipeline,
            seed=int(example["seed"]),
            height=args.height,
            width=args.width,
            dtype=dtype,
            device=device,
        )
        for step_index, candidate_paths in candidate_paths_by_step.items():
            if len(candidate_paths) < 2:
                continue
            clean_trace, corr_trace = collect_step_pair(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                modules=modules,
                step_index=step_index,
            )
            if clean_trace.activations is None or corr_trace.activations is None:
                continue
            order = execution_orders.get(step_index, candidate_paths)
            order_pos = {path: idx for idx, path in enumerate(order)}
            for source_path in candidate_paths:
                for target_path in candidate_paths:
                    if source_path == target_path:
                        continue
                    if order_pos.get(source_path, 10**9) >= order_pos.get(target_path, 10**9):
                        continue
                    if source_path not in clean_trace.activations or target_path not in clean_trace.activations:
                        continue
                    corr_source = corr_trace.activations[source_path]
                    clean_source = clean_trace.activations[source_path]
                    corr_target = corr_trace.activations[target_path]
                    clean_target = clean_trace.activations[target_path]
                    delta_source = clean_source - corr_source
                    delta_target = clean_target - corr_target
                    if float(torch.sum(delta_target.float() * delta_target.float()).item()) <= EPS:
                        continue
                    per_alpha: list[float] = []
                    for alpha_idx in range(1, args.ig_steps + 1):
                        alpha = alpha_idx / args.ig_steps
                        _, captured = run_unet_step_with_patches(
                            pipeline,
                            step_trace=corr_trace,
                            modules=modules,
                            patch_map={source_path: corr_source + (alpha * delta_source)},
                            capture_paths=[target_path],
                        )
                        if target_path not in captured:
                            continue
                        recovery = activation_direction_recovery(captured[target_path], corr_target, delta_target)
                        per_alpha.append(max(recovery, 0.0))
                    if not per_alpha:
                        continue
                    edge_key = f"{source_path}@{step_index}->{target_path}@{step_index}"
                    edge_scores.setdefault(edge_key, []).append(_tensor_mean(per_alpha))

    edge_rows: list[dict[str, Any]] = []
    for edge_key, values in edge_scores.items():
        source, target = edge_key.split("->", 1)
        source_node = positive_lookup.get(source)
        target_node = positive_lookup.get(target)
        if source_node is None or target_node is None:
            continue
        median_score = _median(values)
        if median_score <= 0.0:
            continue
        edge_rows.append(
            {
                "edge": edge_key,
                "edge_type": "same_step_diag",
                "source_node": source,
                "target_node": target,
                "source_step": int(source.rsplit("@", 1)[1]),
                "target_step": int(target.rsplit("@", 1)[1]),
                "edge_score": median_score,
                "edge_mean_score": _tensor_mean(values),
                "edge_variance": _variance(values),
                "edge_recovery_fraction": _tensor_mean(values),
                "source_node_score": float(source_node["median_score"]),
                "target_node_score": float(target_node["median_score"]),
                "num_observations": len(values),
            }
        )
    edge_rows.sort(key=lambda row: row["edge_score"], reverse=True)
    return edge_rows


def score_cross_step_edges(
    pipeline: Any,
    *,
    args: Any,
    prompt: str,
    negative_prompt: str,
    lora_id: str,
    train_examples: Sequence[dict[str, Any]],
    modules: dict[str, Any],
    frontier: Sequence[dict[str, Any]],
    positive_lookup: dict[str, dict[str, Any]],
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, Any]]:
    edge_scores: dict[str, list[float]] = {}
    for example in train_examples:
        latents = sample_initial_latents(
            pipeline,
            seed=int(example["seed"]),
            height=args.height,
            width=args.width,
            dtype=dtype,
            device=device,
        )
        for spec in frontier:
            source_step = int(spec["source_step"])
            target_step = int(spec["target_step"])
            clean_source, corr_source = collect_step_pair(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                modules=modules,
                step_index=source_step,
            )
            clean_target, corr_target = collect_step_pair(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                modules=modules,
                step_index=target_step,
            )
            if (
                clean_source.activations is None
                or corr_source.activations is None
                or clean_target.activations is None
                or corr_target.activations is None
            ):
                continue
            for source_path in spec["source_paths"]:
                if source_path not in clean_source.activations or source_path not in corr_source.activations:
                    continue
                corr_source_activation = corr_source.activations[source_path]
                clean_source_activation = clean_source.activations[source_path]
                delta_source = clean_source_activation - corr_source_activation
                for target_path in spec["target_paths"]:
                    if target_path not in clean_target.activations or target_path not in corr_target.activations:
                        continue
                    corr_target_activation = corr_target.activations[target_path]
                    clean_target_activation = clean_target.activations[target_path]
                    delta_target = clean_target_activation - corr_target_activation
                    if float(torch.sum(delta_target.float() * delta_target.float()).item()) <= EPS:
                        continue
                    per_alpha: list[float] = []
                    for alpha_idx in range(1, args.ig_steps + 1):
                        alpha = alpha_idx / args.ig_steps
                        patched_noise, _ = run_unet_step_with_patches(
                            pipeline,
                            step_trace=corr_source,
                            modules=modules,
                            patch_map={source_path: corr_source_activation + (alpha * delta_source)},
                        )
                        patched_latents = advance_latents_with_scheduler(
                            pipeline,
                            step_trace=corr_source,
                            noise_pred=patched_noise,
                        )
                        _, captured = run_unet_step_with_patches(
                            pipeline,
                            step_trace=corr_target,
                            modules=modules,
                            capture_paths=[target_path],
                            latents_override=patched_latents,
                        )
                        if target_path not in captured:
                            continue
                        recovery = activation_direction_recovery(
                            captured[target_path],
                            corr_target_activation,
                            delta_target,
                        )
                        per_alpha.append(max(recovery, 0.0))
                    if not per_alpha:
                        continue
                    edge_key = f"{source_path}@{source_step}->{target_path}@{target_step}"
                    edge_scores.setdefault(edge_key, []).append(_tensor_mean(per_alpha))

    edge_rows: list[dict[str, Any]] = []
    for edge_key, values in edge_scores.items():
        source, target = edge_key.split("->", 1)
        source_node = positive_lookup.get(source)
        target_node = positive_lookup.get(target)
        if source_node is None or target_node is None:
            continue
        median_score = _median(values)
        if median_score <= 0.0:
            continue
        edge_rows.append(
            {
                "edge": edge_key,
                "edge_type": "cross_step",
                "source_node": source,
                "target_node": target,
                "source_step": int(source.rsplit("@", 1)[1]),
                "target_step": int(target.rsplit("@", 1)[1]),
                "edge_score": median_score,
                "edge_mean_score": _tensor_mean(values),
                "edge_variance": _variance(values),
                "edge_recovery_fraction": _tensor_mean(values),
                "source_node_score": float(source_node["median_score"]),
                "target_node_score": float(target_node["median_score"]),
                "num_observations": len(values),
            }
        )
    edge_rows.sort(key=lambda row: row["edge_score"], reverse=True)
    return edge_rows


def run_attribution_discovery(pipeline: Any, args: Any, out_dir: str | Path) -> dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required for circuit attribution.")
    if args.corrupted_reference != "base_model_same_prompt":
        raise ValueError("Only corrupted_reference='base_model_same_prompt' is supported in v1.")
    if args.faithfulness_target != "directional_noise_recovery":
        raise ValueError("Only faithfulness_target='directional_noise_recovery' is supported in v1.")
    if args.edge_scope not in {"cross_step_primary", "same_step_diag", "same_step_only"}:
        raise ValueError("edge_scope must be one of: cross_step_primary, same_step_diag, same_step_only.")

    prompt = args.prompt or infer_prompt_for_lora(args.image_style, args.lora_info_path, args.lora_id)
    _, negative_prompt = get_prompt(args.image_style)
    candidate_modules = discover_candidate_modules(pipeline.unet, args.candidate_node_classes)
    examples = build_example_specs(prompt, args.num_examples, args.seed_start)
    train_examples, val_examples = split_examples(examples, args.validation_split)
    if not train_examples:
        raise ValueError("Need at least one training example.")

    device = pipeline._execution_device
    dtype = next(pipeline.unet.parameters()).dtype

    node_scores: dict[str, list[float]] = {}
    execution_orders: dict[int, list[str]] = {}
    trace_manifest: list[dict[str, Any]] = []

    for example_idx, example in enumerate(train_examples):
        latents = sample_initial_latents(
            pipeline,
            seed=int(example["seed"]),
            height=args.height,
            width=args.width,
            dtype=dtype,
            device=device,
        )
        for step_index in range(args.denoise_steps):
            clean_trace, corr_trace = collect_step_pair(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=args.lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                modules=candidate_modules,
                step_index=step_index,
            )
            if clean_trace.activations is None or corr_trace.activations is None:
                continue
            direction = clean_trace.noise_pred - corr_trace.noise_pred
            direction_norm = float(torch.sum(direction.float() * direction.float()).item())
            trace_manifest.append(
                {
                    "phase": "train",
                    "example_index": example_idx,
                    "seed": int(example["seed"]),
                    "step_index": int(step_index),
                    "timestep": int(clean_trace.timestep),
                    "direction_norm": direction_norm,
                }
            )
            if direction_norm <= args.direction_norm_floor:
                continue
            if step_index not in execution_orders and clean_trace.execution_order is not None:
                execution_orders[step_index] = list(clean_trace.execution_order)
            for module_path in clean_trace.execution_order or []:
                if module_path not in corr_trace.activations or module_path not in clean_trace.activations:
                    continue
                corr_activation = corr_trace.activations[module_path]
                clean_activation = clean_trace.activations[module_path]
                delta_activation = clean_activation - corr_activation
                alpha_scores: list[float] = []
                for alpha_idx in range(1, args.ig_steps + 1):
                    alpha = alpha_idx / args.ig_steps
                    patched_noise, _ = run_unet_step_with_patches(
                        pipeline,
                        step_trace=corr_trace,
                        modules=candidate_modules,
                        patch_map={module_path: corr_activation + (alpha * delta_activation)},
                    )
                    alpha_scores.append(directional_recovery(patched_noise, corr_trace.noise_pred, direction))
                key = node_key(module_path, step_index)
                node_scores.setdefault(key, []).append(_tensor_mean(alpha_scores))

    node_stats = robust_node_stats(node_scores)
    positive_nodes = [
        {
            "node": key,
            "module_path": key.rsplit("@", 1)[0],
            "step_index": int(key.rsplit("@", 1)[1]),
            **stats,
        }
        for key, stats in node_stats.items()
        if stats["median_score"] > 0.0
    ]
    positive_nodes.sort(key=lambda row: row["median_score"], reverse=True)
    positive_lookup = rows_by_node_key(positive_nodes)

    val_steps = sorted({row["step_index"] for row in positive_nodes})
    validation_cache = cache_validation_pairs(
        pipeline,
        examples=val_examples,
        prompt=prompt,
        negative_prompt=negative_prompt,
        lora_id=args.lora_id,
        height=args.height,
        width=args.width,
        denoise_steps=args.denoise_steps,
        cfg_scale=args.cfg_scale,
        lora_scale=args.lora_scale,
        modules=candidate_modules,
        step_indices=val_steps,
    ) if val_examples and val_steps else {}

    retained_nodes = list(positive_nodes)
    node_recovery = 0.0
    if validation_cache:
        selected_keys: list[str] = []
        for row in positive_nodes:
            selected_keys.append(row["node"])
            recovery = evaluate_node_subset(
                pipeline,
                modules=candidate_modules,
                validation_cache=validation_cache,
                node_subset=selected_keys,
            )
            if recovery >= args.node_faithfulness_target:
                node_recovery = recovery
                retained_nodes = [item for item in positive_nodes if item["node"] in set(selected_keys)]
                break
        else:
            selected_keys = [row["node"] for row in positive_nodes]
            node_recovery = evaluate_node_subset(
                pipeline,
                modules=candidate_modules,
                validation_cache=validation_cache,
                node_subset=selected_keys,
            )
            retained_nodes = positive_nodes

    cross_step_frontier = build_cross_step_edge_frontier(
        positive_nodes,
        source_topk=args.edge_source_topk_per_step,
        target_topk=args.edge_target_topk_per_step,
        max_step_delta=args.max_edge_step_delta,
        denoise_steps=args.denoise_steps,
    )
    same_step_diag_frontier = build_same_step_diag_frontier(
        positive_nodes,
        topk=args.same_step_diag_topk,
    )

    cross_step_edges = score_cross_step_edges(
        pipeline,
        args=args,
        prompt=prompt,
        negative_prompt=negative_prompt,
        lora_id=args.lora_id,
        train_examples=train_examples,
        modules=candidate_modules,
        frontier=cross_step_frontier if args.edge_scope == "cross_step_primary" else [],
        positive_lookup=positive_lookup,
        dtype=dtype,
        device=device,
    )
    same_step_diag_edges = score_same_step_diagnostic_edges(
        pipeline,
        args=args,
        prompt=prompt,
        negative_prompt=negative_prompt,
        lora_id=args.lora_id,
        train_examples=train_examples,
        modules=candidate_modules,
        execution_orders=execution_orders,
        candidate_paths_by_step=same_step_diag_frontier if args.edge_scope in {"same_step_diag", "same_step_only"} else {},
        positive_lookup=positive_lookup,
        dtype=dtype,
        device=device,
    )

    edge_rows = cross_step_edges if args.edge_scope == "cross_step_primary" else []
    retained_edges = list(edge_rows)
    edge_pruned_recovery = 0.0
    baseline_node_keys = [row["node"] for row in retained_nodes]
    node_recovery_baseline = node_recovery
    if validation_cache and baseline_node_keys and node_recovery_baseline <= 0.0:
        node_recovery_baseline = evaluate_node_subset(
            pipeline,
            modules=candidate_modules,
            validation_cache=validation_cache,
            node_subset=baseline_node_keys,
        )

    if validation_cache and edge_rows and retained_nodes:
        selected_edges: list[dict[str, Any]] = []
        for row in edge_rows:
            selected_edges.append(row)
            touched_nodes = sorted({item["source_node"] for item in selected_edges} | {item["target_node"] for item in selected_edges})
            recovery = evaluate_node_subset(
                pipeline,
                modules=candidate_modules,
                    validation_cache=validation_cache,
                    node_subset=touched_nodes,
                )
            if recovery >= args.edge_faithfulness_fraction * node_recovery_baseline:
                retained_edges = list(selected_edges)
                edge_pruned_recovery = recovery
                break
        else:
            retained_edges = edge_rows
            touched_nodes = sorted({item["source_node"] for item in retained_edges} | {item["target_node"] for item in retained_edges})
            edge_pruned_recovery = evaluate_node_subset(
                pipeline,
                modules=candidate_modules,
                validation_cache=validation_cache,
                node_subset=touched_nodes,
            )

    final_node_keys = [row["node"] for row in retained_nodes]
    if retained_edges:
        final_node_keys = sorted({row["source_node"] for row in retained_edges} | {row["target_node"] for row in retained_edges})
        if validation_cache and final_node_keys:
            final_recovery = evaluate_node_subset(
                pipeline,
                modules=candidate_modules,
                validation_cache=validation_cache,
                node_subset=final_node_keys,
            )
            required_recovery = args.edge_faithfulness_fraction * node_recovery_baseline
            if final_recovery < required_recovery:
                current_keys = set(final_node_keys)
                for row in retained_nodes:
                    if row["node"] in current_keys:
                        continue
                    current_keys.add(row["node"])
                    trial_keys = sorted(current_keys)
                    final_recovery = evaluate_node_subset(
                        pipeline,
                        modules=candidate_modules,
                        validation_cache=validation_cache,
                        node_subset=trial_keys,
                    )
                    final_node_keys = trial_keys
                    if final_recovery >= required_recovery:
                        break
    final_nodes = [positive_lookup[key] for key in final_node_keys if key in positive_lookup]

    random_node_control = []
    if validation_cache and final_nodes:
        all_positive_keys = [row["node"] for row in positive_nodes]
        for sampled in sample_random_node_controls(
            all_positive_keys,
            len(final_nodes),
            num_trials=args.random_control_trials,
            seed=args.seed_start,
        ):
            random_node_control.append(
                evaluate_node_subset(
                    pipeline,
                    modules=candidate_modules,
                    validation_cache=validation_cache,
                    node_subset=sampled,
                )
            )

    random_edge_control = []
    if validation_cache and retained_edges and edge_rows:
        rng = random.Random(args.seed_start)
        for _ in range(args.random_control_trials):
            sampled_edges = rng.sample(edge_rows, min(len(retained_edges), len(edge_rows)))
            sampled_nodes = sorted({row["source_node"] for row in sampled_edges} | {row["target_node"] for row in sampled_edges})
            random_edge_control.append(
                evaluate_node_subset(
                    pipeline,
                    modules=candidate_modules,
                    validation_cache=validation_cache,
                    node_subset=sampled_nodes,
                )
            )

    cross_step_edge_recovery = 0.0
    if validation_cache and retained_edges:
        touched_nodes = sorted({row["source_node"] for row in retained_edges} | {row["target_node"] for row in retained_edges})
        cross_step_edge_recovery = evaluate_node_subset(
            pipeline,
            modules=candidate_modules,
            validation_cache=validation_cache,
            node_subset=touched_nodes,
        )

    semantic_metrics: dict[str, Any] = {}
    if args.semantic_eval and validation_cache and final_nodes:
        from sae_semantic_metrics import CLIPSemanticScorer, build_lora_semantic_spec, evaluate_ablation_semantics

        spec = build_lora_semantic_spec(args.image_style, args.lora_info_path, args.lora_id)
        scorer = CLIPSemanticScorer(args.clip_model_name, args.device)
        semantic_rows = []
        for example_idx, example in enumerate(val_examples):
            latents = sample_initial_latents(
                pipeline,
                seed=int(example["seed"]),
                height=args.height,
                width=args.width,
                dtype=dtype,
                device=device,
            )
            baseline = run_pipeline_with_observer(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=args.lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                observer=None,
                with_lora=True,
                output_type="pil",
            )[0][0]
            retained_patch = build_semantic_ablation_patches(
                validation_cache,
                example_idx=example_idx,
                node_subset=[row["node"] for row in final_nodes],
                use_corrupted=True,
            )
            ablated = run_semantic_ablation(
                pipeline,
                modules=candidate_modules,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_id=args.lora_id,
                latents=latents,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                patches_by_step=retained_patch,
            )
            semantic_rows.append(evaluate_ablation_semantics(scorer, spec, baseline, ablated))
        semantic_metrics = {
            "num_examples": len(semantic_rows),
            "rows": semantic_rows,
            "mean_clip_trigger_mean_drop": _tensor_mean([row["clip_trigger_mean_drop"] for row in semantic_rows]),
            "mean_clip_generic_drop": _tensor_mean([row["clip_generic_drop"] for row in semantic_rows]),
            "mean_clip_semantic_specificity": _tensor_mean([row["clip_semantic_specificity"] for row in semantic_rows]),
        }

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_payload = {
        "lora_id": args.lora_id,
        "prompt": prompt,
        "candidate_node_classes": list(args.candidate_node_classes),
        "faithfulness_target": args.faithfulness_target,
        "corrupted_reference": args.corrupted_reference,
        "train_examples": train_examples,
        "validation_examples": val_examples,
        "positive_nodes": positive_nodes,
        "retained_nodes": retained_nodes,
        "final_nodes": final_nodes,
        "cross_step_frontier": cross_step_frontier,
        "same_step_diag_frontier": same_step_diag_frontier,
        "execution_order": execution_orders,
    }
    profile_payload = build_lora_profile_from_positive_nodes(args.lora_id, positive_nodes)
    edge_payload = {
        "lora_id": args.lora_id,
        "retained_edges": retained_edges,
        "cross_step_edges": cross_step_edges,
        "same_step_diag_edges": same_step_diag_edges,
    }
    circuit_payload = {
        "lora_id": args.lora_id,
        "nodes": final_nodes,
        "edges": retained_edges,
        "execution_order": execution_orders,
        "faithfulness_metrics": {
            "node_recovery": node_recovery,
            "node_recovery_baseline": node_recovery_baseline,
            "edge_pruned_recovery": edge_pruned_recovery,
            "cross_step_edge_recovery": cross_step_edge_recovery,
            "random_node_control_mean": _tensor_mean(random_node_control),
            "random_edge_control_mean": _tensor_mean(random_edge_control),
            "random_node_control_rows": random_node_control,
            "random_edge_control_rows": random_edge_control,
        },
        "semantic_metrics": semantic_metrics,
        "edge_scope": args.edge_scope,
        "edge_type_counts": {
            "cross_step": len(cross_step_edges),
            "same_step_diag": len(same_step_diag_edges),
            "retained_cross_step": len(retained_edges),
        },
        "same_step_diag_summary": {
            "num_edges": len(same_step_diag_edges),
            "top_edges": same_step_diag_edges[: min(10, len(same_step_diag_edges))],
        },
        "examples_used": {
            "train": train_examples,
            "validation": val_examples,
        },
        "objective_config": {
            "type": args.faithfulness_target,
            "corrupted_reference": args.corrupted_reference,
            "ig_steps": args.ig_steps,
            "direction_norm_floor": args.direction_norm_floor,
            "edge_scope": args.edge_scope,
            "edge_source_topk_per_step": args.edge_source_topk_per_step,
            "edge_target_topk_per_step": args.edge_target_topk_per_step,
            "max_edge_step_delta": args.max_edge_step_delta,
            "same_step_diag_topk": args.same_step_diag_topk,
        },
    }
    manifest_payload = {
        "lora_id": args.lora_id,
        "examples": examples,
        "trace_rows": trace_manifest,
    }

    save_json(output_dir / "support_ap.json", support_payload)
    save_json(output_dir / "lora_profile.json", profile_payload)
    save_json(output_dir / "edge_scores_ap.json", edge_payload)
    save_json(output_dir / "circuit_ap.json", circuit_payload)
    save_json(output_dir / "trace_manifest.json", manifest_payload)

    return {
        "support": support_payload,
        "profile": profile_payload,
        "edges": edge_payload,
        "circuit": circuit_payload,
        "manifest": manifest_payload,
    }
