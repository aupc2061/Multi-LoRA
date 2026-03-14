from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - optional for non-generation scripts
    torch = None  # type: ignore[assignment]

from utils import get_prompt, load_lora_info


DEFAULT_MODEL_BY_STYLE = {
    "anime": "gsdf/Counterfeit-V2.5",
    "reality": "SG161222/Realistic_Vision_V5.1_noVAE",
}


def load_json_config(config_path: str | None, key: str | None = None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")
    if key is None:
        return payload
    cfg = payload.get(key, payload)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config key '{key}' must contain a JSON object.")
    return cfg


def parse_csv_str(spec: str) -> list[str]:
    values = [token.strip() for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated string list.")
    return values


def parse_csv_int(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return values


def parse_optional_csv_str(spec: str) -> list[str]:
    if not spec.strip():
        return []
    return parse_csv_str(spec)


def pick_dtype(dtype_name: str) -> torch.dtype:
    if torch is None:
        raise ImportError("torch is required for dtype selection.")
    return torch.float16 if dtype_name == "float16" else torch.float32


def get_model_name(image_style: str, model_name: str | None) -> str:
    if model_name:
        return model_name
    if image_style not in DEFAULT_MODEL_BY_STYLE:
        raise ValueError(f"Unsupported image_style: {image_style}")
    return DEFAULT_MODEL_BY_STYLE[image_style]


def load_pipeline(
    *,
    image_style: str,
    model_name: str | None,
    custom_pipeline: str,
    dtype: str,
    device: str,
) -> Any:
    if torch is None:
        raise ImportError("torch is required to load the diffusion pipeline.")
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

    pipeline = DiffusionPipeline.from_pretrained(
        get_model_name(image_style, model_name),
        custom_pipeline=custom_pipeline,
        use_safetensors=True,
        torch_dtype=pick_dtype(dtype),
    ).to(device)

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)
    return pipeline


def load_adapters(
    pipeline: Any,
    *,
    image_style: str,
    lora_ids: Sequence[str],
    lora_path: str,
) -> None:
    lora_root = Path(lora_path) / image_style
    for lora_id in lora_ids:
        pipeline.load_lora_weights(str(lora_root), weight_name=f"{lora_id}.safetensors", adapter_name=lora_id)


def find_lora_entry(image_style: str, lora_info_path: str, lora_id: str) -> tuple[str, dict[str, Any]]:
    lora_info = load_lora_info(image_style, lora_info_path)
    for category, group in lora_info.items():
        for lora in group:
            if lora["id"] == lora_id:
                return category, lora
    raise ValueError(f"LoRA id not found in metadata: {lora_id}")


def infer_prompt_for_lora(image_style: str, lora_info_path: str, lora_id: str) -> str:
    _, lora = find_lora_entry(image_style, lora_info_path, lora_id)
    init_prompt, _ = get_prompt(image_style)
    return init_prompt + ", " + ", ".join(lora["trigger"])


def infer_prompt_for_loras(image_style: str, lora_info_path: str, lora_ids: Sequence[str]) -> str:
    init_prompt, _ = get_prompt(image_style)
    triggers: list[str] = []
    for lora_id in lora_ids:
        _, lora = find_lora_entry(image_style, lora_info_path, lora_id)
        triggers.extend([str(token).strip() for token in lora.get("trigger", []) if str(token).strip()])
    return init_prompt + ", " + ", ".join(triggers)


def build_timestep_windows(num_steps: int, window_size: int, stride: int) -> list[list[int]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows: list[list[int]] = []
    start = 0
    while start < num_steps:
        end = min(num_steps, start + window_size)
        windows.append(list(range(start, end)))
        if end == num_steps:
            break
        start += stride
    return windows


def window_label(step_indices: Sequence[int]) -> str:
    if not step_indices:
        return "empty"
    return f"{step_indices[0]}-{step_indices[-1]}"


def normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-8:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def safe_div(numer: float, denom: float) -> float:
    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


def compute_retention_curve(rows: Sequence[dict[str, Any]], score_key: str = "combined_score") -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)
    total = sum(max(float(row.get(score_key, 0.0)), 0.0) for row in ranked)
    cumulative = 0.0
    curve: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        cumulative += max(float(row.get(score_key, 0.0)), 0.0)
        curve.append(
            {
                "num_regions": idx,
                "region": row["region_id"],
                "cumulative_score": cumulative,
                "retention_fraction": safe_div(cumulative, total),
            }
        )
    return curve


def region_key(module: str, step: int) -> str:
    return f"{module}@{step}"


def aggregate_support_to_timestep_scores(
    rows: Sequence[dict[str, Any]],
    *,
    lora_id: str,
    num_steps: int,
    score_key: str = "combined_score",
) -> list[float]:
    scores = [0.0 for _ in range(num_steps)]
    for row in rows:
        if row.get("lora_id") != lora_id:
            continue
        score = float(row.get(score_key, 0.0))
        for step in row.get("step_indices", []):
            if 0 <= int(step) < num_steps:
                scores[int(step)] += score
    return scores


def load_support_rows(path: str | Path, *, top_n: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("top_support", payload.get("regions", []))
    if top_n > 0:
        rows = rows[:top_n]
    return payload, rows


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    numer = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denom = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return safe_div(numer, denom)


def rows_to_region_weight_map(rows: Sequence[dict[str, Any]], score_key: str = "combined_score") -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        out[str(row["region_id"])] = float(row.get(score_key, 0.0))
    return out


def rows_to_step_weight_map(rows: Sequence[dict[str, Any]], score_key: str = "combined_score") -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for row in rows:
        score = float(row.get(score_key, 0.0))
        for step in row.get("step_indices", []):
            scores[str(step)] += score
    return dict(scores)


def method_slug(lora_ids: Sequence[str]) -> str:
    return "__".join(lora_ids)


@dataclass
class ModuleOutputController:
    module: Any
    output_scale: float
    target_steps: set[int] | None
    apply_to: str
    cfg_scale: float

    def __post_init__(self) -> None:
        self.enabled = False
        self.step_idx = 0
        self.records: list[dict[str, Any]] = []
        self._handle = self.module.register_forward_hook(self._hook)

    def configure(self, enabled: bool) -> None:
        self.enabled = enabled
        self.step_idx = 0
        self.records = []

    def close(self) -> None:
        self._handle.remove()

    def _select_batch_indices(self, bsz: int, device: torch.device) -> torch.Tensor:
        if torch is None:
            raise ImportError("torch is required for module output control.")
        if self.apply_to == "all":
            return torch.arange(bsz, device=device)
        if self.cfg_scale > 1.0 and bsz >= 2:
            return torch.arange(bsz // 2, bsz, device=device)
        return torch.arange(bsz, device=device)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        step = self.step_idx
        self.step_idx += 1

        should_apply = self.enabled and (self.target_steps is None or step in self.target_steps)
        if not should_apply:
            return output

        batch_idx = self._select_batch_indices(tensor.shape[0], tensor.device)
        if batch_idx.numel() == 0:
            return output

        tensor_out = tensor.clone()
        pre_abs = tensor[batch_idx].abs().mean().item()
        tensor_out[batch_idx] = tensor_out[batch_idx] * self.output_scale
        post_abs = tensor_out[batch_idx].abs().mean().item()
        self.records.append(
            {
                "step": int(step),
                "abs_mean_pre": float(pre_abs),
                "abs_mean_post": float(post_abs),
            }
        )

        if isinstance(output, tuple):
            return (tensor_out, *output[1:])
        return tensor_out


class StepAssignmentCallback:
    def __init__(self, schedule: Sequence[str | list[str] | None]) -> None:
        self.schedule = list(schedule)

    def __call__(self, pipeline: Any, step_index: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        next_step = step_index + 1
        if next_step >= len(self.schedule):
            return {}
        target = self.schedule[next_step]
        if target is None or target == "":
            pipeline.disable_lora()
            return {}
        pipeline.enable_lora()
        pipeline.set_adapters(target)
        return {}


def build_cycle_schedule(lora_ids: Sequence[str], num_steps: int, switch_step: int) -> list[str]:
    if not lora_ids:
        raise ValueError("lora_ids cannot be empty")
    if switch_step <= 0:
        raise ValueError("switch_step must be positive")
    schedule: list[str] = []
    idx = 0
    for step in range(num_steps):
        schedule.append(lora_ids[idx])
        if (step + 1) % switch_step == 0:
            idx = (idx + 1) % len(lora_ids)
    return schedule


def run_generation(
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
    output_type: str,
    method: str,
    assignment_schedule: Sequence[str | list[str] | None] | None = None,
    switch_step: int = 5,
    lora_timestep_mask: Sequence[bool] | None = None,
    spatial_mask: torch.Tensor | list[torch.Tensor] | None = None,
    spatial_adapters: Sequence[str] | None = None,
) -> Any:
    if torch is None:
        raise ImportError("torch is required for image generation.")
    if not lora_ids:
        raise ValueError("lora_ids cannot be empty")

    callback = None
    callback_inputs = None
    extra_kwargs: dict[str, Any] = {}
    pipeline.enable_lora()

    if method == "merge":
        pipeline.set_adapters(list(lora_ids))
    elif method == "composite":
        pipeline.set_adapters(list(lora_ids))
        extra_kwargs["lora_composite"] = True
    elif method == "switch":
        schedule = build_cycle_schedule(lora_ids, denoise_steps, switch_step)
        pipeline.set_adapters(schedule[0])
        callback = StepAssignmentCallback(schedule)
        callback_inputs = ["latents"]
    elif method == "assignment":
        if assignment_schedule is None:
            raise ValueError("assignment_schedule is required for method='assignment'")
        if len(assignment_schedule) != denoise_steps:
            raise ValueError("assignment_schedule length must match denoise_steps")
        first = assignment_schedule[0]
        if first is None or first == "":
            pipeline.disable_lora()
        else:
            pipeline.set_adapters(first)
        callback = StepAssignmentCallback(assignment_schedule)
        callback_inputs = ["latents"]
    elif method == "timestep_mask":
        pipeline.set_adapters(list(lora_ids))
        extra_kwargs["lora_timestep_mask"] = list(lora_timestep_mask or [])
    elif method == "spatial":
        if spatial_mask is None or spatial_adapters is None:
            raise ValueError("spatial_mask and spatial_adapters are required for method='spatial'")
        pipeline.set_adapters([spatial_adapters[0]])
        extra_kwargs["lora_timestep_spatial_mask"] = spatial_mask
        extra_kwargs["lora_timestep_spatial_adapters"] = list(spatial_adapters)
    else:
        raise ValueError(f"Unknown generation method: {method}")

    generator = torch.Generator(device=device).manual_seed(seed)
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=denoise_steps,
        guidance_scale=cfg_scale,
        generator=generator,
        output_type=output_type,
        return_dict=False,
        cross_attention_kwargs={"scale": lora_scale},
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=callback_inputs,
        **extra_kwargs,
    )
