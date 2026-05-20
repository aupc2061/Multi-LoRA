from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from circuit_utils import load_json_config, parse_csv_int, parse_csv_str


def _parse_csv_or_list_str(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return parse_csv_str(value)


def _parse_csv_or_list_int(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return parse_csv_int(value)


@dataclass
class SD35NullMaskConfig:
    model_name: str
    backbone_family: str = "sd3.5"
    inventory_path: str = "sd35_dit_lora_info.json"
    local_root: str = "models/dit_lora/sd35"
    lora_ids: list[str] = field(default_factory=list)
    lora_weights: list[float] = field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = ""
    seeds: list[int] = field(default_factory=lambda: [111])
    denoise_steps: int = 28
    lookahead_steps: list[int] = field(default_factory=lambda: [2, 4])
    intervention_block_start: int = -1
    intervention_block_end: int = -1
    svd_rank: int = 1
    methods: list[str] = field(default_factory=lambda: ["merge", "switch", "sd35_mask_only", "sd35_mask_nullproj"])
    trigger_token_override: dict[str, str] = field(default_factory=dict)
    mask_confidence_threshold: float = 0.0
    save_attention_masks: bool = True
    save_projected_delta_stats: bool = True
    switch_step: int = 5
    dtype: str = "float16"
    device: str = "cuda"
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 7.0
    lora_scale: float = 1.0
    out_dir: str = "results/sd35_nullmask"
    dry_run: bool = False

    @classmethod
    def from_config(cls, config_path: str | None = None, key: str = "sd35_nullmask_mixing") -> "SD35NullMaskConfig":
        payload = load_json_config(config_path, key=key)
        if not payload:
            raise ValueError(f"Missing config payload for key '{key}'")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SD35NullMaskConfig":
        data = dict(payload)
        data["lora_ids"] = _parse_csv_or_list_str(data["lora_ids"])
        data["lora_weights"] = [float(value) for value in data.get("lora_weights", [])]
        data["seeds"] = _parse_csv_or_list_int(data.get("seeds", [111]))
        data["lookahead_steps"] = _parse_csv_or_list_int(data.get("lookahead_steps", [2, 4]))
        data["methods"] = _parse_csv_or_list_str(data.get("methods", []))
        cfg = cls(**data)
        if len(cfg.lora_ids) != 2:
            raise ValueError("SD3.5 null-mask v1 supports exactly 2 LoRA ids.")
        if cfg.lora_weights and len(cfg.lora_weights) != len(cfg.lora_ids):
            raise ValueError("lora_weights length must match lora_ids length.")
        if not cfg.lora_weights:
            cfg.lora_weights = [1.0 for _ in cfg.lora_ids]
        return cfg

    @property
    def pair_id(self) -> str:
        return "__".join(self.lora_ids)

    @property
    def output_root(self) -> Path:
        return Path(self.out_dir)


@dataclass
class SD35NullMaskBenchmarkConfig:
    model_name: str
    inventory_path: str = "sd35_dit_lora_info.json"
    local_root: str = "models/dit_lora/sd35"
    methods: list[str] = field(default_factory=lambda: ["merge", "switch", "sd35_mask_only", "sd35_mask_nullproj"])
    benchmark_pairs: list[list[str]] = field(default_factory=list)
    optional_pairs: list[list[str]] = field(default_factory=list)
    benchmark_seeds: list[int] = field(default_factory=lambda: [111, 222, 333])
    denoise_steps: int = 28
    lookahead_steps: list[int] = field(default_factory=lambda: [2, 4])
    svd_rank: int = 1
    switch_step: int = 5
    trigger_token_override: dict[str, str] = field(default_factory=dict)
    out_dir: str = "results/sd35_nullmask_benchmark"
    dry_run: bool = False

    @classmethod
    def from_config(cls, config_path: str | None = None, key: str = "sd35_nullmask_benchmark") -> "SD35NullMaskBenchmarkConfig":
        payload = load_json_config(config_path, key=key)
        if not payload:
            raise ValueError(f"Missing config payload for key '{key}'")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SD35NullMaskBenchmarkConfig":
        data = dict(payload)
        data["methods"] = _parse_csv_or_list_str(data.get("methods", []))
        data["benchmark_seeds"] = _parse_csv_or_list_int(data.get("benchmark_seeds", [111, 222, 333]))
        data["lookahead_steps"] = _parse_csv_or_list_int(data.get("lookahead_steps", [2, 4]))
        return cls(**data)
