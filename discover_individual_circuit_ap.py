from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]

from circuit_attribution import run_attribution_discovery
from circuit_utils import load_adapters, load_json_config, load_pipeline, parse_csv_str


def load_discovery_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="discovery_ap")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_discovery_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Discover LoRA circuits using attribution patching.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lora_id", type=str, required=False, default=None)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--denoise_steps", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--candidate_node_classes",
        type=str,
        default="ResnetBlock2D,Transformer2DModel",
    )
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=111)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--topk_nodes_per_step", type=int, default=32)
    parser.add_argument("--faithfulness_target", type=str, default="directional_noise_recovery")
    parser.add_argument("--corrupted_reference", type=str, default="base_model_same_prompt")
    parser.add_argument("--validation_split", type=float, default=0.34)
    parser.add_argument("--node_faithfulness_target", type=float, default=0.90)
    parser.add_argument("--edge_faithfulness_fraction", type=float, default=0.95)
    parser.add_argument("--direction_norm_floor", type=float, default=1e-6)
    parser.add_argument("--edge_scope", type=str, default="same_step_only")
    parser.add_argument("--random_control_trials", type=int, default=5)

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--out_dir", type=str, default="sae_data/individual_circuit_ap")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_id:
        parser.error("--lora_id is required.")
    args.candidate_node_classes = parse_csv_str(args.candidate_node_classes)
    return args


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ImportError("torch is required to run discover_individual_circuit_ap.py")

    pipeline = load_pipeline(
        image_style=args.image_style,
        model_name=args.model_name,
        custom_pipeline=args.custom_pipeline,
        dtype=args.dtype,
        device=args.device,
    )
    load_adapters(
        pipeline,
        image_style=args.image_style,
        lora_ids=[args.lora_id],
        lora_path=args.lora_path,
    )

    out_dir = Path(args.out_dir) / args.lora_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_attribution_discovery(pipeline, args, out_dir)
    print(f"Saved attribution circuit outputs to {out_dir}")


if __name__ == "__main__":
    main()
