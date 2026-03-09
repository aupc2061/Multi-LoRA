"""Generate base vs ablated images for visual inspection of top SAE features.

For each (lora_id, seed, step_mode) combination, generates:
  1. A baseline image with the LoRA active (no SAE intervention)
  2. An ablated image where the top-k delta-ranked features are zeroed mid-denoising

Images are saved side-by-side for direct visual comparison of what the top features encode.

Usage:
    python generate_ablation_images.py --config configs/sae_generate_images.json
    python generate_ablation_images.py --checkpoint ... --delta_json ... --lora_ids character_3,object_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae import SparseAutoencoder
from sae.feature_hooks import resolve_module
from sae_intervene import SAEInterventionController, parse_target_steps, pick_dtype, run_once
from utils import get_prompt, load_lora_info


# ── Config loading ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")
    return payload.get("generate", payload)


# ── Arg parsing ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    config_defaults: dict[str, Any] = {}
    if bootstrap_args.config:
        config_defaults = load_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Generate base vs ablated images for top SAE features."
    )
    parser.add_argument("--config", type=str, default=None)

    # Required experiment args
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--delta_json", type=str, default=None)
    parser.add_argument("--lora_ids", type=str, default=None, help="Comma-separated lora ids.")

    # Ablation settings
    parser.add_argument("--k", type=int, default=16, help="Number of top features to ablate.")
    parser.add_argument("--step_modes", type=str, default="last", help="Comma-separated: last|all|step indices.")
    parser.add_argument("--seeds", type=str, default="111,222")
    parser.add_argument("--apply_to", type=str, default="cond", choices=["cond", "all"])

    # Model / pipeline
    parser.add_argument("--model_name", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--hook_module", type=str, default="conv_norm_out")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument("--out_dir", type=str, default="sae_data/ablation_images")

    if config_defaults:
        valid_keys = {action.dest for action in parser._actions}
        filtered = {k: v for k, v in config_defaults.items() if k in valid_keys}
        parser.set_defaults(**filtered)

    args = parser.parse_args()

    missing = [
        name for name in ["checkpoint", "delta_json", "lora_ids"]
        if getattr(args, name) in (None, "")
    ]
    if missing:
        parser.error(
            "Missing required arguments (or config values): "
            + ", ".join(f"--{name}" for name in missing)
        )

    return args


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_csv_str(spec: str) -> list[str]:
    return [tok.strip() for tok in spec.split(",") if tok.strip()]


def parse_csv_ints(spec: str) -> list[int]:
    return [int(tok.strip()) for tok in spec.split(",") if tok.strip()]


def load_top_features(delta_json: Path, k: int) -> list[int]:
    payload = json.loads(delta_json.read_text(encoding="utf-8"))
    top = payload.get("top_features", [])
    if len(top) < k:
        raise ValueError(f"delta_json has only {len(top)} features, but k={k} requested.")
    return [int(entry["feature"]) for entry in top[:k]]


def infer_prompt_for_lora(image_style: str, lora_info_path: str, lora_id: str) -> str:
    lora_info = load_lora_info(image_style, lora_info_path)
    init_prompt, _ = get_prompt(image_style)
    for group in lora_info.values():
        for lora in group:
            if lora["id"] == lora_id:
                return init_prompt + ", " + ", ".join(lora["trigger"])
    raise ValueError(f"LoRA id not found in metadata: {lora_id}")


def make_run_args(
    base: argparse.Namespace, lora_id: str, seed: int
) -> argparse.Namespace:
    """Build the minimal args namespace that run_once expects."""
    return argparse.Namespace(
        lora_id=lora_id,
        seed=seed,
        device=base.device,
        height=base.height,
        width=base.width,
        denoise_steps=base.denoise_steps,
        cfg_scale=base.cfg_scale,
        lora_scale=base.lora_scale,
        ablate_scale=0.0,  # full suppression always
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_ids = parse_csv_str(args.lora_ids)
    step_modes = parse_csv_str(args.step_modes)
    seeds = parse_csv_ints(args.seeds)
    top_features = load_top_features(Path(args.delta_json), args.k)

    # Load SAE
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    latent_dim = int(ckpt["latent_dim"])

    sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    sae.load_state_dict(ckpt["model_state"])
    sae = sae.to(args.device).eval()

    mean = ckpt["mean"].to(args.device)
    std = ckpt["std"].to(args.device)

    # Load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_name,
        custom_pipeline=args.custom_pipeline,
        use_safetensors=True,
        torch_dtype=pick_dtype(args.dtype),
    ).to(args.device)

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # Pre-load all LoRA adapters
    lora_root = Path(args.lora_path) / args.image_style
    lora_info = load_lora_info(args.image_style, args.lora_info_path)
    all_lora_ids = [lora["id"] for group in lora_info.values() for lora in group]
    for lid in all_lora_ids:
        pipeline.load_lora_weights(
            str(lora_root), weight_name=f"{lid}.safetensors", adapter_name=lid
        )

    hook_target = resolve_module(pipeline.unet, args.hook_module)
    _, negative_prompt = get_prompt(args.image_style)

    manifest: list[dict[str, Any]] = []
    total = len(lora_ids) * len(step_modes) * len(seeds)
    done = 0

    for lora_id in lora_ids:
        prompt = infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_id)

        for step_mode in step_modes:
            target_steps = parse_target_steps(step_mode, args.denoise_steps)

            for seed in seeds:
                run_args = make_run_args(args, lora_id=lora_id, seed=seed)

                controller = SAEInterventionController(
                    module=hook_target,
                    sae=sae,
                    mean=mean,
                    std=std,
                    feature_indices=top_features,
                    apply_to=args.apply_to,
                    cfg_scale=args.cfg_scale,
                )

                # ── Baseline image (no intervention) ──
                base_result, _ = run_once(
                    pipeline=pipeline,
                    controller=controller,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    args=run_args,
                    intervene=False,
                    target_steps=target_steps,
                    output_type="pil",
                )
                base_img = base_result[0][0]

                # ── Ablated image (top features zeroed) ──
                int_result, int_records = run_once(
                    pipeline=pipeline,
                    controller=controller,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    args=run_args,
                    intervene=True,
                    target_steps=target_steps,
                    output_type="pil",
                )
                int_img = int_result[0][0]

                controller.close()

                # ── Save images ──
                tag = f"{lora_id}_step-{step_mode}_seed-{seed}"
                base_path = out_dir / f"base_{tag}.png"
                ablated_path = out_dir / f"ablated_{tag}.png"
                base_img.save(base_path)
                int_img.save(ablated_path)

                entry = {
                    "lora_id": lora_id,
                    "step_mode": step_mode,
                    "seed": seed,
                    "prompt": prompt,
                    "k": args.k,
                    "top_features": top_features,
                    "base_image": str(base_path.relative_to(out_dir)),
                    "ablated_image": str(ablated_path.relative_to(out_dir)),
                }
                manifest.append(entry)

                done += 1
                print(f"[{done}/{total}] Saved {tag}")

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} image pairs saved to {out_dir}")


if __name__ == "__main__":
    main()
