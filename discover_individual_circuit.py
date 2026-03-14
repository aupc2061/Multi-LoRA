from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from circuit_utils import (
    ModuleOutputController,
    build_timestep_windows,
    compute_retention_curve,
    infer_prompt_for_lora,
    load_json_config,
    load_pipeline,
    normalize_scores,
    parse_csv_str,
    region_key,
    window_label,
)
from sae.feature_hooks import resolve_module
from sae_semantic_metrics import CLIPSemanticScorer, build_lora_semantic_spec, evaluate_ablation_semantics
from utils import get_prompt


def load_discovery_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="discovery")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = load_discovery_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Discover an individual LoRA circuit proxy over module x timestep.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lora_id", type=str, required=False, default=None)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hook_modules", type=str, default="mid_block,up_blocks.2,up_blocks.3,conv_norm_out")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--window_stride", type=int, default=5)
    parser.add_argument("--output_scale", type=float, default=0.0)
    parser.add_argument("--apply_to", type=str, default="cond", choices=["cond", "all"])
    parser.add_argument("--top_regions", type=int, default=24)

    parser.add_argument("--weight_timestep", type=float, default=0.35)
    parser.add_argument("--weight_latent", type=float, default=0.35)
    parser.add_argument("--weight_semantic", type=float, default=0.30)
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--out_dir", type=str, default="sae_data/individual_circuit")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_id:
        parser.error("--lora_id is required.")
    return args


def extract_lora_importance(
    pipeline: Any,
    *,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[float]]:
    pipeline.enable_lora()
    pipeline.set_adapters([args.lora_id])

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
        cross_attention_kwargs={"scale": args.lora_scale},
        return_lora_step_importance=True,
    )
    latents, _, importance = result
    return latents[0].detach().float().cpu(), [float(value) for value in importance]


def run_module_ablation(
    pipeline: Any,
    *,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
    hook_module: str,
    step_indices: list[int],
    baseline_latents: torch.Tensor,
    baseline_image: Any | None,
    scorer: CLIPSemanticScorer | None,
    semantic_spec: Any | None,
) -> dict[str, Any]:
    target_module = resolve_module(pipeline.unet, hook_module)
    controller = ModuleOutputController(
        module=target_module,
        output_scale=args.output_scale,
        target_steps=set(step_indices),
        apply_to=args.apply_to,
        cfg_scale=args.cfg_scale,
    )
    controller.configure(enabled=True)

    try:
        pipeline.enable_lora()
        pipeline.set_adapters([args.lora_id])
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        latent_result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.denoise_steps,
            guidance_scale=args.cfg_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
            cross_attention_kwargs={"scale": args.lora_scale},
        )
        latents = latent_result[0][0].detach().float().cpu()
        latent_mse = torch.mean((baseline_latents - latents) ** 2).item()

        semantic_metrics = {}
        if scorer is not None and semantic_spec is not None and baseline_image is not None:
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            image_result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.denoise_steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                output_type="pil",
                return_dict=False,
                cross_attention_kwargs={"scale": args.lora_scale},
            )
            semantic_metrics = evaluate_ablation_semantics(
                scorer=scorer,
                spec=semantic_spec,
                base_image=baseline_image,
                edited_image=image_result[0][0],
            )

        return {
            "latent_mse": float(latent_mse),
            "semantic_metrics": semantic_metrics,
            "controller_records": list(controller.records),
        }
    finally:
        controller.close()


def main() -> None:
    args = parse_args()
    hook_modules = parse_csv_str(args.hook_modules)
    windows = build_timestep_windows(args.denoise_steps, args.window_size, args.window_stride)

    pipeline = load_pipeline(
        image_style=args.image_style,
        model_name=args.model_name,
        custom_pipeline=args.custom_pipeline,
        dtype=args.dtype,
        device=args.device,
    )

    from circuit_utils import load_adapters  # local import to keep helper import light

    load_adapters(
        pipeline,
        image_style=args.image_style,
        lora_ids=[args.lora_id],
        lora_path=args.lora_path,
    )

    prompt = infer_prompt_for_lora(args.image_style, args.lora_info_path, args.lora_id)
    _, negative_prompt = get_prompt(args.image_style)

    baseline_latents, timestep_importance = extract_lora_importance(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
    )

    baseline_image = None
    scorer = None
    semantic_spec = None
    if args.semantic_eval:
        semantic_spec = build_lora_semantic_spec(args.image_style, args.lora_info_path, args.lora_id)
        scorer = CLIPSemanticScorer(args.clip_model_name, args.device)
        pipeline.enable_lora()
        pipeline.set_adapters([args.lora_id])
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        baseline_result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.denoise_steps,
            guidance_scale=args.cfg_scale,
            generator=generator,
            output_type="pil",
            return_dict=False,
            cross_attention_kwargs={"scale": args.lora_scale},
        )
        baseline_image = baseline_result[0][0]

    rows: list[dict[str, Any]] = []
    for hook_module in hook_modules:
        for step_indices in windows:
            ablation = run_module_ablation(
                pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                args=args,
                hook_module=hook_module,
                step_indices=step_indices,
                baseline_latents=baseline_latents,
                baseline_image=baseline_image,
                scorer=scorer,
                semantic_spec=semantic_spec,
            )
            timestep_score = sum(timestep_importance[step] for step in step_indices)
            semantic_specificity = float(ablation["semantic_metrics"].get("clip_semantic_specificity", 0.0))
            rows.append(
                {
                    "lora_id": args.lora_id,
                    "hook_module": hook_module,
                    "window": window_label(step_indices),
                    "step_indices": step_indices,
                    "region_id": region_key(hook_module, step_indices[0]) if len(step_indices) == 1 else f"{hook_module}@{window_label(step_indices)}",
                    "timestep_score": float(timestep_score),
                    "timestep_score_mean": float(timestep_score / max(len(step_indices), 1)),
                    "latent_mse": float(ablation["latent_mse"]),
                    "semantic_specificity": semantic_specificity,
                    "semantic_metrics": ablation["semantic_metrics"],
                    "controller_records": ablation["controller_records"],
                }
            )

    timestep_norm = normalize_scores([row["timestep_score"] for row in rows])
    latent_norm = normalize_scores([row["latent_mse"] for row in rows])
    semantic_norm = normalize_scores([max(row["semantic_specificity"], 0.0) for row in rows])

    for idx, row in enumerate(rows):
        row["timestep_score_norm"] = float(timestep_norm[idx])
        row["latent_mse_norm"] = float(latent_norm[idx])
        row["semantic_specificity_norm"] = float(semantic_norm[idx])
        row["combined_score"] = float(
            args.weight_timestep * row["timestep_score_norm"]
            + args.weight_latent * row["latent_mse_norm"]
            + args.weight_semantic * row["semantic_specificity_norm"]
        )

    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    top_support = rows[: args.top_regions]
    retention_curve = compute_retention_curve(top_support)

    timestep_scores = [0.0 for _ in range(args.denoise_steps)]
    for row in top_support:
        for step in row["step_indices"]:
            timestep_scores[int(step)] += float(row["combined_score"])

    module_scores: dict[str, float] = {}
    for row in top_support:
        module_scores[row["hook_module"]] = module_scores.get(row["hook_module"], 0.0) + float(row["combined_score"])

    out_dir = Path(args.out_dir) / args.lora_id
    out_dir.mkdir(parents=True, exist_ok=True)

    support_payload = {
        "lora_id": args.lora_id,
        "image_style": args.image_style,
        "prompt": prompt,
        "hook_modules": hook_modules,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "denoise_steps": args.denoise_steps,
        "weights": {
            "timestep": args.weight_timestep,
            "latent": args.weight_latent,
            "semantic": args.weight_semantic,
        },
        "timestep_importance": timestep_importance,
        "top_support": top_support,
        "regions": rows,
    }
    (out_dir / "support.json").write_text(json.dumps(support_payload, indent=2), encoding="utf-8")

    scores_payload = {
        "lora_id": args.lora_id,
        "module_scores": module_scores,
        "timestep_scores": timestep_scores,
        "retention_curve": retention_curve,
    }
    (out_dir / "circuit_scores.json").write_text(json.dumps(scores_payload, indent=2), encoding="utf-8")
    (out_dir / "retention_curve.json").write_text(json.dumps(retention_curve, indent=2), encoding="utf-8")

    print(f"Saved circuit discovery outputs to {out_dir}")


if __name__ == "__main__":
    main()
