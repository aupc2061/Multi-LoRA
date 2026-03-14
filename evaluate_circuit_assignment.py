from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from circuit_utils import (
    build_cycle_schedule,
    infer_prompt_for_lora,
    infer_prompt_for_loras,
    load_adapters,
    load_json_config,
    load_pipeline,
    method_slug,
    parse_csv_int,
    parse_csv_str,
    run_generation,
    safe_div,
)
from sae_semantic_metrics import CLIPSemanticScorer, build_lora_semantic_spec
from utils import get_prompt


def load_eval_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="evaluation")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_eval_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Evaluate baseline and circuit-guided multi-LoRA assignments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lora_ids", type=str, required=False, default=None)
    parser.add_argument("--assignment_json", type=str, default="")
    parser.add_argument("--union_json", type=str, default="")
    parser.add_argument("--methods", type=str, default="merge,composite,switch")
    parser.add_argument("--switch_step", type=int, default=5)

    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=str, default="111")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--out_dir", type=str, default="sae_data/circuit_eval")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_ids:
        parser.error("--lora_ids is required.")
    return args


def load_schedule(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schedule = payload.get("schedule", payload.get("union_assignment_schedule"))
    if schedule is None:
        raise ValueError(f"No schedule found in {path}")
    return [str(item) if item is not None else "" for item in schedule]


def score_image_for_spec(scorer: CLIPSemanticScorer, image: Any, spec: Any) -> float:
    values = scorer.score_image_text([image] * len(spec.prompt_variants), spec.prompt_variants)
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    lora_ids = parse_csv_str(args.lora_ids)
    methods = parse_csv_str(args.methods)
    seeds = parse_csv_int(args.seeds)

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
        lora_ids=lora_ids,
        lora_path=args.lora_path,
    )

    scorer = CLIPSemanticScorer(args.clip_model_name, args.device)
    specs = {lora_id: build_lora_semantic_spec(args.image_style, args.lora_info_path, lora_id) for lora_id in lora_ids}
    _, negative_prompt = get_prompt(args.image_style)
    composition_prompt = infer_prompt_for_loras(args.image_style, args.lora_info_path, lora_ids)

    single_baselines: dict[str, list[float]] = {lora_id: [] for lora_id in lora_ids}
    for lora_id in lora_ids:
        for seed in seeds:
            image = run_generation(
                pipeline=pipeline,
                prompt=infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_id),
                negative_prompt=negative_prompt,
                lora_ids=[lora_id],
                seed=seed,
                device=args.device,
                height=args.height,
                width=args.width,
                denoise_steps=args.denoise_steps,
                cfg_scale=args.cfg_scale,
                lora_scale=args.lora_scale,
                output_type="pil",
                method="merge",
            )[0][0]
            single_baselines[lora_id].append(score_image_for_spec(scorer, image, specs[lora_id]))

    assignment_schedule = load_schedule(args.assignment_json) if args.assignment_json else None
    union_schedule = load_schedule(args.union_json) if args.union_json else None

    out_dir = Path(args.out_dir) / method_slug(lora_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for method in methods:
        per_lora_scores = {lora_id: [] for lora_id in lora_ids}
        image_paths: list[str] = []

        for seed in seeds:
            if method == "union_assignment":
                if union_schedule is None:
                    raise ValueError("--union_json is required when methods include union_assignment")
                image = run_generation(
                    pipeline=pipeline,
                    prompt=composition_prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    output_type="pil",
                    method="assignment",
                    assignment_schedule=union_schedule,
                )[0][0]
            elif method == "assignment":
                if assignment_schedule is None:
                    raise ValueError("--assignment_json is required when methods include assignment")
                image = run_generation(
                    pipeline=pipeline,
                    prompt=composition_prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    output_type="pil",
                    method="assignment",
                    assignment_schedule=assignment_schedule,
                )[0][0]
            elif method == "switch":
                image = run_generation(
                    pipeline=pipeline,
                    prompt=composition_prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    output_type="pil",
                    method="switch",
                    switch_step=args.switch_step,
                )[0][0]
            else:
                image = run_generation(
                    pipeline=pipeline,
                    prompt=composition_prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    output_type="pil",
                    method=method,
                )[0][0]

            if args.save_images:
                image_path = out_dir / f"{method}_seed{seed}.png"
                image.save(image_path)
                image_paths.append(str(image_path))

            for lora_id in lora_ids:
                per_lora_scores[lora_id].append(score_image_for_spec(scorer, image, specs[lora_id]))

        per_lora_metrics = []
        ratios = []
        for lora_id in lora_ids:
            mean_score = float(sum(per_lora_scores[lora_id]) / len(per_lora_scores[lora_id]))
            baseline = float(sum(single_baselines[lora_id]) / len(single_baselines[lora_id]))
            retention_ratio = safe_div(mean_score, baseline)
            ratios.append(retention_ratio)
            per_lora_metrics.append(
                {
                    "lora_id": lora_id,
                    "mean_score": mean_score,
                    "baseline_score": baseline,
                    "retention_ratio": retention_ratio,
                }
            )

        results.append(
            {
                "method": method,
                "per_lora_metrics": per_lora_metrics,
                "summary": {
                    "mean_retention": float(sum(ratios) / len(ratios)),
                    "min_retention": float(min(ratios)),
                    "conflict_penalty": float(sum(max(0.0, 1.0 - ratio) for ratio in ratios) / len(ratios)),
                },
                "image_paths": image_paths,
            }
        )

    report = {
        "lora_ids": lora_ids,
        "methods": methods,
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
