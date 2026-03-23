from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from circuit_utils import (
    infer_prompt_for_lora,
    infer_prompt_for_loras,
    load_adapters,
    load_json_config,
    load_pipeline,
    parse_csv_int,
    parse_csv_str,
    run_generation,
)
from selective_lora import build_selective_policy, load_lora_profile, run_selective_generation
from utils import get_prompt


def parse_csv_float(spec: str) -> list[float]:
    values = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return values


def load_mixing_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="selective_mixing")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_mixing_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Evaluate conflict-aware selective LoRA mixing.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lora_ids", type=str, required=False, default=None)
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--profile_root", type=str, default="sae_data/individual_circuit_ap_crossstep")
    parser.add_argument("--methods", type=str, default="merge,timestep_only,module_only,selective_module_step")
    parser.add_argument("--support_mode", type=str, default="union", choices=["union", "intersection"])
    parser.add_argument("--top_modules_per_lora", type=int, default=6)
    parser.add_argument("--top_steps_per_lora", type=int, default=6)

    parser.add_argument("--prompt", type=str, default="")
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
    parser.add_argument("--out_dir", type=str, default="sae_data/selective_mixing")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_ids:
        parser.error("--lora_ids is required.")
    return args


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def average_prompt_score(scorer: CLIPSemanticScorer, image: Any, prompts: list[str]) -> float:
    values = scorer.score_image_text([image] * len(prompts), prompts)
    return _mean(values)


def score_generation(
    *,
    scorer: CLIPSemanticScorer,
    image: Any,
    specs: dict[str, Any],
    single_refs: dict[str, dict[int, dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    per_lora: list[dict[str, Any]] = []
    retention_values: list[float] = []
    specificity_values: list[float] = []
    image_similarity_values: list[float] = []
    for lora_id, spec in specs.items():
        trigger_score = average_prompt_score(scorer, image, spec.prompt_variants)
        generic_score = scorer.score_image_text([image], [spec.generic_prompt])[0]
        full_prompt_score = scorer.score_image_text([image], [spec.full_prompt])[0]
        ref = single_refs[lora_id][seed]
        retention = trigger_score / max(ref["trigger_score"], 1e-8)
        image_similarity = scorer.score_image_pairs([image], [ref["image"]])[0]
        specificity = trigger_score - generic_score
        retention_values.append(retention)
        specificity_values.append(specificity)
        image_similarity_values.append(image_similarity)
        per_lora.append(
            {
                "lora_id": lora_id,
                "trigger_score": trigger_score,
                "generic_score": generic_score,
                "full_prompt_score": full_prompt_score,
                "semantic_specificity": specificity,
                "retention_ratio": retention,
                "image_similarity_to_single": image_similarity,
            }
        )
    return {
        "per_lora": per_lora,
        "summary": {
            "mean_retention": _mean(retention_values),
            "min_retention": min(retention_values) if retention_values else 0.0,
            "pairwise_concept_retention": _mean(retention_values),
            "mean_semantic_specificity": _mean(specificity_values),
            "mean_image_similarity_to_single": _mean(image_similarity_values),
            "generic_quality_score": _mean([row["generic_score"] for row in per_lora]),
        },
    }


def main() -> None:
    args = parse_args()
    from sae_semantic_metrics import CLIPSemanticScorer, build_lora_semantic_spec

    lora_ids = parse_csv_str(args.lora_ids)
    methods = parse_csv_str(args.methods)
    seeds = parse_csv_int(args.seeds)
    weights = parse_csv_float(args.lora_weights) if args.lora_weights else [1.0] * len(lora_ids)
    if len(weights) != len(lora_ids):
        raise ValueError("--lora_weights must match the number of lora_ids")
    lora_weights = {lora_id: weights[idx] for idx, lora_id in enumerate(lora_ids)}

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
    profiles = {lora_id: load_lora_profile(Path(args.profile_root) / lora_id) for lora_id in lora_ids}
    policy = build_selective_policy(
        lora_ids,
        profiles,
        lora_weights=lora_weights,
        denoise_steps=args.denoise_steps,
        top_modules=args.top_modules_per_lora,
        top_steps=args.top_steps_per_lora,
        support_mode=args.support_mode,
    )

    prompt = args.prompt or infer_prompt_for_loras(args.image_style, args.lora_info_path, lora_ids)
    _, negative_prompt = get_prompt(args.image_style)

    single_refs: dict[str, dict[int, dict[str, Any]]] = {lora_id: {} for lora_id in lora_ids}
    for lora_id in lora_ids:
        single_prompt = infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_id)
        for seed in seeds:
            result = run_generation(
                pipeline=pipeline,
                prompt=single_prompt,
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
            )
            image = result[0][0]
            single_refs[lora_id][seed] = {
                "image": image,
                "trigger_score": average_prompt_score(scorer, image, specs[lora_id].prompt_variants),
            }

    out_dir = Path(args.out_dir) / "__".join(lora_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    runtime_cache: dict[str, list[float]] = {}
    for method in methods:
        rows: list[dict[str, Any]] = []
        runtimes: list[float] = []
        image_paths: list[str] = []
        for seed in seeds:
            start = time.perf_counter()
            if method == "merge":
                result = run_generation(
                    pipeline=pipeline,
                    prompt=prompt,
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
                    method="merge",
                )
                image = result[0][0]
                runtime = time.perf_counter() - start
            elif method == "timestep_only":
                schedule = [step if step else None for step in policy["timestep_schedule"]]
                result = run_generation(
                    pipeline=pipeline,
                    prompt=prompt,
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
                    assignment_schedule=schedule,
                )
                image = result[0][0]
                runtime = time.perf_counter() - start
            elif method == "module_only":
                result, runtime = run_selective_generation(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    module_assignments_by_step=policy["module_only_assignments"],
                )
                image = result[0][0]
            elif method == "selective_module_step":
                result, runtime = run_selective_generation(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    lora_ids=lora_ids,
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    module_assignments_by_step=policy["module_step_assignments"],
                )
                image = result[0][0]
            else:
                raise ValueError(f"Unsupported method: {method}")

            metrics = score_generation(
                scorer=scorer,
                image=image,
                specs=specs,
                single_refs=single_refs,
                seed=seed,
            )
            rows.append({"seed": seed, **metrics})
            runtimes.append(runtime)
            if args.save_images:
                image_path = out_dir / f"{method}_seed{seed}.png"
                image.save(image_path)
                image_paths.append(str(image_path))

        runtime_cache[method] = runtimes
        summary_rows = [row["summary"] for row in rows]
        results.append(
            {
                "method": method,
                "rows": rows,
                "summary": {
                    "mean_retention": _mean([row["mean_retention"] for row in summary_rows]),
                    "min_retention": min(row["min_retention"] for row in summary_rows),
                    "pairwise_concept_retention": _mean([row["pairwise_concept_retention"] for row in summary_rows]),
                    "mean_semantic_specificity": _mean([row["mean_semantic_specificity"] for row in summary_rows]),
                    "mean_image_similarity_to_single": _mean([row["mean_image_similarity_to_single"] for row in summary_rows]),
                    "generic_quality_score": _mean([row["generic_quality_score"] for row in summary_rows]),
                    "mean_runtime_sec": _mean(runtimes),
                },
                "image_paths": image_paths,
            }
        )

    merge_runtime = _mean(runtime_cache.get("merge", []))
    for result in results:
        mean_runtime = float(result["summary"]["mean_runtime_sec"])
        result["summary"]["runtime_overhead_vs_merge"] = (mean_runtime / merge_runtime) if merge_runtime > 0 else 0.0

    report = {
        "lora_ids": lora_ids,
        "lora_weights": lora_weights,
        "prompt": prompt,
        "methods": methods,
        "policy": {
            "support_mode": args.support_mode,
            "top_modules_per_lora": args.top_modules_per_lora,
            "top_steps_per_lora": args.top_steps_per_lora,
            "timestep_schedule": policy["timestep_schedule"],
            "module_only_assignments": policy["module_only_assignments"],
            "module_step_assignments": policy["module_step_assignments"],
        },
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved selective mixing report to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
