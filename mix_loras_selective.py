from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

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
    parser.add_argument(
        "--methods",
        type=str,
        default="merge,switch,timestep_only,module_only,selective_module_step",
    )
    parser.add_argument("--support_mode", type=str, default="union", choices=["union", "intersection"])
    parser.add_argument("--top_modules_per_lora", type=int, default=6)
    parser.add_argument("--top_steps_per_lora", type=int, default=6)
    parser.add_argument("--switch_step", type=int, default=5)

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


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def average_prompt_score(scorer: Any, image: Any, prompts: list[str]) -> float:
    values = scorer.score_image_text([image] * len(prompts), prompts)
    return _mean(values)


def build_specs(image_style: str, lora_info_path: str, lora_ids: Sequence[str]) -> dict[str, Any]:
    from sae_semantic_metrics import build_lora_semantic_spec

    return {
        lora_id: build_lora_semantic_spec(image_style, lora_info_path, lora_id)
        for lora_id in lora_ids
    }


def resolve_lora_weights(lora_ids: Sequence[str], lora_weights_spec: str) -> dict[str, float]:
    weights = parse_csv_float(lora_weights_spec) if lora_weights_spec else [1.0] * len(lora_ids)
    if len(weights) != len(lora_ids):
        raise ValueError("--lora_weights must match the number of lora_ids")
    return {lora_id: weights[idx] for idx, lora_id in enumerate(lora_ids)}


def load_profiles(profile_root: str | Path, lora_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    root = Path(profile_root)
    return {lora_id: load_lora_profile(root / lora_id) for lora_id in lora_ids}


def build_policy(
    *,
    lora_ids: Sequence[str],
    profiles: dict[str, dict[str, Any]],
    lora_weights: dict[str, float],
    denoise_steps: int,
    top_modules_per_lora: int,
    top_steps_per_lora: int,
    support_mode: str,
) -> dict[str, Any]:
    return build_selective_policy(
        lora_ids,
        profiles,
        lora_weights=lora_weights,
        denoise_steps=denoise_steps,
        top_modules=top_modules_per_lora,
        top_steps=top_steps_per_lora,
        support_mode=support_mode,
    )


def score_generation(
    *,
    scorer: Any,
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


def build_single_refs(
    *,
    pipeline: Any,
    scorer: Any,
    lora_ids: Sequence[str],
    specs: dict[str, Any],
    seeds: Sequence[int],
    image_style: str,
    lora_info_path: str,
    negative_prompt: str,
    device: str,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
) -> dict[str, dict[int, dict[str, Any]]]:
    single_refs: dict[str, dict[int, dict[str, Any]]] = {lora_id: {} for lora_id in lora_ids}
    for lora_id in lora_ids:
        single_prompt = infer_prompt_for_lora(image_style, lora_info_path, lora_id)
        for seed in seeds:
            result = run_generation(
                pipeline=pipeline,
                prompt=single_prompt,
                negative_prompt=negative_prompt,
                lora_ids=[lora_id],
                seed=seed,
                device=device,
                height=height,
                width=width,
                denoise_steps=denoise_steps,
                cfg_scale=cfg_scale,
                lora_scale=lora_scale,
                output_type="pil",
                method="merge",
            )
            image = result[0][0]
            single_refs[lora_id][seed] = {
                "image": image,
                "trigger_score": average_prompt_score(scorer, image, specs[lora_id].prompt_variants),
            }
    return single_refs


def generate_image_for_method(
    *,
    method: str,
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
    policy: dict[str, Any],
    switch_step: int,
) -> tuple[Any, float]:
    start = time.perf_counter()
    if method == "merge":
        result = run_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            output_type="pil",
            method="merge",
        )
        return result[0][0], float(time.perf_counter() - start)
    if method == "switch":
        result = run_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            output_type="pil",
            method="switch",
            switch_step=switch_step,
        )
        return result[0][0], float(time.perf_counter() - start)
    if method == "composite":
        result = run_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            output_type="pil",
            method="composite",
        )
        return result[0][0], float(time.perf_counter() - start)
    if method == "timestep_only":
        schedule = [step if step else None for step in policy["timestep_schedule"]]
        result = run_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            output_type="pil",
            method="assignment",
            assignment_schedule=schedule,
        )
        return result[0][0], float(time.perf_counter() - start)
    if method == "module_only":
        result, runtime = run_selective_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            module_assignments_by_step=policy["module_only_assignments"],
        )
        return result[0][0], runtime
    if method == "selective_module_step":
        result, runtime = run_selective_generation(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_ids=lora_ids,
            seed=seed,
            device=device,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            cfg_scale=cfg_scale,
            lora_scale=lora_scale,
            module_assignments_by_step=policy["module_step_assignments"],
        )
        return result[0][0], runtime
    raise ValueError(f"Unsupported method: {method}")


def summarize_method_rows(rows: Sequence[dict[str, Any]], runtimes: Sequence[float]) -> dict[str, float]:
    summary_rows = [row["summary"] for row in rows]
    return {
        "mean_retention": _mean([row["mean_retention"] for row in summary_rows]),
        "min_retention": min(row["min_retention"] for row in summary_rows) if summary_rows else 0.0,
        "pairwise_concept_retention": _mean([row["pairwise_concept_retention"] for row in summary_rows]),
        "mean_semantic_specificity": _mean([row["mean_semantic_specificity"] for row in summary_rows]),
        "mean_image_similarity_to_single": _mean([row["mean_image_similarity_to_single"] for row in summary_rows]),
        "generic_quality_score": _mean([row["generic_quality_score"] for row in summary_rows]),
        "mean_runtime_sec": _mean(runtimes),
    }


def evaluate_mixing_combination(
    *,
    pipeline: Any,
    scorer: Any,
    lora_ids: Sequence[str],
    lora_weights: dict[str, float],
    profile_root: str | Path,
    methods: Sequence[str],
    support_mode: str,
    top_modules_per_lora: int,
    top_steps_per_lora: int,
    prompt: str,
    image_style: str,
    lora_info_path: str,
    device: str,
    height: int,
    width: int,
    denoise_steps: int,
    cfg_scale: float,
    lora_scale: float,
    seeds: Sequence[int],
    switch_step: int,
    save_images: bool,
    out_dir: str | Path,
) -> dict[str, Any]:
    specs = build_specs(image_style, lora_info_path, lora_ids)
    profiles = load_profiles(profile_root, lora_ids)
    policy = build_policy(
        lora_ids=lora_ids,
        profiles=profiles,
        lora_weights=lora_weights,
        denoise_steps=denoise_steps,
        top_modules_per_lora=top_modules_per_lora,
        top_steps_per_lora=top_steps_per_lora,
        support_mode=support_mode,
    )
    _, negative_prompt = get_prompt(image_style)
    single_refs = build_single_refs(
        pipeline=pipeline,
        scorer=scorer,
        lora_ids=lora_ids,
        specs=specs,
        seeds=seeds,
        image_style=image_style,
        lora_info_path=lora_info_path,
        negative_prompt=negative_prompt,
        device=device,
        height=height,
        width=width,
        denoise_steps=denoise_steps,
        cfg_scale=cfg_scale,
        lora_scale=lora_scale,
    )

    combination_out_dir = Path(out_dir)
    combination_out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    runtime_cache: dict[str, list[float]] = {}
    for method in methods:
        rows: list[dict[str, Any]] = []
        runtimes: list[float] = []
        image_paths: list[str] = []
        for seed in seeds:
            image, runtime = generate_image_for_method(
                method=method,
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_ids=lora_ids,
                seed=seed,
                device=device,
                height=height,
                width=width,
                denoise_steps=denoise_steps,
                cfg_scale=cfg_scale,
                lora_scale=lora_scale,
                policy=policy,
                switch_step=switch_step,
            )
            metrics = score_generation(
                scorer=scorer,
                image=image,
                specs=specs,
                single_refs=single_refs,
                seed=seed,
            )
            rows.append({"seed": seed, **metrics})
            runtimes.append(runtime)
            if save_images:
                image_path = combination_out_dir / f"{method}_seed{seed}.png"
                image.save(image_path)
                image_paths.append(str(image_path))

        runtime_cache[method] = runtimes
        results.append(
            {
                "method": method,
                "rows": rows,
                "summary": summarize_method_rows(rows, runtimes),
                "image_paths": image_paths,
            }
        )

    merge_runtime = _mean(runtime_cache.get("merge", []))
    for result in results:
        mean_runtime = float(result["summary"]["mean_runtime_sec"])
        result["summary"]["runtime_overhead_vs_merge"] = (mean_runtime / merge_runtime) if merge_runtime > 0 else 0.0

    return {
        "lora_ids": list(lora_ids),
        "lora_weights": dict(lora_weights),
        "prompt": prompt,
        "methods": list(methods),
        "policy": {
            "support_mode": support_mode,
            "top_modules_per_lora": top_modules_per_lora,
            "top_steps_per_lora": top_steps_per_lora,
            "switch_step": switch_step,
            "timestep_schedule": policy["timestep_schedule"],
            "module_only_assignments": policy["module_only_assignments"],
            "module_step_assignments": policy["module_step_assignments"],
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    from sae_semantic_metrics import CLIPSemanticScorer

    lora_ids = parse_csv_str(args.lora_ids)
    methods = parse_csv_str(args.methods)
    seeds = parse_csv_int(args.seeds)
    lora_weights = resolve_lora_weights(lora_ids, args.lora_weights)

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
    prompt = args.prompt or infer_prompt_for_loras(args.image_style, args.lora_info_path, lora_ids)

    out_dir = Path(args.out_dir) / "__".join(lora_ids)
    report = evaluate_mixing_combination(
        pipeline=pipeline,
        scorer=scorer,
        lora_ids=lora_ids,
        lora_weights=lora_weights,
        profile_root=args.profile_root,
        methods=methods,
        support_mode=args.support_mode,
        top_modules_per_lora=args.top_modules_per_lora,
        top_steps_per_lora=args.top_steps_per_lora,
        prompt=prompt,
        image_style=args.image_style,
        lora_info_path=args.lora_info_path,
        device=args.device,
        height=args.height,
        width=args.width,
        denoise_steps=args.denoise_steps,
        cfg_scale=args.cfg_scale,
        lora_scale=args.lora_scale,
        seeds=seeds,
        switch_step=args.switch_step,
        save_images=args.save_images,
        out_dir=out_dir,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved selective mixing report to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
