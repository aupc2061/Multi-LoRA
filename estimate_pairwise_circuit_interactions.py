from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from circuit_utils import (
    find_lora_entry,
    infer_prompt_for_lora,
    infer_prompt_for_loras,
    load_json_config,
    load_pipeline,
    method_slug,
    parse_csv_int,
    parse_optional_csv_str,
    rows_to_region_weight_map,
    rows_to_step_weight_map,
    safe_div,
    weighted_jaccard,
)
from circuit_utils import load_adapters, load_support_rows, run_generation
from utils import get_prompt, load_lora_info


def load_pairwise_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="pairwise")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_pairwise_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Estimate pairwise overlap and conflict between discovered LoRA circuits.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--support_root", type=str, default="sae_data/individual_circuit")
    parser.add_argument("--lora_ids", type=str, default="")
    parser.add_argument("--top_regions", type=int, default=24)

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

    parser.add_argument("--empirical_method", type=str, default="merge", choices=["merge", "composite", "switch"])
    parser.add_argument("--switch_step", type=int, default=5)
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--out_path", type=str, default="sae_data/pairwise_circuit_interactions.json")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    return parser.parse_args()


def classify_compatibility(overlap: float, empirical_conflict: float | None) -> str:
    if empirical_conflict is not None:
        if empirical_conflict >= 0.25:
            return "destructive_overlap"
        if overlap >= 0.2:
            return "shared_compatible"
        return "disjoint_support"
    if overlap >= 0.35:
        return "shared_compatible"
    if overlap >= 0.15:
        return "mild_overlap"
    return "disjoint_support"


def main() -> None:
    args = parse_args()
    seeds = parse_csv_int(args.seeds)

    if args.lora_ids.strip():
        lora_ids = parse_optional_csv_str(args.lora_ids)
    else:
        lora_info = load_lora_info(args.image_style, args.lora_info_path)
        lora_ids = [lora["id"] for group in lora_info.values() for lora in group]

    support_root = Path(args.support_root)
    support_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for lora_id in lora_ids:
        support_cache[lora_id] = load_support_rows(support_root / lora_id / "support.json", top_n=args.top_regions)

    scorer = None
    pipeline = None
    _, negative_prompt = get_prompt(args.image_style)

    rows: list[dict[str, Any]] = []
    for lora_a, lora_b in itertools.combinations(lora_ids, 2):
        _, rows_a = support_cache[lora_a]
        _, rows_b = support_cache[lora_b]

        region_overlap = weighted_jaccard(rows_to_region_weight_map(rows_a), rows_to_region_weight_map(rows_b))
        step_overlap = weighted_jaccard(rows_to_step_weight_map(rows_a), rows_to_step_weight_map(rows_b))

        empirical_conflict = None
        semantic_detail = None
        if args.semantic_eval:
            from sae_semantic_metrics import CLIPSemanticScorer, build_lora_semantic_spec

            if scorer is None:
                scorer = CLIPSemanticScorer(args.clip_model_name, args.device)
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

        if scorer is not None and pipeline is not None:
            spec_a = build_lora_semantic_spec(args.image_style, args.lora_info_path, lora_a)
            spec_b = build_lora_semantic_spec(args.image_style, args.lora_info_path, lora_b)
            ratios_a: list[float] = []
            ratios_b: list[float] = []

            for seed in seeds:
                single_a = run_generation(
                    pipeline=pipeline,
                    prompt=infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_a),
                    negative_prompt=negative_prompt,
                    lora_ids=[lora_a],
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
                single_b = run_generation(
                    pipeline=pipeline,
                    prompt=infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_b),
                    negative_prompt=negative_prompt,
                    lora_ids=[lora_b],
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
                pair_image = run_generation(
                    pipeline=pipeline,
                    prompt=infer_prompt_for_loras(args.image_style, args.lora_info_path, [lora_a, lora_b]),
                    negative_prompt=negative_prompt,
                    lora_ids=[lora_a, lora_b],
                    seed=seed,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    denoise_steps=args.denoise_steps,
                    cfg_scale=args.cfg_scale,
                    lora_scale=args.lora_scale,
                    output_type="pil",
                    method=args.empirical_method,
                    switch_step=args.switch_step,
                )[0][0]

                base_a = scorer.score_image_text([single_a] * len(spec_a.prompt_variants), spec_a.prompt_variants)
                base_b = scorer.score_image_text([single_b] * len(spec_b.prompt_variants), spec_b.prompt_variants)
                pair_a = scorer.score_image_text([pair_image] * len(spec_a.prompt_variants), spec_a.prompt_variants)
                pair_b = scorer.score_image_text([pair_image] * len(spec_b.prompt_variants), spec_b.prompt_variants)

                ratios_a.append(safe_div(sum(pair_a) / len(pair_a), sum(base_a) / len(base_a)))
                ratios_b.append(safe_div(sum(pair_b) / len(pair_b), sum(base_b) / len(base_b)))

            empirical_conflict = float(
                1.0 - ((sum(ratios_a) / len(ratios_a)) + (sum(ratios_b) / len(ratios_b))) / 2.0
            )
            semantic_detail = {
                "mean_retention_a": float(sum(ratios_a) / len(ratios_a)),
                "mean_retention_b": float(sum(ratios_b) / len(ratios_b)),
            }

        category_a, _ = find_lora_entry(args.image_style, args.lora_info_path, lora_a)
        category_b, _ = find_lora_entry(args.image_style, args.lora_info_path, lora_b)
        compatibility = classify_compatibility(region_overlap, empirical_conflict)

        rows.append(
            {
                "pair_id": method_slug([lora_a, lora_b]),
                "lora_a": lora_a,
                "lora_b": lora_b,
                "category_a": category_a,
                "category_b": category_b,
                "support_overlap_score": float(region_overlap),
                "timestep_overlap_score": float(step_overlap),
                "empirical_conflict_score": empirical_conflict,
                "compatibility": compatibility,
                "semantic_detail": semantic_detail,
                "top_regions_a": len(rows_a),
                "top_regions_b": len(rows_b),
            }
        )

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        matrix.setdefault(row["lora_a"], {})[row["lora_b"]] = row
        matrix.setdefault(row["lora_b"], {})[row["lora_a"]] = row

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "image_style": args.image_style,
                "lora_ids": lora_ids,
                "rows": rows,
                "matrix": matrix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved pairwise interactions to {out_path}")


if __name__ == "__main__":
    main()
