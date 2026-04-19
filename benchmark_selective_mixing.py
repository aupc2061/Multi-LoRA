from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from circuit_utils import (
    get_model_name,
    infer_prompt_for_loras,
    load_adapters,
    load_json_config,
    load_pipeline,
    parse_csv_int,
    parse_csv_str,
)
from mix_loras_selective import evaluate_mixing_combination, resolve_lora_weights
from utils import load_lora_info

LOGGER = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "mean_retention",
    "min_retention",
    "pairwise_concept_retention",
    "mean_image_similarity_to_single",
    "generic_quality_score",
    "mean_semantic_specificity",
    "mean_runtime_sec",
    "runtime_overhead_vs_merge",
]


def load_benchmark_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="selective_mixing_benchmark")


def parse_optional_csv_str(spec: str) -> list[str]:
    if not spec.strip():
        return []
    return parse_csv_str(spec)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_benchmark_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Benchmark selective LoRA mixing against merge/switch baselines.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--combination_size", type=int, default=2)
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_ids", type=str, default="")
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="all_pairs",
        choices=["all_pairs", "character_vs_other"],
    )
    parser.add_argument("--methods", type=str, default="merge,switch,selective_module_step")
    parser.add_argument("--include_ablations", type=str, default="")
    parser.add_argument("--profile_root", type=str, default="sae_data/individual_circuit_ap_crossstep")
    parser.add_argument(
        "--profile_mode",
        type=str,
        default="auto_profile_missing",
        choices=["profiles_only_if_present", "auto_profile_missing"],
    )
    parser.add_argument("--benchmark_seeds", type=str, default="111,222")
    parser.add_argument("--denoise_steps", type=int, default=25)
    parser.add_argument("--top_modules_per_lora", type=int, default=6)
    parser.add_argument("--top_steps_per_lora", type=int, default=6)
    parser.add_argument("--support_mode", type=str, default="union", choices=["union", "intersection"])
    parser.add_argument("--switch_step", type=int, default=5)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_per_seed_metrics", action="store_true")
    parser.add_argument("--export_eval_layout", action="store_true")
    parser.add_argument("--resume_existing_pairs", action="store_true")
    parser.add_argument("--start_at_combination", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="sae_data/selective_mixing_benchmark")

    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--lora_weights", type=str, default="")

    parser.add_argument("--attribution_num_examples", type=int, default=4)
    parser.add_argument("--attribution_seed_start", type=int, default=111)
    parser.add_argument("--attribution_ig_steps", type=int, default=4)
    parser.add_argument("--attribution_topk_nodes_per_step", type=int, default=12)
    parser.add_argument("--attribution_validation_split", type=float, default=0.25)
    parser.add_argument("--attribution_node_faithfulness_target", type=float, default=0.95)
    parser.add_argument("--attribution_edge_faithfulness_fraction", type=float, default=0.9)
    parser.add_argument("--attribution_random_control_trials", type=int, default=3)
    parser.add_argument("--attribution_edge_source_topk_per_step", type=int, default=6)
    parser.add_argument("--attribution_edge_target_topk_per_step", type=int, default=6)
    parser.add_argument("--attribution_direction_norm_floor", type=float, default=1e-5)

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if args.combination_size != 2:
        raise ValueError("v1 benchmark only supports --combination_size 2.")
    return args


def load_lora_inventory(image_style: str, lora_info_path: str) -> list[dict[str, Any]]:
    payload = load_lora_info(image_style, lora_info_path)
    entries: list[dict[str, Any]] = []
    for category, rows in payload.items():
        for row in rows:
            entries.append({"category": category, **row})
    return entries


def enumerate_benchmark_pairs(
    entries: Sequence[dict[str, Any]],
    *,
    selected_lora_ids: Sequence[str] | None = None,
    selected_categories: Sequence[str] | None = None,
    pair_mode: str = "all_pairs",
) -> list[dict[str, Any]]:
    allowed_ids = set(selected_lora_ids or [])
    allowed_categories = set(selected_categories or [])
    filtered = [
        entry
        for entry in entries
        if (not allowed_ids or entry["id"] in allowed_ids)
        and (not allowed_categories or entry["category"] in allowed_categories)
    ]
    out: list[dict[str, Any]] = []
    for left, right in combinations(sorted(filtered, key=lambda item: item["id"]), 2):
        if pair_mode == "character_vs_other":
            categories = {left["category"], right["category"]}
            if "character" not in categories or len(categories) != 2:
                continue
        lora_ids = [left["id"], right["id"]]
        categories = [left["category"], right["category"]]
        out.append(
            {
                "combination_id": "__".join(lora_ids),
                "lora_ids": lora_ids,
                "categories": categories,
                "category_pair": "+".join(sorted(categories)),
                "entries": [left, right],
            }
        )
    return out


def profile_path(profile_root: str | Path, lora_id: str) -> Path:
    return Path(profile_root) / lora_id / "lora_profile.json"


def discover_profile_status(profile_root: str | Path, lora_id: str) -> dict[str, Any]:
    path = profile_path(profile_root, lora_id)
    return {
        "lora_id": lora_id,
        "exists": path.exists(),
        "profile_path": str(path),
    }


def ensure_lora_profile(args: argparse.Namespace, lora_id: str) -> dict[str, Any]:
    status = discover_profile_status(args.profile_root, lora_id)
    if status["exists"]:
        status["generation"] = "existing"
        LOGGER.info("[profile %s] found existing profile at %s", lora_id, status["profile_path"])
        return status
    if args.profile_mode == "profiles_only_if_present":
        status["generation"] = "missing"
        status["error"] = "profile_missing"
        LOGGER.warning("[profile %s] missing profile and auto-generation disabled", lora_id)
        return status

    LOGGER.info("[profile %s] missing profile, starting attribution generation", lora_id)
    command = [
        sys.executable,
        "-m",
        "circuit_ap.discover_individual_circuit_ap",
        "--lora_id",
        lora_id,
        "--image_style",
        args.image_style,
        "--lora_info_path",
        args.lora_info_path,
        "--lora_path",
        args.lora_path,
        "--custom_pipeline",
        args.custom_pipeline,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--lora_scale",
        str(args.lora_scale),
        "--height",
        str(min(args.height, 512)),
        "--width",
        str(min(args.width, 512)),
        "--denoise_steps",
        str(min(args.denoise_steps, 10)),
        "--cfg_scale",
        str(args.cfg_scale),
        "--num_examples",
        str(args.attribution_num_examples),
        "--seed_start",
        str(args.attribution_seed_start),
        "--ig_steps",
        str(args.attribution_ig_steps),
        "--topk_nodes_per_step",
        str(args.attribution_topk_nodes_per_step),
        "--validation_split",
        str(args.attribution_validation_split),
        "--node_faithfulness_target",
        str(args.attribution_node_faithfulness_target),
        "--edge_faithfulness_fraction",
        str(args.attribution_edge_faithfulness_fraction),
        "--random_control_trials",
        str(args.attribution_random_control_trials),
        "--edge_scope",
        "cross_step_primary",
        "--edge_source_topk_per_step",
        str(args.attribution_edge_source_topk_per_step),
        "--edge_target_topk_per_step",
        str(args.attribution_edge_target_topk_per_step),
        "--max_edge_step_delta",
        "1",
        "--direction_norm_floor",
        str(args.attribution_direction_norm_floor),
        "--out_dir",
        args.profile_root,
    ]
    completed = subprocess.run(command, text=True, check=False)
    status = discover_profile_status(args.profile_root, lora_id)
    status["generation"] = "generated" if status["exists"] else "failed"
    status["returncode"] = completed.returncode
    if not status["exists"]:
        status["error"] = "profile_generation_failed"
        LOGGER.error("[profile %s] generation failed (returncode=%s)", lora_id, completed.returncode)
    else:
        LOGGER.info("[profile %s] generated profile at %s", lora_id, status["profile_path"])
    return status


def summarize_pair_deltas(method_summaries: dict[str, dict[str, float]]) -> dict[str, Any]:
    selective = method_summaries.get("selective_module_step")
    if selective is None:
        return {}
    out: dict[str, Any] = {}
    for baseline in ("merge", "switch"):
        baseline_summary = method_summaries.get(baseline)
        if baseline_summary is None:
            continue
        out[f"selective_vs_{baseline}"] = {
            metric: float(selective.get(metric, 0.0) - baseline_summary.get(metric, 0.0))
            for metric in DEFAULT_METRICS
        }
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(median(values)) if values else 0.0


def compute_win_tie_loss(
    pair_results: Sequence[dict[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str,
    tie_eps: float = 1e-8,
) -> dict[str, int]:
    wins = ties = losses = 0
    for row in pair_results:
        summaries = row.get("method_summaries", {})
        if method not in summaries or baseline not in summaries:
            continue
        delta = float(summaries[method].get(metric, 0.0)) - float(summaries[baseline].get(metric, 0.0))
        if delta > tie_eps:
            wins += 1
        elif delta < -tie_eps:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def compute_aggregate_summary(
    pair_results: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    headline_method: str = "selective_module_step",
) -> dict[str, Any]:
    evaluated = [row for row in pair_results if row.get("status") == "ok"]
    method_metrics: dict[str, Any] = {}
    for method in methods:
        summaries = [row["method_summaries"][method] for row in evaluated if method in row.get("method_summaries", {})]
        metric_summary: dict[str, Any] = {}
        for metric in DEFAULT_METRICS:
            values = [float(summary.get(metric, 0.0)) for summary in summaries]
            metric_summary[metric] = {"mean": _mean(values), "median": _median(values)}
        method_metrics[method] = metric_summary

    comparisons: dict[str, Any] = {}
    for baseline in ("merge", "switch"):
        if baseline not in methods or headline_method not in methods:
            continue
        metric_rows: dict[str, Any] = {}
        for metric in DEFAULT_METRICS:
            deltas = []
            for row in evaluated:
                summaries = row.get("method_summaries", {})
                if headline_method not in summaries or baseline not in summaries:
                    continue
                deltas.append(float(summaries[headline_method].get(metric, 0.0)) - float(summaries[baseline].get(metric, 0.0)))
            metric_rows[metric] = {
                "mean_delta": _mean(deltas),
                "median_delta": _median(deltas),
                **compute_win_tie_loss(evaluated, method=headline_method, baseline=baseline, metric=metric),
            }
        comparisons[f"{headline_method}_vs_{baseline}"] = metric_rows

    category_breakdown: dict[str, Any] = {}
    for category_pair in sorted({row.get("category_pair", "") for row in evaluated}):
        bucket = [row for row in evaluated if row.get("category_pair") == category_pair]
        category_breakdown[category_pair] = {
            method: {
                metric: _mean([float(item["method_summaries"][method].get(metric, 0.0)) for item in bucket if method in item["method_summaries"]])
                for metric in DEFAULT_METRICS
            }
            for method in methods
        }

    return {
        "num_combinations_evaluated": len(evaluated),
        "num_combinations_skipped": len(pair_results) - len(evaluated),
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "category_pair_breakdown": category_breakdown,
    }


def write_aggregate_csv(summary: dict[str, Any], out_path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for method, metrics in summary.get("method_metrics", {}).items():
        for metric, stats in metrics.items():
            rows.append(
                {
                    "section": "method",
                    "subject": method,
                    "metric": metric,
                    "mean": stats.get("mean", 0.0),
                    "median": stats.get("median", 0.0),
                    "wins": "",
                    "ties": "",
                    "losses": "",
                }
            )
    for subject, metrics in summary.get("comparisons", {}).items():
        for metric, stats in metrics.items():
            rows.append(
                {
                    "section": "comparison",
                    "subject": subject,
                    "metric": metric,
                    "mean": stats.get("mean_delta", 0.0),
                    "median": stats.get("median_delta", 0.0),
                    "wins": stats.get("wins", 0),
                    "ties": stats.get("ties", 0),
                    "losses": stats.get("losses", 0),
                }
            )
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "subject", "metric", "mean", "median", "wins", "ties", "losses"])
        writer.writeheader()
        writer.writerows(rows)


def export_eval_layout(out_root: Path, pair_result: dict[str, Any]) -> None:
    report = pair_result.get("report")
    if not report:
        return
    layout_root = out_root / "eval_layout"
    layout_root.mkdir(parents=True, exist_ok=True)
    for method_result in report.get("results", []):
        method = method_result["method"]
        for image_path in method_result.get("image_paths", []):
            src = Path(image_path)
            if not src.exists():
                continue
            dst = layout_root / f"{method}_{pair_result['combination_id']}_{src.name}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())


def pair_result_from_existing_report(combo: dict[str, Any], combo_out_dir: Path, combo_profile_status: dict[str, Any]) -> dict[str, Any]:
    report_path = combo_out_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method_summaries = {item["method"]: item["summary"] for item in report["results"]}
    return {
        **combo,
        "status": "ok",
        "profiles": combo_profile_status,
        "prompt": report.get("prompt", ""),
        "report_path": str(report_path),
        "method_summaries": method_summaries,
        "deltas": summarize_pair_deltas(method_summaries),
        "report": report,
        "resumed_from_existing_report": True,
    }


def main() -> None:
    args = parse_args()
    from sae_semantic_metrics import CLIPSemanticScorer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    methods = parse_csv_str(args.methods)
    ablations = parse_optional_csv_str(args.include_ablations)
    all_methods = methods + [method for method in ablations if method not in methods]
    seeds = parse_csv_int(args.benchmark_seeds)
    out_root = Path(args.out_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark] starting run_name={args.run_name}", flush=True)
    LOGGER.info("Starting selective mixing benchmark: run_name=%s", args.run_name)
    LOGGER.info(
        "Resolved diffusion model: %s (image_style=%s)",
        get_model_name(args.image_style, args.model_name),
        args.image_style,
    )
    print(
        f"[benchmark] resolved model={get_model_name(args.image_style, args.model_name)} style={args.image_style}",
        flush=True,
    )
    LOGGER.info("Profile mode: %s", args.profile_mode)
    LOGGER.info("Methods: %s", ", ".join(all_methods))
    LOGGER.info("Seeds: %s", ", ".join(str(seed) for seed in seeds))

    inventory = load_lora_inventory(args.image_style, args.lora_info_path)
    selected_lora_ids = parse_optional_csv_str(args.lora_ids)
    selected_categories = parse_optional_csv_str(args.categories)
    combinations_to_run = enumerate_benchmark_pairs(
        inventory,
        selected_lora_ids=selected_lora_ids or None,
        selected_categories=selected_categories or None,
        pair_mode=args.pair_mode,
    )

    if not combinations_to_run:
        raise ValueError("No combinations matched the benchmark filters.")
    if args.start_at_combination:
        start_index = next(
            (idx for idx, combo in enumerate(combinations_to_run) if combo["combination_id"] == args.start_at_combination),
            None,
        )
        if start_index is None:
            raise ValueError(f"--start_at_combination not found: {args.start_at_combination}")
        LOGGER.info("Starting at combination %s (skipping %d earlier pair(s))", args.start_at_combination, start_index)
        combinations_to_run = combinations_to_run[start_index:]
    LOGGER.info("Enumerated %d benchmark pair(s)", len(combinations_to_run))

    all_lora_ids = sorted({lora_id for combo in combinations_to_run for lora_id in combo["lora_ids"]})
    LOGGER.info("Checking profiles for %d unique LoRA(s)", len(all_lora_ids))
    print(f"[benchmark] checking profiles for {len(all_lora_ids)} LoRAs", flush=True)
    profile_status = {lora_id: ensure_lora_profile(args, lora_id) for lora_id in all_lora_ids}

    LOGGER.info("Loading diffusion pipeline")
    print("[benchmark] loading diffusion pipeline", flush=True)
    pipeline = load_pipeline(
        image_style=args.image_style,
        model_name=args.model_name,
        custom_pipeline=args.custom_pipeline,
        dtype=args.dtype,
        device=args.device,
    )
    LOGGER.info("Pipeline loaded, loading %d adapter(s)", len(all_lora_ids))
    print(f"[benchmark] pipeline loaded, loading {len(all_lora_ids)} adapters", flush=True)
    load_adapters(
        pipeline,
        image_style=args.image_style,
        lora_ids=all_lora_ids,
        lora_path=args.lora_path,
    )
    LOGGER.info("Adapters loaded")
    print("[benchmark] adapters loaded", flush=True)
    scorer = CLIPSemanticScorer(args.clip_model_name, args.device)
    LOGGER.info("CLIP scorer ready: %s", args.clip_model_name)
    print(f"[benchmark] clip scorer ready: {args.clip_model_name}", flush=True)

    pair_results: list[dict[str, Any]] = []
    per_seed_results: list[dict[str, Any]] = []
    attempted_manifest: list[dict[str, Any]] = []
    total_pairs = len(combinations_to_run)
    for pair_index, combo in enumerate(combinations_to_run, start=1):
        lora_ids = combo["lora_ids"]
        LOGGER.info("[%d/%d] Evaluating pair %s", pair_index, total_pairs, combo["combination_id"])
        print(f"[benchmark] [{pair_index}/{total_pairs}] pair={combo['combination_id']}", flush=True)
        combo_profile_status = {lora_id: profile_status[lora_id] for lora_id in lora_ids}
        attempted_manifest.append(
            {
                "combination_id": combo["combination_id"],
                "lora_ids": lora_ids,
                "categories": combo["categories"],
                "profiles": combo_profile_status,
            }
        )
        if any(not status.get("exists", False) for status in combo_profile_status.values()):
            LOGGER.warning("[%d/%d] Skipping %s due to missing profile(s)", pair_index, total_pairs, combo["combination_id"])
            print(f"[benchmark] [{pair_index}/{total_pairs}] skipped missing profile(s)", flush=True)
            pair_results.append(
                {
                    **combo,
                    "status": "skipped",
                    "skip_reason": "missing_profile",
                    "profiles": combo_profile_status,
                }
            )
            continue

        lora_weights = resolve_lora_weights(lora_ids, args.lora_weights)
        prompt = infer_prompt_for_loras(args.image_style, args.lora_info_path, lora_ids)
        combo_out_dir = out_root / combo["combination_id"]
        if args.resume_existing_pairs and (combo_out_dir / "report.json").exists():
            LOGGER.info("[%d/%d] Skipping %s; existing report found", pair_index, total_pairs, combo["combination_id"])
            print(f"[benchmark] [{pair_index}/{total_pairs}] skipped existing report", flush=True)
            pair_results.append(pair_result_from_existing_report(combo, combo_out_dir, combo_profile_status))
            continue
        try:
            report = evaluate_mixing_combination(
                pipeline=pipeline,
                scorer=scorer,
                lora_ids=lora_ids,
                lora_weights=lora_weights,
                profile_root=args.profile_root,
                methods=all_methods,
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
                out_dir=combo_out_dir,
                log_prefix=f"[{pair_index}/{total_pairs} {combo['combination_id']}] ",
            )
            (combo_out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            method_summaries = {item["method"]: item["summary"] for item in report["results"]}
            LOGGER.info("[%d/%d] Saved pair report to %s", pair_index, total_pairs, combo_out_dir / "report.json")
            result_row = {
                **combo,
                "status": "ok",
                "profiles": combo_profile_status,
                "prompt": prompt,
                "report_path": str(combo_out_dir / "report.json"),
                "method_summaries": method_summaries,
                "deltas": summarize_pair_deltas(method_summaries),
                "report": report,
            }
            pair_results.append(result_row)
            if args.save_per_seed_metrics:
                for method_result in report["results"]:
                    for seed_row in method_result.get("rows", []):
                        per_seed_results.append(
                            {
                                "combination_id": combo["combination_id"],
                                "category_pair": combo["category_pair"],
                                "method": method_result["method"],
                                **seed_row,
                            }
                        )
            if args.export_eval_layout:
                export_eval_layout(out_root, result_row)
        except Exception as exc:
            LOGGER.exception("[%d/%d] Pair %s failed", pair_index, total_pairs, combo["combination_id"])
            pair_results.append(
                {
                    **combo,
                    "status": "error",
                    "profiles": combo_profile_status,
                    "error": repr(exc),
                }
            )

    summary = compute_aggregate_summary(pair_results, methods=all_methods)
    summary["benchmark_config"] = {
        "run_name": args.run_name,
        "image_style": args.image_style,
        "combination_size": args.combination_size,
        "pair_mode": args.pair_mode,
        "resume_existing_pairs": args.resume_existing_pairs,
        "start_at_combination": args.start_at_combination,
        "methods": all_methods,
        "profile_mode": args.profile_mode,
        "benchmark_seeds": seeds,
        "denoise_steps": args.denoise_steps,
        "top_modules_per_lora": args.top_modules_per_lora,
        "top_steps_per_lora": args.top_steps_per_lora,
        "support_mode": args.support_mode,
        "switch_step": args.switch_step,
    }

    manifest = {
        "attempted_combinations": attempted_manifest,
        "profile_status": profile_status,
    }

    (out_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "pair_results.json").write_text(json.dumps(pair_results, indent=2), encoding="utf-8")
    (out_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_aggregate_csv(summary, out_root / "aggregate_summary.csv")
    if args.save_per_seed_metrics:
        (out_root / "per_seed_results.json").write_text(json.dumps(per_seed_results, indent=2), encoding="utf-8")
    LOGGER.info("Saved selective mixing benchmark to %s", out_root)
    print(f"[benchmark] saved outputs to {out_root}", flush=True)


if __name__ == "__main__":
    main()
