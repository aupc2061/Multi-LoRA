from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from circuit_utils import (
    aggregate_support_to_timestep_scores,
    load_json_config,
    load_support_rows,
    method_slug,
    parse_csv_str,
    safe_div,
)


def load_union_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="union")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_union_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Build a union circuit from individual LoRA circuit supports.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--support_root", type=str, default="sae_data/individual_circuit")
    parser.add_argument("--lora_ids", type=str, required=False, default=None)
    parser.add_argument("--top_regions", type=int, default=24)
    parser.add_argument("--score_key", type=str, default="combined_score")
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="sae_data/union_circuit")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_ids:
        parser.error("--lora_ids is required.")
    return args


def main() -> None:
    args = parse_args()
    lora_ids = parse_csv_str(args.lora_ids)
    support_root = Path(args.support_root)

    union_rows: dict[str, dict[str, Any]] = {}
    timestep_scores_by_lora: dict[str, list[float]] = {}
    total_support_score = 0.0
    num_steps = args.denoise_steps

    for lora_id in lora_ids:
        payload, rows = load_support_rows(support_root / lora_id / "support.json", top_n=args.top_regions)
        num_steps = int(payload.get("denoise_steps", args.denoise_steps))
        timestep_scores_by_lora[lora_id] = aggregate_support_to_timestep_scores(
            rows,
            lora_id=lora_id,
            num_steps=num_steps,
            score_key=args.score_key,
        )
        for row in rows:
            region_id = str(row["region_id"])
            score = float(row.get(args.score_key, 0.0))
            total_support_score += max(score, 0.0)
            if region_id not in union_rows:
                union_rows[region_id] = {
                    "region_id": region_id,
                    "hook_module": row["hook_module"],
                    "window": row["window"],
                    "step_indices": list(row["step_indices"]),
                    "total_score": 0.0,
                    "member_scores": {},
                }
            union_rows[region_id]["total_score"] += score
            union_rows[region_id]["member_scores"][lora_id] = score

    ranked_union = sorted(union_rows.values(), key=lambda row: float(row["total_score"]), reverse=True)
    retention_curve = []
    cumulative = 0.0
    for idx, row in enumerate(ranked_union, start=1):
        cumulative += max(float(row["total_score"]), 0.0)
        retention_curve.append(
            {
                "num_regions": idx,
                "region_id": row["region_id"],
                "cumulative_score": cumulative,
                "retention_fraction": safe_div(cumulative, total_support_score),
            }
        )

    assignment_schedule: list[str] = []
    per_step_owner_scores: list[dict[str, Any]] = []
    for step in range(num_steps):
        candidates = {
            lora_id: timestep_scores_by_lora.get(lora_id, [0.0] * num_steps)[step]
            for lora_id in lora_ids
        }
        owner = max(candidates, key=candidates.get)
        assignment_schedule.append(owner)
        per_step_owner_scores.append(
            {
                "step": step,
                "owner": owner,
                "scores": candidates,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_slug(lora_ids)}.json"
    out_path.write_text(
        json.dumps(
            {
                "lora_ids": lora_ids,
                "top_regions": args.top_regions,
                "score_key": args.score_key,
                "union_regions": ranked_union,
                "retention_curve": retention_curve,
                "timestep_scores_by_lora": timestep_scores_by_lora,
                "union_assignment_schedule": assignment_schedule,
                "per_step_owner_scores": per_step_owner_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved union circuit to {out_path}")


if __name__ == "__main__":
    main()
