from __future__ import annotations

import argparse
import json
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


def load_solver_config(config_path: str | None) -> dict[str, Any]:
    return load_json_config(config_path, key="solver")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = load_solver_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Solve a timestep assignment from individual circuit supports.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--support_root", type=str, default="sae_data/individual_circuit")
    parser.add_argument("--pairwise_json", type=str, default="sae_data/pairwise_circuit_interactions.json")
    parser.add_argument("--lora_ids", type=str, required=False, default=None)
    parser.add_argument("--top_regions", type=int, default=24)
    parser.add_argument("--score_key", type=str, default="combined_score")
    parser.add_argument("--denoise_steps", type=int, default=50)

    parser.add_argument("--coverage_ratio", type=float, default=0.35)
    parser.add_argument("--conflict_weight", type=float, default=0.75)
    parser.add_argument("--undercoverage_bonus", type=float, default=0.25)

    parser.add_argument("--out_dir", type=str, default="sae_data/assignment_result")

    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_keys})

    args = parser.parse_args()
    if not args.lora_ids:
        parser.error("--lora_ids is required.")
    return args


def load_conflict_matrix(path: str | Path) -> dict[str, dict[str, float]]:
    if not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    matrix: dict[str, dict[str, float]] = {}
    for row in rows:
        a = row["lora_a"]
        b = row["lora_b"]
        score = row.get("empirical_conflict_score")
        if score is None:
            score = row.get("support_overlap_score", 0.0)
        matrix.setdefault(a, {})[b] = float(score)
        matrix.setdefault(b, {})[a] = float(score)
    return matrix


def pick_best_step_for_lora(
    *,
    lora_id: str,
    unassigned_steps: set[int],
    utility: dict[str, list[float]],
    conflict: dict[str, dict[str, float]],
) -> int | None:
    best_step = None
    best_score = float("-inf")
    for step in sorted(unassigned_steps):
        score = utility[lora_id][step]
        penalty = 0.0
        for other_id, values in utility.items():
            if other_id == lora_id:
                continue
            penalty += conflict.get(lora_id, {}).get(other_id, 0.0) * values[step]
        candidate = score - penalty
        if candidate > best_score:
            best_score = candidate
            best_step = step
    return best_step


def main() -> None:
    args = parse_args()
    lora_ids = parse_csv_str(args.lora_ids)
    support_root = Path(args.support_root)
    conflict = load_conflict_matrix(args.pairwise_json)

    utility: dict[str, list[float]] = {}
    totals: dict[str, float] = {}
    num_steps = args.denoise_steps
    for lora_id in lora_ids:
        payload, rows = load_support_rows(support_root / lora_id / "support.json", top_n=args.top_regions)
        denoise_steps = int(payload.get("denoise_steps", args.denoise_steps))
        num_steps = denoise_steps
        utility[lora_id] = aggregate_support_to_timestep_scores(
            rows,
            lora_id=lora_id,
            num_steps=denoise_steps,
            score_key=args.score_key,
        )
        totals[lora_id] = float(sum(utility[lora_id]))

    schedule: list[str | None] = [None for _ in range(num_steps)]
    assigned_utility = {lora_id: 0.0 for lora_id in lora_ids}
    required_utility = {lora_id: totals[lora_id] * args.coverage_ratio for lora_id in lora_ids}
    unassigned_steps = set(range(num_steps))

    # Coverage pass: ensure each LoRA gets some of its strongest support.
    for lora_id in sorted(lora_ids, key=lambda item: totals[item], reverse=True):
        while assigned_utility[lora_id] < required_utility[lora_id] and unassigned_steps:
            step = pick_best_step_for_lora(
                lora_id=lora_id,
                unassigned_steps=unassigned_steps,
                utility=utility,
                conflict=conflict,
            )
            if step is None:
                break
            schedule[step] = lora_id
            assigned_utility[lora_id] += utility[lora_id][step]
            unassigned_steps.remove(step)

    # Greedy pass: maximize retained utility minus conflict-weighted opportunity loss.
    for step in sorted(unassigned_steps):
        best_lora = None
        best_score = float("-inf")
        for lora_id in lora_ids:
            score = utility[lora_id][step]
            penalty = 0.0
            for other_id in lora_ids:
                if other_id == lora_id:
                    continue
                penalty += args.conflict_weight * conflict.get(lora_id, {}).get(other_id, 0.0) * utility[other_id][step]

            remaining_target = max(required_utility[lora_id] - assigned_utility[lora_id], 0.0)
            bonus = args.undercoverage_bonus * safe_div(remaining_target, totals[lora_id]) if totals[lora_id] > 0 else 0.0
            objective = score - penalty + bonus
            if objective > best_score:
                best_score = objective
                best_lora = lora_id

        assert best_lora is not None
        schedule[step] = best_lora
        assigned_utility[best_lora] += utility[best_lora][step]

    per_lora_metrics = []
    coverage_values = []
    for lora_id in lora_ids:
        retained = assigned_utility[lora_id]
        total = totals[lora_id]
        retention = safe_div(retained, total)
        coverage_values.append(retention)
        per_lora_metrics.append(
            {
                "lora_id": lora_id,
                "retained_utility": retained,
                "total_utility": total,
                "retention_ratio": retention,
                "required_utility": required_utility[lora_id],
            }
        )

    conflict_penalty = 0.0
    per_step_rows = []
    for step, owner in enumerate(schedule):
        owner = str(owner)
        step_penalty = 0.0
        scores = {lora_id: utility[lora_id][step] for lora_id in lora_ids}
        for other_id in lora_ids:
            if other_id == owner:
                continue
            step_penalty += conflict.get(owner, {}).get(other_id, 0.0) * utility[other_id][step]
        conflict_penalty += step_penalty
        per_step_rows.append(
            {
                "step": step,
                "owner": owner,
                "scores": scores,
                "conflict_penalty": step_penalty,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_slug(lora_ids)}.json"
    out_path.write_text(
        json.dumps(
            {
                "lora_ids": lora_ids,
                "schedule": schedule,
                "per_step": per_step_rows,
                "per_lora_metrics": per_lora_metrics,
                "summary": {
                    "mean_retention": float(sum(coverage_values) / len(coverage_values)),
                    "min_retention": float(min(coverage_values) if coverage_values else 0.0),
                    "coverage_ratio_target": args.coverage_ratio,
                    "conflict_penalty_total": float(conflict_penalty),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved assignment result to {out_path}")


if __name__ == "__main__":
    main()
