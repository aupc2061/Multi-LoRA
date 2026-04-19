from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from utils import load_lora_info


DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_METHODS = ["merge", "switch", "selective_module_step"]


def load_dotenv_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_csv_str(spec: str) -> list[str]:
    values = [token.strip() for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated string list.")
    return values


def encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based evaluation for selective LoRA mixing benchmark outputs.")
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="merge,switch,selective_module_step")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--fallback_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=1.0)
    parser.add_argument("--detail", type=str, default="auto", choices=["low", "high", "auto"])
    parser.add_argument("--eval_mode", type=str, default="ranking", choices=["ranking", "pairwise"])
    parser.add_argument("--baseline_methods", type=str, default="merge,switch")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the openai package to run LLM evaluation: pip install openai") from exc

    load_dotenv_file(args.env_file)
    api_key = os.environ.get(args.api_key_env) or os.environ.get(args.fallback_api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} or {args.fallback_api_key_env} before running evaluation.")
    return OpenAI(
        api_key=api_key,
        base_url=args.base_url
    )


def load_lora_lookup(image_style: str, lora_info_path: str) -> dict[str, dict[str, Any]]:
    payload = load_lora_info(image_style, lora_info_path)
    out: dict[str, dict[str, Any]] = {}
    for category, rows in payload.items():
        for row in rows:
            out[row["id"]] = {"category": category, **row}
    return out


def describe_loras(lora_ids: Sequence[str], lookup: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, lora_id in enumerate(lora_ids, start=1):
        row = lookup.get(lora_id, {"category": "unknown", "name": lora_id, "trigger": []})
        triggers = ", ".join(str(item) for item in row.get("trigger", []))
        lines.append(f"{index}. {row['category']} ({row.get('name', lora_id)}): {triggers}")
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def response_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return str(response.output_text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def method_image_path(pair_dir: Path, method: str, seed: int) -> Path:
    return pair_dir / f"{method}_seed{seed}.png"


def load_pair_jobs(benchmark_dir: Path, methods: Sequence[str], *, limit: int = 0) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for report_path in sorted(benchmark_dir.glob("*/report.json")):
        pair_dir = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        lora_ids = report.get("lora_ids", pair_dir.name.split("__"))
        seeds = sorted({int(row["seed"]) for result in report.get("results", []) for row in result.get("rows", [])})
        available_methods = {result["method"] for result in report.get("results", [])}
        selected_methods = [method for method in methods if method in available_methods]
        for seed in seeds:
            image_paths = {method: method_image_path(pair_dir, method, seed) for method in selected_methods}
            if all(path.exists() for path in image_paths.values()):
                jobs.append(
                    {
                        "combination_id": pair_dir.name,
                        "pair_dir": pair_dir,
                        "lora_ids": lora_ids,
                        "seed": seed,
                        "image_paths": image_paths,
                    }
                )
    return jobs[:limit] if limit > 0 else jobs


def ranking_prompt(job: dict[str, Any], lookup: dict[str, dict[str, Any]], methods: Sequence[str]) -> str:
    method_lines = "\n".join(f"- {method}: Image {idx + 1}" for idx, method in enumerate(methods))
    return f"""
You are evaluating synthetic images generated by different LoRA composition methods.
Treat the listed people/names as fictional visual descriptors. Do not evaluate real-world identity.

Target LoRA elements:
{describe_loras(job["lora_ids"], lookup)}

Images:
{method_lines}

Score every image independently on:
1. composition_quality: how well all listed LoRA elements and their trigger features appear together in one coherent image.
2. image_quality: realism, aesthetics, lack of artifacts, lack of deformations.

Use a 0 to 10 scale. Use one decimal place if useful.
Then rank the methods from best to worst for composition quality and image quality. Rank 1 is best.

Return only valid JSON with this schema:
{{
  "scores": {{
    "<method>": {{
      "composition_quality": <number>,
      "image_quality": <number>,
      "overall_quality": <number>,
      "notes": "<short reason>"
    }}
  }},
  "ranks": {{
    "composition_quality": ["<best_method>", "..."],
    "image_quality": ["<best_method>", "..."],
    "overall_quality": ["<best_method>", "..."]
  }},
  "best_method": "<method>"
}}
""".strip()


def pairwise_prompt(job: dict[str, Any], lookup: dict[str, dict[str, Any]], left: str, right: str) -> str:
    return f"""
You are evaluating two synthetic images generated by two different LoRA composition methods.
Treat the listed people/names as fictional visual descriptors. Do not evaluate real-world identity.

Target LoRA elements:
{describe_loras(job["lora_ids"], lookup)}

Image 1 method: {left}
Image 2 method: {right}

Score both images independently on:
1. composition_quality: how well all listed LoRA elements and trigger features appear together.
2. image_quality: realism, aesthetics, lack of artifacts, lack of deformations.

Use a 0 to 10 scale. Then choose the winner for each metric. Ties are allowed only if genuinely indistinguishable.

Return only valid JSON with this schema:
{{
  "scores": {{
    "{left}": {{"composition_quality": <number>, "image_quality": <number>, "overall_quality": <number>, "notes": "<short reason>"}},
    "{right}": {{"composition_quality": <number>, "image_quality": <number>, "overall_quality": <number>, "notes": "<short reason>"}}
  }},
  "winners": {{
    "composition_quality": "{left}|{right}|tie",
    "image_quality": "{left}|{right}|tie",
    "overall_quality": "{left}|{right}|tie"
  }}
}}
""".strip()


def call_responses_api(
    client: Any,
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    detail: str,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    sleep_sec: float,
) -> tuple[dict[str, Any], str]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "input_image", "image_url": encode_image_data_url(path), "detail": detail})

    last_text = ""
    for attempt in range(1, max_retries + 1):
        response = client.responses.create(
            model=model,
            instructions=(
                "You are a strict visual evaluator for text-to-image LoRA composition. "
                "Return only valid JSON. Do not include markdown."
            ),
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        last_text = response_text(response)
        try:
            return extract_json_object(last_text), last_text
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec * attempt)
    raise RuntimeError("Responses API call failed unexpectedly.")


def rank_index(ranks: Sequence[str], method: str) -> int:
    try:
        return list(ranks).index(method) + 1
    except ValueError:
        return len(ranks) + 1


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_ranking_records(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    metrics = ["composition_quality", "image_quality", "overall_quality"]
    method_summary: dict[str, Any] = {}
    for method in methods:
        method_records = [row for row in records if method in row.get("parsed", {}).get("scores", {})]
        metric_summary: dict[str, Any] = {}
        for metric in metrics:
            scores = [float(row["parsed"]["scores"][method].get(metric, 0.0)) for row in method_records]
            ranks = [rank_index(row["parsed"].get("ranks", {}).get(metric, []), method) for row in method_records]
            metric_summary[metric] = {
                "mean_score": _mean(scores),
                "mean_rank": _mean(ranks),
            }
        method_summary[method] = metric_summary

    pairwise_from_scores: dict[str, Any] = {}
    for left, right in combinations(methods, 2):
        comparison_key = f"{left}_vs_{right}"
        pairwise_from_scores[comparison_key] = {}
        for metric in metrics:
            left_wins = right_wins = ties = 0
            for row in records:
                scores = row.get("parsed", {}).get("scores", {})
                if left not in scores or right not in scores:
                    continue
                left_score = float(scores[left].get(metric, 0.0))
                right_score = float(scores[right].get(metric, 0.0))
                if left_score > right_score:
                    left_wins += 1
                elif right_score > left_score:
                    right_wins += 1
                else:
                    ties += 1
            pairwise_from_scores[comparison_key][metric] = {
                left: left_wins,
                right: right_wins,
                "ties": ties,
            }

    return {
        "num_evaluations": len(records),
        "method_summary": method_summary,
        "pairwise_from_scores": pairwise_from_scores,
    }


def _rank_methods_by_score(scores_by_method: dict[str, float]) -> list[str]:
    return [
        method
        for method, _ in sorted(
            scores_by_method.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def summarize_pair_averaged_ranking_records(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    metrics = ["composition_quality", "image_quality", "overall_quality"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["combination_id"], []).append(row)

    pair_rows: list[dict[str, Any]] = []
    for combination_id, rows in sorted(grouped.items()):
        method_scores: dict[str, dict[str, float]] = {}
        for method in methods:
            metric_scores: dict[str, float] = {}
            for metric in metrics:
                values = [
                    float(row["parsed"]["scores"][method].get(metric, 0.0))
                    for row in rows
                    if method in row.get("parsed", {}).get("scores", {})
                ]
                metric_scores[metric] = _mean(values)
            method_scores[method] = metric_scores

        ranks_by_metric = {
            metric: _rank_methods_by_score({method: method_scores[method][metric] for method in methods})
            for metric in metrics
        }
        pair_rows.append(
            {
                "combination_id": combination_id,
                "num_seed_evaluations": len(rows),
                "method_scores": method_scores,
                "ranks": ranks_by_metric,
                "best_method": ranks_by_metric["overall_quality"][0] if ranks_by_metric["overall_quality"] else "",
            }
        )

    method_summary: dict[str, Any] = {}
    for method in methods:
        method_summary[method] = {}
        for metric in metrics:
            scores = [
                float(pair["method_scores"][method][metric])
                for pair in pair_rows
                if method in pair["method_scores"]
            ]
            ranks = [
                rank_index(pair["ranks"][metric], method)
                for pair in pair_rows
                if method in pair["method_scores"]
            ]
            method_summary[method][metric] = {
                "mean_pair_average_score": _mean(scores),
                "mean_pair_rank": _mean(ranks),
            }

    pairwise_from_pair_averages: dict[str, Any] = {}
    for left, right in combinations(methods, 2):
        key = f"{left}_vs_{right}"
        pairwise_from_pair_averages[key] = {}
        for metric in metrics:
            left_wins = right_wins = ties = 0
            for pair in pair_rows:
                left_score = float(pair["method_scores"][left][metric])
                right_score = float(pair["method_scores"][right][metric])
                if left_score > right_score:
                    left_wins += 1
                elif right_score > left_score:
                    right_wins += 1
                else:
                    ties += 1
            pairwise_from_pair_averages[key][metric] = {
                left: left_wins,
                right: right_wins,
                "ties": ties,
            }

    return {
        "num_pairs": len(pair_rows),
        "method_summary": method_summary,
        "pairwise_from_pair_averages": pairwise_from_pair_averages,
        "pair_rows": pair_rows,
    }


def summarize_pairwise_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in records:
        left = row["left_method"]
        right = row["right_method"]
        key = f"{left}_vs_{right}"
        out.setdefault(key, {metric: {left: 0, right: 0, "ties": 0} for metric in ["composition_quality", "image_quality", "overall_quality"]})
        winners = row.get("parsed", {}).get("winners", {})
        for metric, winner in winners.items():
            if winner == left:
                out[key][metric][left] += 1
            elif winner == right:
                out[key][metric][right] += 1
            else:
                out[key][metric]["ties"] += 1
    return {"num_evaluations": len(records), "pairwise_wins": out}


def write_ranking_csv(records: Sequence[dict[str, Any]], methods: Sequence[str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "combination_id",
                "seed",
                "method",
                "composition_quality",
                "image_quality",
                "overall_quality",
                "composition_rank",
                "image_rank",
                "overall_rank",
            ],
        )
        writer.writeheader()
        for row in records:
            parsed = row.get("parsed", {})
            for method in methods:
                scores = parsed.get("scores", {}).get(method)
                if not scores:
                    continue
                writer.writerow(
                    {
                        "combination_id": row["combination_id"],
                        "seed": row["seed"],
                        "method": method,
                        "composition_quality": scores.get("composition_quality", ""),
                        "image_quality": scores.get("image_quality", ""),
                        "overall_quality": scores.get("overall_quality", ""),
                        "composition_rank": rank_index(parsed.get("ranks", {}).get("composition_quality", []), method),
                        "image_rank": rank_index(parsed.get("ranks", {}).get("image_quality", []), method),
                        "overall_rank": rank_index(parsed.get("ranks", {}).get("overall_quality", []), method),
                    }
                )


def write_pair_averaged_ranking_csv(pair_summary: dict[str, Any], methods: Sequence[str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "combination_id",
                "num_seed_evaluations",
                "method",
                "avg_composition_quality",
                "avg_image_quality",
                "avg_overall_quality",
                "composition_rank",
                "image_rank",
                "overall_rank",
            ],
        )
        writer.writeheader()
        for pair in pair_summary.get("pair_rows", []):
            for method in methods:
                if method not in pair["method_scores"]:
                    continue
                scores = pair["method_scores"][method]
                writer.writerow(
                    {
                        "combination_id": pair["combination_id"],
                        "num_seed_evaluations": pair["num_seed_evaluations"],
                        "method": method,
                        "avg_composition_quality": scores.get("composition_quality", ""),
                        "avg_image_quality": scores.get("image_quality", ""),
                        "avg_overall_quality": scores.get("overall_quality", ""),
                        "composition_rank": rank_index(pair["ranks"].get("composition_quality", []), method),
                        "image_rank": rank_index(pair["ranks"].get("image_quality", []), method),
                        "overall_rank": rank_index(pair["ranks"].get("overall_quality", []), method),
                    }
                )


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    out_dir = Path(args.out_dir) if args.out_dir else benchmark_dir / "llm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_csv_str(args.methods)
    baseline_methods = parse_csv_str(args.baseline_methods)
    lookup = load_lora_lookup(args.image_style, args.lora_info_path)
    jobs = load_pair_jobs(benchmark_dir, methods, limit=args.limit)
    if not jobs:
        raise ValueError(f"No evaluable jobs found in {benchmark_dir}. Did you run the benchmark with --save_images?")

    client = build_client(args)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(f"[llm-eval] mode={args.eval_mode} jobs={len(jobs)} model={args.model}", flush=True)

    if args.eval_mode == "ranking":
        for idx, job in enumerate(jobs, start=1):
            selected_methods = [method for method in methods if method in job["image_paths"]]
            prompt = ranking_prompt(job, lookup, selected_methods)
            image_paths = [job["image_paths"][method] for method in selected_methods]
            print(f"[llm-eval] [{idx}/{len(jobs)}] ranking {job['combination_id']} seed={job['seed']}", flush=True)
            try:
                parsed, raw_text = call_responses_api(
                    client,
                    model=args.model,
                    prompt=prompt,
                    image_paths=image_paths,
                    detail=args.detail,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    sleep_sec=args.sleep_sec,
                )
                records.append(
                    {
                        "combination_id": job["combination_id"],
                        "seed": job["seed"],
                        "methods": selected_methods,
                        "image_paths": {method: str(path) for method, path in job["image_paths"].items()},
                        "parsed": parsed,
                        "raw_text": raw_text,
                    }
                )
            except Exception as exc:
                failures.append({"combination_id": job["combination_id"], "seed": job["seed"], "error": repr(exc)})
            time.sleep(args.sleep_sec)
        summary = summarize_ranking_records(records, methods)
        pair_averaged_summary = summarize_pair_averaged_ranking_records(records, methods)
        summary["pair_averaged"] = pair_averaged_summary
        write_ranking_csv(records, methods, out_dir / "llm_eval_scores.csv")
        write_pair_averaged_ranking_csv(pair_averaged_summary, methods, out_dir / "llm_eval_pair_averaged_scores.csv")
    else:
        for idx, job in enumerate(jobs, start=1):
            comparisons_to_run = [(baseline, "selective_module_step") for baseline in baseline_methods if baseline in job["image_paths"] and "selective_module_step" in job["image_paths"]]
            for left, right in comparisons_to_run:
                prompt = pairwise_prompt(job, lookup, left, right)
                print(f"[llm-eval] [{idx}/{len(jobs)}] pairwise {job['combination_id']} seed={job['seed']} {left} vs {right}", flush=True)
                try:
                    parsed, raw_text = call_responses_api(
                        client,
                        model=args.model,
                        prompt=prompt,
                        image_paths=[job["image_paths"][left], job["image_paths"][right]],
                        detail=args.detail,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        sleep_sec=args.sleep_sec,
                    )
                    records.append(
                        {
                            "combination_id": job["combination_id"],
                            "seed": job["seed"],
                            "left_method": left,
                            "right_method": right,
                            "image_paths": {left: str(job["image_paths"][left]), right: str(job["image_paths"][right])},
                            "parsed": parsed,
                            "raw_text": raw_text,
                        }
                    )
                except Exception as exc:
                    failures.append({"combination_id": job["combination_id"], "seed": job["seed"], "left_method": left, "right_method": right, "error": repr(exc)})
                time.sleep(args.sleep_sec)
        summary = summarize_pairwise_records(records)

    payload = {
        "config": {
            "benchmark_dir": str(benchmark_dir),
            "model": args.model,
            "base_url": args.base_url,
            "eval_mode": args.eval_mode,
            "methods": methods,
            "baseline_methods": baseline_methods,
        },
        "summary": summary,
        "records": records,
        "failures": failures,
    }
    (out_dir / "llm_eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "llm_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[llm-eval] saved outputs to {out_dir}", flush=True)
    if failures:
        print(f"[llm-eval] failures={len(failures)}; inspect llm_eval_results.json", flush=True)


if __name__ == "__main__":
    main()
