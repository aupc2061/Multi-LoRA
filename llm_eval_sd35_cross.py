"""
LLM-as-judge evaluation for SD3.5 cross-directory LoRA composition comparison.

Evaluates four methods where images come from two separate result directories:
  merge, switch          → --baseline_dir  (results/sd35_charxstyle_sweep)
  sd35_mask_only,
  sd35_mask_nullproj     → --mask_dir       (results/sd35_charxstyle_sweep_1)

Usage:
    python llm_eval_sd35_cross.py \\
        --baseline_dir results/sd35_charxstyle_sweep \\
        --mask_dir     results/sd35_charxstyle_sweep_1 \\
        --lora_info_path sd35_dit_lora_info.json \\
        --model openai/gpt-4.5-preview
"""
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
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────────

BASELINE_METHODS = ["merge", "switch"]
MASK_METHODS     = ["sd35_mask_only", "sd35_mask_nullproj"]
ALL_METHODS      = BASELINE_METHODS + MASK_METHODS

DEFAULT_MODEL    = "gpt-5.5"
DEFAULT_BASE_URL = "https://pluto-prod-hawang-llm-proxy-2026-0-0-4000.or2.colligo.dev"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline_dir",    required=True,  help="Dir with merge/switch results (sd35_charxstyle_sweep)")
    p.add_argument("--mask_dir",        required=True,  help="Dir with soft-mask results  (sd35_charxstyle_sweep_1)")
    p.add_argument("--lora_info_path",  default="sd35_dit_lora_info.json")
    p.add_argument("--model",           default=DEFAULT_MODEL)
    p.add_argument("--base_url",        default=DEFAULT_BASE_URL)
    p.add_argument("--api_key_env",     default="OPENROUTER_API_KEY")
    p.add_argument("--fallback_api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--env_file",        default=".env")
    p.add_argument("--out_dir",         default="",     help="Output dir; defaults to <mask_dir>/llm_eval")
    p.add_argument("--methods",         default=",".join(ALL_METHODS),
                   help="Comma-separated methods to evaluate (default: all four)")
    p.add_argument("--max_output_tokens", type=int, default=2048)
    p.add_argument("--temperature",     type=float, default=0.0)
    p.add_argument("--max_retries",     type=int,   default=3)
    p.add_argument("--sleep_sec",       type=float, default=1.5)
    p.add_argument("--detail",          default="auto", choices=["low", "high", "auto"])
    p.add_argument("--limit",           type=int,   default=0, help="Max jobs (0 = all)")
    return p.parse_args()


def build_client(args: argparse.Namespace) -> Any:
    load_dotenv(args.env_file)
    api_key = os.environ.get(args.api_key_env) or os.environ.get(args.fallback_api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} or {args.fallback_api_key_env} in environment or .env")
    return OpenAI(api_key=api_key, base_url=args.base_url)


# ── adapter info ───────────────────────────────────────────────────────────────

def load_lora_lookup(path: str) -> dict[str, dict[str, Any]]:
    """Load sd35_dit_lora_info.json → flat dict keyed by adapter id."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, list):
            # Nested {category: [{id, ...}, ...]}
            for category, rows in raw.items():
                for row in rows:
                    if isinstance(row, dict):
                        key = row.get("id") or row.get("adapter_id")
                        if key:
                            out[key] = {"category": category, **row}
        else:
            # Flat {id: info}
            out = {k: v for k, v in raw.items() if isinstance(v, dict)}
    return out


def describe_loras(lora_ids: Sequence[str], lookup: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, lid in enumerate(lora_ids, 1):
        row = lookup.get(lid, {})
        category = row.get("category", "unknown")
        name     = row.get("name", lid)
        trigger  = row.get("trigger", "")
        if isinstance(trigger, list):
            trigger = ", ".join(str(t) for t in trigger) if trigger else "none"
        lines.append(f"{i}. {category} — {name}  (trigger: {trigger or 'none'})")
    return "\n".join(lines)


# ── job discovery ──────────────────────────────────────────────────────────────

def _parse_results_json(path: Path) -> tuple[list[str], dict[int, dict[str, Path]]]:
    """Parse old results.json format → (lora_ids, {seed: {method: abs_path}})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    pair_dir = path.parent
    lora_ids = pair_dir.name.split("__")  # derive from folder name
    seed_map: dict[int, dict[str, Path]] = {}
    for rec in records:
        method = rec.get("method", "")
        seed   = int(rec.get("seed", 0))
        img    = pair_dir / f"{method}_seed{seed}.png"
        if img.exists():
            seed_map.setdefault(seed, {})[method] = img
    return lora_ids, seed_map


def _parse_report_json(path: Path) -> tuple[list[str], dict[int, dict[str, Path]]]:
    """Parse new report.json format → (lora_ids, {seed: {method: abs_path}})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lora_ids = data.get("lora_ids", path.parent.name.split("__"))
    images   = data.get("images", [])
    pair_dir = path.parent
    seed_map: dict[int, dict[str, Path]] = {}
    for img in images:
        method = img.get("method", "")
        seed   = int(img.get("seed", 0))
        p      = pair_dir / f"{method}_seed{seed}.png"
        if p.exists():
            seed_map.setdefault(seed, {})[method] = p
    return lora_ids, seed_map


def load_pair_jobs(
    baseline_dir: Path,
    mask_dir: Path,
    methods: Sequence[str],
    *,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """
    Build job list by joining images from two directories.

    For each method the image is sourced from whichever dir actually
    produced it: BASELINE_METHODS from baseline_dir, MASK_METHODS from mask_dir.

    Only pairs present in both directories with all requested methods are included.
    """
    # Index baseline dir (results.json or report.json)
    baseline_index: dict[str, tuple[list[str], dict[int, dict[str, Path]]]] = {}
    for rp in sorted(baseline_dir.glob("*/results.json")):
        pid = rp.parent.name
        baseline_index[pid] = _parse_results_json(rp)
    for rp in sorted(baseline_dir.glob("*/report.json")):
        pid = rp.parent.name
        if pid not in baseline_index:
            baseline_index[pid] = _parse_report_json(rp)

    # Index mask dir
    mask_index: dict[str, tuple[list[str], dict[int, dict[str, Path]]]] = {}
    for rp in sorted(mask_dir.glob("*/report.json")):
        pid = rp.parent.name
        mask_index[pid] = _parse_report_json(rp)
    for rp in sorted(mask_dir.glob("*/results.json")):
        pid = rp.parent.name
        if pid not in mask_index:
            mask_index[pid] = _parse_results_json(rp)

    common = sorted(set(baseline_index) & set(mask_index))
    if not common:
        raise ValueError(
            f"No pair_ids found in both directories.\n"
            f"  baseline_dir pairs: {sorted(baseline_index)}\n"
            f"  mask_dir pairs:     {sorted(mask_index)}"
        )

    jobs: list[dict[str, Any]] = []
    for pair_id in common:
        lora_ids, _ = mask_index[pair_id]  # report.json has explicit lora_ids
        _, baseline_seeds = baseline_index[pair_id]
        _, mask_seeds     = mask_index[pair_id]
        common_seeds = sorted(set(baseline_seeds) & set(mask_seeds))

        for seed in common_seeds:
            image_paths: dict[str, Path] = {}
            for method in methods:
                if method in BASELINE_METHODS:
                    src = baseline_seeds.get(seed, {})
                else:
                    src = mask_seeds.get(seed, {})
                if method in src:
                    image_paths[method] = src[method]

            missing = [m for m in methods if m not in image_paths]
            if missing:
                print(f"  [skip] {pair_id} seed={seed}: missing images for {missing}", flush=True)
                continue

            jobs.append({
                "combination_id": pair_id,
                "lora_ids": lora_ids,
                "seed": seed,
                "image_paths": image_paths,
            })

    return jobs[:limit] if limit > 0 else jobs


# ── LLM call ───────────────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def response_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return str(response.output_text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if t:
                chunks.append(str(t))
    return "\n".join(chunks)


def call_llm(
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
    for p in image_paths:
        content.append({"type": "input_image", "image_url": encode_image(p), "detail": detail})

    last_text = ""
    for attempt in range(1, max_retries + 1):
        resp = client.responses.create(
            model=model,
            instructions=(
                "You are a strict visual evaluator for text-to-image LoRA composition. "
                "Return only valid JSON. Do not include markdown fences."
            ),
            input=[{"role": "user", "content": content}],
        )
        last_text = response_text(resp)
        try:
            return extract_json(last_text), last_text
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec * attempt)
    raise RuntimeError("LLM call failed after all retries.")


# ── prompts ────────────────────────────────────────────────────────────────────

def ranking_prompt(
    job: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    methods: Sequence[str],
) -> str:
    method_lines = "\n".join(f"- {m}: Image {i+1}" for i, m in enumerate(methods))
    return f"""
You are evaluating synthetic images generated by different LoRA composition methods.
Treat all names and visual concepts as fictional descriptors. Do not evaluate real-world identity.

Target LoRA concepts in the image:
{describe_loras(job["lora_ids"], lookup)}

Methods and their image assignments:
{method_lines}

Score every image independently on these criteria (0–10 scale, one decimal place):
1. composition_quality: how well ALL listed LoRA concepts and their trigger features appear together in one coherent image.
2. image_quality: realism, aesthetics, absence of artifacts or visual deformations.
3. overall_quality: holistic assessment combining both criteria.

Then rank all methods from best (rank 1) to worst for each metric.

Return ONLY valid JSON matching this exact schema:
{{
  "scores": {{
    "<method>": {{
      "composition_quality": <number>,
      "image_quality": <number>,
      "overall_quality": <number>,
      "notes": "<one sentence reason>"
    }}
  }},
  "ranks": {{
    "composition_quality": ["<best_method>", "<second>", ...],
    "image_quality":       ["<best_method>", "<second>", ...],
    "overall_quality":     ["<best_method>", "<second>", ...]
  }},
  "best_method": "<method>"
}}
""".strip()


# ── aggregation ────────────────────────────────────────────────────────────────

def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def rank_index(ranks: Sequence[str], method: str) -> int:
    try:
        return list(ranks).index(method) + 1
    except ValueError:
        return len(ranks) + 1


def summarize(records: list[dict[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    metrics = ["composition_quality", "image_quality", "overall_quality"]

    # Per-method aggregate
    method_summary: dict[str, Any] = {}
    for method in methods:
        rows = [r for r in records if method in r.get("parsed", {}).get("scores", {})]
        metric_agg: dict[str, Any] = {}
        for metric in metrics:
            scores = [float(r["parsed"]["scores"][method].get(metric, 0.0)) for r in rows]
            ranks  = [rank_index(r["parsed"].get("ranks", {}).get(metric, []), method) for r in rows]
            metric_agg[metric] = {"mean_score": _mean(scores), "mean_rank": _mean(ranks)}
        method_summary[method] = metric_agg

    # Head-to-head win counts
    h2h: dict[str, Any] = {}
    for left, right in combinations(methods, 2):
        key = f"{left}_vs_{right}"
        h2h[key] = {}
        for metric in metrics:
            lw = rw = ties = 0
            for r in records:
                scores = r.get("parsed", {}).get("scores", {})
                if left not in scores or right not in scores:
                    continue
                ls = float(scores[left].get(metric, 0.0))
                rs = float(scores[right].get(metric, 0.0))
                if ls > rs:
                    lw += 1
                elif rs > ls:
                    rw += 1
                else:
                    ties += 1
            h2h[key][metric] = {left: lw, right: rw, "ties": ties}

    # Per-pair summary
    grouped: dict[str, list] = {}
    for r in records:
        grouped.setdefault(r["combination_id"], []).append(r)

    pair_rows: list[dict[str, Any]] = []
    for cid, rows in sorted(grouped.items()):
        method_scores: dict[str, dict[str, float]] = {}
        for method in methods:
            method_scores[method] = {}
            for metric in metrics:
                vals = [
                    float(r["parsed"]["scores"][method].get(metric, 0.0))
                    for r in rows if method in r.get("parsed", {}).get("scores", {})
                ]
                method_scores[method][metric] = _mean(vals)
        ranks_by_metric = {
            metric: sorted(methods, key=lambda m: -method_scores[m][metric])
            for metric in metrics
        }
        pair_rows.append({
            "combination_id": cid,
            "lora_ids": rows[0].get("lora_ids", cid.split("__")),
            "num_seeds": len(rows),
            "method_scores": method_scores,
            "ranks": ranks_by_metric,
            "best_method": ranks_by_metric["overall_quality"][0] if ranks_by_metric["overall_quality"] else "",
        })

    return {
        "num_evaluations": len(records),
        "method_summary": method_summary,
        "head_to_head": h2h,
        "pair_rows": pair_rows,
    }


# ── CSV output ─────────────────────────────────────────────────────────────────

def write_scores_csv(records: list[dict[str, Any]], methods: Sequence[str], path: Path) -> None:
    fields = [
        "combination_id", "seed", "method",
        "composition_quality", "image_quality", "overall_quality",
        "composition_rank", "image_rank", "overall_rank", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in records:
            parsed = row.get("parsed", {})
            for method in methods:
                sc = parsed.get("scores", {}).get(method)
                if not sc:
                    continue
                w.writerow({
                    "combination_id":    row["combination_id"],
                    "seed":              row["seed"],
                    "method":            method,
                    "composition_quality": sc.get("composition_quality", ""),
                    "image_quality":       sc.get("image_quality", ""),
                    "overall_quality":     sc.get("overall_quality", ""),
                    "composition_rank": rank_index(parsed.get("ranks", {}).get("composition_quality", []), method),
                    "image_rank":       rank_index(parsed.get("ranks", {}).get("image_quality", []),       method),
                    "overall_rank":     rank_index(parsed.get("ranks", {}).get("overall_quality", []),     method),
                    "notes":            sc.get("notes", ""),
                })


def write_pair_summary_csv(summary: dict[str, Any], methods: Sequence[str], path: Path) -> None:
    fields = (
        ["combination_id", "lora_ids", "num_seeds", "best_method"]
        + [f"{m}_composition" for m in methods]
        + [f"{m}_image"       for m in methods]
        + [f"{m}_overall"     for m in methods]
        + [f"{m}_overall_rank" for m in methods]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pair in summary.get("pair_rows", []):
            row: dict[str, Any] = {
                "combination_id": pair["combination_id"],
                "lora_ids":       " + ".join(pair.get("lora_ids", [])),
                "num_seeds":      pair["num_seeds"],
                "best_method":    pair["best_method"],
            }
            for m in methods:
                sc = pair["method_scores"].get(m, {})
                row[f"{m}_composition"] = sc.get("composition_quality", "")
                row[f"{m}_image"]       = sc.get("image_quality", "")
                row[f"{m}_overall"]     = sc.get("overall_quality", "")
                row[f"{m}_overall_rank"] = rank_index(pair["ranks"].get("overall_quality", []), m)
            w.writerow(row)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    baseline_dir = Path(args.baseline_dir)
    mask_dir     = Path(args.mask_dir)
    out_dir      = Path(args.out_dir) if args.out_dir else mask_dir / "llm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] baseline_dir : {baseline_dir}", flush=True)
    print(f"[eval] mask_dir     : {mask_dir}",     flush=True)
    print(f"[eval] methods      : {methods}",      flush=True)
    print(f"[eval] model        : {args.model}",   flush=True)
    print(f"[eval] out_dir      : {out_dir}",      flush=True)

    lookup = load_lora_lookup(args.lora_info_path)
    jobs   = load_pair_jobs(baseline_dir, mask_dir, methods, limit=args.limit)

    if not jobs:
        raise ValueError("No evaluable jobs found. Check that both directories have matching pair folders and seed images.")

    print(f"[eval] {len(jobs)} job(s) to evaluate", flush=True)

    client   = build_client(args)
    records:  list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, job in enumerate(jobs, 1):
        cid = job["combination_id"]
        seed = job["seed"]
        print(f"[eval] [{idx}/{len(jobs)}] {cid}  seed={seed}", flush=True)

        prompt      = ranking_prompt(job, lookup, methods)
        image_paths = [job["image_paths"][m] for m in methods]

        try:
            parsed, raw = call_llm(
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
            # Print per-method scores inline for quick inspection
            for method in methods:
                sc = parsed.get("scores", {}).get(method, {})
                print(
                    f"       {method:<28} "
                    f"comp={sc.get('composition_quality','?'):>4}  "
                    f"img={sc.get('image_quality','?'):>4}  "
                    f"overall={sc.get('overall_quality','?'):>4}",
                    flush=True,
                )
            print(f"       best_method: {parsed.get('best_method', '?')}", flush=True)

            records.append({
                "combination_id": cid,
                "lora_ids":       job["lora_ids"],
                "seed":           seed,
                "methods":        methods,
                "image_paths":    {m: str(p) for m, p in job["image_paths"].items()},
                "parsed":         parsed,
                "raw_text":       raw,
            })
        except Exception as exc:
            print(f"  [FAIL] {exc}", flush=True)
            failures.append({"combination_id": cid, "seed": seed, "error": repr(exc)})

        time.sleep(args.sleep_sec)

    # ── save outputs ─────────────────────────────────────────────────────────
    summary = summarize(records, methods)

    payload = {
        "config": {
            "baseline_dir": str(baseline_dir),
            "mask_dir":     str(mask_dir),
            "model":        args.model,
            "methods":      methods,
        },
        "summary":  summary,
        "records":  records,
        "failures": failures,
    }

    (out_dir / "eval_results.json" ).write_text(json.dumps(payload,  indent=2), encoding="utf-8")
    (out_dir / "eval_summary.json" ).write_text(json.dumps(summary,  indent=2), encoding="utf-8")
    write_scores_csv     (records, methods, out_dir / "eval_scores.csv")
    write_pair_summary_csv(summary, methods, out_dir / "eval_pair_summary.csv")

    # ── print aggregate table ─────────────────────────────────────────────────
    print("\n" + "═" * 70, flush=True)
    print(f"{'METHOD':<28}  {'COMP':>6}  {'IMG':>6}  {'OVERALL':>8}  {'AVG RANK':>8}", flush=True)
    print("─" * 70, flush=True)
    for method in methods:
        ms = summary["method_summary"].get(method, {})
        comp    = ms.get("composition_quality", {}).get("mean_score",  0.0)
        img     = ms.get("image_quality",        {}).get("mean_score",  0.0)
        overall = ms.get("overall_quality",       {}).get("mean_score",  0.0)
        rank    = ms.get("overall_quality",       {}).get("mean_rank",   0.0)
        print(f"{method:<28}  {comp:6.2f}  {img:6.2f}  {overall:8.2f}  {rank:8.2f}", flush=True)
    print("═" * 70, flush=True)

    print(f"\n[eval] outputs saved to {out_dir}", flush=True)
    if failures:
        print(f"[eval] {len(failures)} failure(s) — see eval_results.json", flush=True)


if __name__ == "__main__":
    main()
