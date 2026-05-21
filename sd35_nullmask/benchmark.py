from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuit_utils import load_json_config

from .config import SD35NullMaskBenchmarkConfig, SD35NullMaskConfig
from .inventory import benchmark_pairs_from_inventory_config, build_inventory_context, resolve_selected_adapters, validate_selected_adapters
from .methods import validate_methods
from .prompting import build_pair_prompt


def build_benchmark_plan(config: SD35NullMaskBenchmarkConfig, config_path: str | None = None) -> dict[str, Any]:
    raw = load_json_config(config_path, key="dit_adapter_inventory") if config_path else {}
    pair_specs = config.benchmark_pairs or benchmark_pairs_from_inventory_config(raw)
    inventory = build_inventory_context(config.inventory_path, config.local_root)
    validate_methods(config.methods)

    rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        adapters = resolve_selected_adapters(inventory, pair)
        prompt, resolved_triggers = build_pair_prompt("", adapters, override=config.trigger_token_override)
        rows.append(
            {
                "pair_id": "__".join(pair),
                "lora_ids": pair,
                "categories": [adapter.category for adapter in adapters],
                "local_available": all(adapter.local_available for adapter in adapters),
                "prompt_suffix": prompt,
                "trigger_phrases": [item.phrase for item in resolved_triggers],
                "seeds": config.benchmark_seeds,
                "methods": config.methods,
            }
        )

    return {
        "model_name": config.model_name,
        "inventory_path": config.inventory_path,
        "local_root": config.local_root,
        "num_pairs": len(rows),
        "methods": config.methods,
        "rows": rows,
    }


def run_benchmark(config: SD35NullMaskBenchmarkConfig, *, config_path: str | None = None) -> dict[str, Any]:
    plan = build_benchmark_plan(config, config_path=config_path)
    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "benchmark_plan.json"
    report_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if config.dry_run:
        return {"status": "dry_run", "report_path": str(report_path), "num_pairs": plan["num_pairs"]}

    from .backend import SD35PipelineBackend
    from .config import SD35NullMaskConfig

    pipeline_config = SD35NullMaskConfig(model_name=config.model_name)
    backend = SD35PipelineBackend(pipeline_config)
    backend.load_pipeline()

    inventory = build_inventory_context(config.inventory_path, config.local_root)
    pair_results: list[dict[str, Any]] = []

    for row in plan["rows"]:
        pair_id: str = row["pair_id"]
        lora_ids: list[str] = row["lora_ids"]
        prompt: str = row["prompt_suffix"]

        if not row["local_available"]:
            pair_results.append({"pair_id": pair_id, "status": "skipped", "reason": "missing_local_files"})
            continue

        backend.pipeline.unload_lora_weights()

        adapters = validate_selected_adapters(inventory, lora_ids)
        _, resolved_triggers = build_pair_prompt("", adapters, override=config.trigger_token_override)
        backend.load_adapters(adapters, resolved_triggers)

        pair_config = SD35NullMaskConfig(
            model_name=config.model_name,
            inventory_path=config.inventory_path,
            local_root=config.local_root,
            lora_ids=lora_ids,
            lora_weights=[1.0] * len(lora_ids),
            prompt=prompt,
            denoise_steps=config.denoise_steps,
            lookahead_steps=config.lookahead_steps,
            svd_rank=config.svd_rank,
            switch_step=config.switch_step,
            methods=row["methods"],
            out_dir=config.out_dir,
            trigger_token_override=config.trigger_token_override,
        )

        records = backend.run_all_methods(
            methods=row["methods"],
            seeds=row["seeds"],
            prompt=prompt,
            pair_id=pair_id,
            out_root=out_root,
            config=pair_config,
        )

        pair_out = out_root / pair_id
        pair_report: dict[str, Any] = {
            "pair_id": pair_id,
            "lora_ids": lora_ids,
            "prompt": prompt,
            "status": "done",
            "images": records,
        }
        (pair_out / "report.json").write_text(json.dumps(pair_report, indent=2), encoding="utf-8")
        pair_results.append({"pair_id": pair_id, "status": "done", "report_path": str(pair_out / "report.json")})

    summary: dict[str, Any] = {
        "status": "done",
        "report_path": str(report_path),
        "num_pairs": len(plan["rows"]),
        "num_completed": sum(1 for r in pair_results if r["status"] == "done"),
        "pairs": pair_results,
    }
    (out_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
