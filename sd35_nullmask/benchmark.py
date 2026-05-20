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

    return {
        "status": "planned_only",
        "report_path": str(report_path),
        "message": "Execution hooks for full SD3.5 null-mask benchmarking are scaffolded but not enabled in this environment yet.",
    }
