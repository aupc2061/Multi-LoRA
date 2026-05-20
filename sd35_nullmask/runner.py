from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backend import SD35PipelineBackend
from .config import SD35NullMaskConfig
from .inventory import build_inventory_context, resolve_selected_adapters, validate_selected_adapters
from .methods import validate_methods
from .prompting import build_pair_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SD3.5 null-mask LoRA mixing prototype.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--out_dir", type=str, default="")
    return parser.parse_args()


def build_run_manifest(config: SD35NullMaskConfig) -> dict[str, Any]:
    inventory = build_inventory_context(config.inventory_path, config.local_root)
    adapters = resolve_selected_adapters(inventory, config.lora_ids)
    prompt, resolved_triggers = build_pair_prompt(config.prompt, adapters, override=config.trigger_token_override)
    return {
        "pair_id": config.pair_id,
        "model_name": config.model_name,
        "inventory_path": config.inventory_path,
        "local_root": config.local_root,
        "prompt": prompt,
        "negative_prompt": config.negative_prompt,
        "methods": validate_methods(config.methods),
        "adapters": [
            {
                "adapter_id": adapter.adapter_id,
                "category": adapter.category,
                "repo_id": adapter.repo_id,
                "local_files": adapter.local_files,
                "local_available": adapter.local_available,
                "trigger_phrase": trigger.phrase,
            }
            for adapter, trigger in zip(adapters, resolved_triggers)
        ],
        "settings": {
            "denoise_steps": config.denoise_steps,
            "lookahead_steps": config.lookahead_steps,
            "intervention_block_start": config.intervention_block_start,
            "intervention_block_end": config.intervention_block_end,
            "svd_rank": config.svd_rank,
            "switch_step": config.switch_step,
            "seeds": config.seeds,
        },
    }


def main() -> None:
    args = parse_args()
    config = SD35NullMaskConfig.from_config(args.config)
    if args.out_dir:
        config.out_dir = args.out_dir
    if args.dry_run:
        config.dry_run = True

    manifest = build_run_manifest(config)
    out_root = Path(config.out_dir) / config.pair_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.preflight_only or config.dry_run:
        print(json.dumps({"status": "preflight_ready", "manifest_path": str(out_root / "manifest.json")}, indent=2))
        return

    inventory = build_inventory_context(config.inventory_path, config.local_root)
    adapters = validate_selected_adapters(inventory, config.lora_ids)
    _, resolved_triggers = build_pair_prompt(config.prompt, adapters, override=config.trigger_token_override)
    backend = SD35PipelineBackend(config)
    preflight = backend.run_preflight(adapters, resolved_triggers)
    backend.save_json(out_root / "preflight.json", preflight)
    print(json.dumps({"status": "runtime_ready", "preflight_path": str(out_root / "preflight.json")}, indent=2))


if __name__ == "__main__":
    main()
