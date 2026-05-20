from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dit_adapter_inventory import build_lookup, discover_local_files, flatten_inventory, load_inventory


@dataclass
class InventoryAdapter:
    adapter_id: str
    category: str
    repo_id: str
    source_url: str
    trigger: list[str]
    notes: str
    expected_local_dir: str
    local_files: list[str]

    @property
    def local_available(self) -> bool:
        return bool(self.local_files)


@dataclass
class InventoryContext:
    inventory_path: str
    local_root: str
    adapters: dict[str, InventoryAdapter]


def build_inventory_context(inventory_path: str, local_root: str) -> InventoryContext:
    payload = load_inventory(inventory_path)
    lookup = build_lookup(flatten_inventory(payload))
    adapters: dict[str, InventoryAdapter] = {}
    for adapter_id, row in lookup.items():
        local_files = discover_local_files(local_root, adapter_id)
        adapters[adapter_id] = InventoryAdapter(
            adapter_id=adapter_id,
            category=str(row["category"]),
            repo_id=str(row["repo_id"]),
            source_url=str(row["source_url"]),
            trigger=[str(token) for token in row.get("trigger", [])],
            notes=str(row.get("notes", "")),
            expected_local_dir=str(Path(local_root) / adapter_id),
            local_files=local_files,
        )
    return InventoryContext(inventory_path=inventory_path, local_root=local_root, adapters=adapters)


def validate_selected_adapters(context: InventoryContext, adapter_ids: list[str]) -> list[InventoryAdapter]:
    resolved = resolve_selected_adapters(context, adapter_ids)
    missing = [adapter.adapter_id for adapter in resolved if not adapter.local_available]
    if missing:
        missing_dirs = [context.adapters[adapter_id].expected_local_dir for adapter_id in missing]
        raise FileNotFoundError(
            "Selected adapters are missing locally: "
            + ", ".join(missing)
            + ". Expected directories: "
            + ", ".join(missing_dirs)
        )
    return resolved


def resolve_selected_adapters(context: InventoryContext, adapter_ids: list[str]) -> list[InventoryAdapter]:
    resolved: list[InventoryAdapter] = []
    for adapter_id in adapter_ids:
        adapter = context.adapters.get(adapter_id)
        if adapter is None:
            raise ValueError(f"Adapter id not found in inventory: {adapter_id}")
        resolved.append(adapter)
    return resolved


def benchmark_pairs_from_inventory_config(config_payload: dict[str, Any]) -> list[list[str]]:
    rows = config_payload.get("benchmark_pairs", [])
    out: list[list[str]] = []
    for pair in rows:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid benchmark pair: {pair}")
        out.append([str(pair[0]), str(pair[1])])
    return out
