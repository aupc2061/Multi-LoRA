"""
Local utility module for DIT (diffusion transformer) adapter inventory management.

Handles loading, flattening, and local file discovery for LoRA adapter inventories.
Consumed by sd35_nullmask/inventory.py via:
    from dit_adapter_inventory import build_lookup, discover_local_files, flatten_inventory, load_inventory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_inventory(path: str) -> dict[str, Any]:
    """Load inventory JSON from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a category-keyed inventory dict into a flat list of adapter rows.

    Each row is a copy of the original item dict with these fields normalised/added:
      - ``category``   : the top-level key the item was found under
      - ``source_url`` : copied from ``url`` when not already present
      - ``repo_id``    : kept as-is if present; otherwise derived from a HuggingFace
                         URL (everything after ``huggingface.co/``), or empty string
      - ``notes``      : copied from ``name`` when not already present
    """
    rows: list[dict[str, Any]] = []
    for category, items in payload.items():
        if not isinstance(items, list):
            continue
        for item in items:
            row: dict[str, Any] = dict(item)
            row["category"] = category

            # Normalise source_url
            if "source_url" not in row:
                row["source_url"] = row.get("url", "")

            # Normalise repo_id — explicit field wins; otherwise derive from HF URL
            if not row.get("repo_id"):
                url = row.get("url", "")
                if "huggingface.co/" in url:
                    after_hf = url.split("huggingface.co/")[-1].strip("/")
                    parts = after_hf.split("/")
                    row["repo_id"] = "/".join(parts[:2]) if len(parts) >= 2 else after_hf
                else:
                    row["repo_id"] = ""

            # Normalise notes
            if "notes" not in row:
                row["notes"] = row.get("name", "")

            rows.append(row)
    return rows


def build_lookup(flat_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a dict keyed by each row's ``id`` field.

    Rows without an ``id`` field are silently skipped.
    """
    return {row["id"]: row for row in flat_rows if "id" in row}


def discover_local_files(local_root: str, adapter_id: str) -> list[str]:
    """Return a sorted list of local weight file paths under ``local_root/adapter_id/``.

    Searches for ``.safetensors``, ``.bin``, and ``.pt`` files.
    Returns an empty list if the directory does not exist or contains no weight files.
    """
    adapter_dir = Path(local_root) / adapter_id
    if not adapter_dir.is_dir():
        return []
    found: list[str] = []
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        found.extend(str(p) for p in sorted(adapter_dir.glob(pattern)))
    return found
