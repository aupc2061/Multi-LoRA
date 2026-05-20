from __future__ import annotations

from dataclasses import dataclass

from .inventory import InventoryAdapter


@dataclass
class ResolvedTrigger:
    adapter_id: str
    phrase: str


def resolve_trigger_phrase(adapter: InventoryAdapter, override: dict[str, str] | None = None) -> ResolvedTrigger:
    override = override or {}
    if adapter.adapter_id in override and override[adapter.adapter_id].strip():
        return ResolvedTrigger(adapter_id=adapter.adapter_id, phrase=override[adapter.adapter_id].strip())
    if adapter.trigger:
        return ResolvedTrigger(adapter_id=adapter.adapter_id, phrase=str(adapter.trigger[0]).strip())
    raise ValueError(
        f"Adapter '{adapter.adapter_id}' has no trigger phrase in inventory. "
        "Provide trigger_token_override for this adapter."
    )


def build_pair_prompt(base_prompt: str, adapters: list[InventoryAdapter], override: dict[str, str] | None = None) -> tuple[str, list[ResolvedTrigger]]:
    resolved = [resolve_trigger_phrase(adapter, override=override) for adapter in adapters]
    prompt = base_prompt.strip()
    suffix = ", ".join(item.phrase for item in resolved if item.phrase)
    if suffix:
        prompt = f"{prompt}, {suffix}" if prompt else suffix
    return prompt, resolved
