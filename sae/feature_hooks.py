from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LastActivationRecorder:
    module: Any
    capture_last_n: int = 1  # number of most-recent steps to retain

    def __post_init__(self) -> None:
        # Stores every activation fired by the hook during a pipeline call.
        # get_all_steps() returns the last capture_last_n of them.
        self._steps: list[torch.Tensor] = []
        self._handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self._steps.append(output)

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def last(self) -> torch.Tensor | None:
        """Most recent activation (backward-compatible with old callers)."""
        return self._steps[-1] if self._steps else None

    def get_all_steps(self) -> list[torch.Tensor]:
        """Return the last *capture_last_n* recorded activations."""
        return self._steps[-self.capture_last_n :]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._steps = []

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def resolve_module(root: Any, module_path: str) -> Any:
    node = root
    for token in module_path.split('.'):
        if not hasattr(node, token):
            raise ValueError(f"Module path '{module_path}' is invalid at token '{token}'.")
        node = getattr(node, token)
    return node
