from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LastActivationRecorder:
    module: Any

    def __post_init__(self) -> None:
        self.last = None
        self._handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self.last = output

    def reset(self) -> None:
        self.last = None

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
