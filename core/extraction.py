"""Framework-independent activation extraction interface."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from core.activations import ActivationSet


@dataclass(frozen=True)
class ExtractionSpec:
    location: str
    reduction: str = "last_token"
    batch_size: int = 8
    dtype: str = "float32"


class ActivationExtractor(Protocol):
    def extract(self, inputs: Iterable[Any], spec: ExtractionSpec) -> ActivationSet: ...
