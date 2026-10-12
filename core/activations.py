"""Activation data structures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ActivationSource:
    """Description of where a set of activations came from."""

    model_id: str
    location: str
    reduction: str
    model_revision: str | None = None


@dataclass(frozen=True)
class ActivationSet:
    """A two-dimensional activation matrix and row-aligned metadata."""

    values: FloatArray
    sample_ids: Sequence[str] | None
    metadata: Sequence[Mapping[str, Any]] | None
    source: ActivationSource

    def __post_init__(self) -> None:
        values = as_activation_values(self.values)
        if self.sample_ids is not None and len(self.sample_ids) != len(values):
            raise ValueError("sample_ids must have one entry per activation")
        if self.metadata is not None and len(self.metadata) != len(values):
            raise ValueError("metadata must have one entry per activation")
        object.__setattr__(self, "values", values)
        if self.sample_ids is not None:
            object.__setattr__(self, "sample_ids", tuple(str(item) for item in self.sample_ids))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", tuple(dict(item) for item in self.metadata))

    def select(self, indices: npt.ArrayLike) -> ActivationSet:
        selected = np.atleast_1d(np.arange(len(self.values))[indices])
        return ActivationSet(
            values=self.values[selected],
            sample_ids=(
                [self.sample_ids[int(index)] for index in selected]
                if self.sample_ids is not None
                else None
            ),
            metadata=(
                [self.metadata[int(index)] for index in selected]
                if self.metadata is not None
                else None
            ),
            source=self.source,
        )

    def save(self, path: str | Path) -> None:
        """Save arrays and JSON-compatible metadata without pickle."""
        np.savez_compressed(
            Path(path),
            values=self.values,
            sample_ids=np.asarray(self.sample_ids or (), dtype=np.str_),
            metadata=np.asarray(
                [json.dumps(item) for item in (self.metadata or ())], dtype=np.str_
            ),
            has_sample_ids=np.asarray(self.sample_ids is not None),
            has_metadata=np.asarray(self.metadata is not None),
            source=np.asarray(json.dumps(asdict(self.source)), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> ActivationSet:
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                values=data["values"],
                sample_ids=data["sample_ids"].tolist() if bool(data["has_sample_ids"]) else None,
                metadata=(
                    [json.loads(item) for item in data["metadata"].tolist()]
                    if bool(data["has_metadata"])
                    else None
                ),
                source=ActivationSource(**json.loads(str(data["source"]))),
            )


def as_activation_values(values: ActivationSet | npt.ArrayLike) -> FloatArray:
    """Return a validated floating-point activation matrix."""
    if isinstance(values, ActivationSet):
        return values.values
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("activations must have shape (n_points, n_dimensions)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("activations must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("activations must be numeric")
    dtype = np.float32 if array.dtype.itemsize < 4 else array.dtype
    array = np.asarray(array, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("activations must contain only finite values")
    return array
