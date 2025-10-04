from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .types import OnPolicyDataset


def load_dataset(path: Union[str, Path]) -> OnPolicyDataset:
    """Load an on-policy dataset from a JSON file."""
    raw = json.loads(Path(path).read_text())
    return OnPolicyDataset.from_dict(raw)


def save_dataset(dataset: OnPolicyDataset, path: Union[str, Path], *, indent: int = 2) -> None:
    """Persist a dataset to disk."""
    Path(path).write_text(json.dumps(dataset.to_dict(), indent=indent, ensure_ascii=False))
