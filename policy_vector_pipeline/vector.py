from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch


@dataclass
class MeanDifferenceVector:
    """Mean(on) - Mean(off) per layer."""

    layer_vectors: Dict[int, torch.Tensor]

    def to(self, device: torch.device) -> "MeanDifferenceVector":
        return MeanDifferenceVector({layer: vec.to(device) for layer, vec in self.layer_vectors.items()})

    def norm(self, layer: int) -> float:
        return torch.linalg.norm(self.layer_vectors[layer]).item()

    def save(self, path: Path) -> None:
        torch.save({layer: vec.cpu() for layer, vec in self.layer_vectors.items()}, path)

    @staticmethod
    def load(path: Path) -> "MeanDifferenceVector":
        data = torch.load(path, map_location="cpu")
        return MeanDifferenceVector(layer_vectors={int(k): v for k, v in data.items()})

    @staticmethod
    def from_activations(
        on_policy: Dict[int, Iterable[torch.Tensor]],
        off_policy: Dict[int, Iterable[torch.Tensor]],
    ) -> "MeanDifferenceVector":
        layers = sorted(set(on_policy.keys()) & set(off_policy.keys()))
        layer_vectors: Dict[int, torch.Tensor] = {}
        for layer in layers:
            on_stack = torch.stack([tensor.to(torch.float32) for tensor in on_policy[layer]])
            off_stack = torch.stack([tensor.to(torch.float32) for tensor in off_policy[layer]])
            diff = on_stack.mean(dim=0) - off_stack.mean(dim=0)
            layer_vectors[layer] = diff
        return MeanDifferenceVector(layer_vectors=layer_vectors)
