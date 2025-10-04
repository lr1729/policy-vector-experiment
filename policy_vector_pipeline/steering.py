from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

try:
    from persona_vectors.activation_steer import ActivationSteerer
except ImportError:  # pragma: no cover - fallback for trimmed repo
    ActivationSteerer = None  # type: ignore

from .generation import GenerationSettings, run_generation
from .vector import MeanDifferenceVector


@dataclass
class SteeringResult:
    prompt: str
    completion: str
    alpha: float
    layer: int
    metadata: Dict[str, object]


class SteeringRunner:
    """Apply a mean-difference vector during generation."""

    def __init__(
        self,
        model,
        tokenizer,
        vector: MeanDifferenceVector,
        *,
        default_layer: Optional[int] = None,
        positions: str = "response",
    ) -> None:
        if ActivationSteerer is None:
            raise ImportError(
                "persona_vectors.activation_steer is required for steering. "
                "Ensure the persona_vectors package is on PYTHONPATH."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.positions = positions
        self.default_layer = default_layer or max(vector.layer_vectors.keys())

    def generate(
        self,
        prompt: str,
        *,
        alpha: float = 1.0,
        layer: Optional[int] = None,
        settings: Optional[GenerationSettings] = None,
        **kwargs,
    ) -> SteeringResult:
        layer = self._resolve_layer(layer)
        vec = self.vector.layer_vectors[layer]
        with ActivationSteerer(
            self.model,
            vec,
            coeff=alpha,
            layer_idx=layer,
            positions=self.positions,
        ):
            completion = run_generation(
                self.model,
                self.tokenizer,
                prompt,
                settings=settings,
                num_samples=1,
            )[0]
        return SteeringResult(
            prompt=prompt,
            completion=completion,
            alpha=alpha,
            layer=layer,
            metadata={"positions": self.positions},
        )

    def sweep(
        self,
        prompt: str,
        alphas: Iterable[float],
        *,
        layer: Optional[int] = None,
        settings: Optional[GenerationSettings] = None,
    ) -> List[SteeringResult]:
        results: List[SteeringResult] = []
        for alpha in alphas:
            results.append(
                self.generate(
                    prompt,
                    alpha=alpha,
                    layer=layer,
                    settings=settings,
                )
            )
        return results

    def _resolve_layer(self, layer: Optional[int]) -> int:
        if layer is not None:
            return layer
        return self.default_layer
