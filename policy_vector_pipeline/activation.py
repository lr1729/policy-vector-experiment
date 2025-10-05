from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .types import OnPolicyDataset, ReasoningExample, ResponseVariant


@dataclass
class ActivationRecord:
    example_id: str
    variant_kind: str  # "on" or "off"
    layer_activations: Dict[int, torch.Tensor]
    metadata: Dict[str, object]


class ActivationCollector:
    """Extract response-token activations for on/off-policy reasoning."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        layers: Optional[Iterable[int]] = None,
        reduction: str = "mean",
        response_only: bool = True,
        include_answer: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reduction = reduction
        self.response_only = response_only
        self.include_answer = include_answer
        if device is None:
            if hasattr(model, "device"):
                device = model.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        total_layers = getattr(model.config, "num_hidden_layers", None)
        if total_layers is None:
            raise ValueError("Model must expose num_hidden_layers in config")

        if layers is None:
            start = total_layers // 2
            end = total_layers - 1
            layers = list(range(start, end))
        self.layers = sorted(set(int(idx) for idx in layers))

        self.model.eval()

    # ------------------------------------------------------------------
    def collect_example(
        self,
        example: ReasoningExample,
        variant_type: str,
        *,
        progress: bool = False,
    ) -> List[ActivationRecord]:
        variants = list(example.all_variants(variant_type))
        records: List[ActivationRecord] = []

        iterator = enumerate(variants)
        if progress:
            iterator = tqdm(iterator, desc=f"{example.example_id}:{variant_type}", total=len(variants))

        for idx, variant in iterator:
            acts = self._extract_for_variant(example.prompt, variant)
            record = ActivationRecord(
                example_id=example.example_id,
                variant_kind=variant_type,
                layer_activations=acts,
                metadata={
                    "variant_index": idx,
                    "prompt": example.prompt,
                    "correct_answer": example.correct_answer,
                },
            )
            records.append(record)
        return records

    def collect_dataset(
        self,
        dataset: OnPolicyDataset,
        *,
        progress: bool = True,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[int, List[torch.Tensor]]]:
        """Return dicts mapping variant type -> layer -> activations list."""
        store: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "on": defaultdict(list),
            "off": defaultdict(list),
        }

        examples = dataset.examples[: limit or len(dataset)]
        outer_iter = examples
        if progress:
            outer_iter = tqdm(outer_iter, desc="Collecting activations")

        for example in outer_iter:
            for kind in ("on", "off"):
                records = self.collect_example(example, kind, progress=False)
                for record in records:
                    for layer, tensor in record.layer_activations.items():
                        store[kind][layer].append(tensor.cpu())
        return store

    # ------------------------------------------------------------------
    def _extract_for_variant(self, prompt: str, variant: ResponseVariant) -> Dict[int, torch.Tensor]:
        response_text = variant.compose(include_answer=self.include_answer)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        full_inputs = self.tokenizer(
            prompt + response_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states  # tuple len = num_layers + 1
        prompt_len = prompt_ids.shape[1]

        activations: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layers:
            # hidden_states[0] is embeddings
            tensor = hidden_states[layer_idx + 1][0]  # shape [seq_len, hidden]
            if self.response_only:
                tensor = tensor[prompt_len:, :]
            if tensor.numel() == 0:
                raise ValueError("No response tokens captured; check prompt formatting")
            reduced = self._reduce_tensor(tensor)
            activations[layer_idx] = reduced.cpu()
        return activations

    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean(dim=0)
        if self.reduction == "last":
            return tensor[-1]
        if self.reduction == "none":
            return tensor
        raise ValueError("reduction must be 'mean', 'last', or 'none'")
