from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .types import OnPolicyDataset, ReasoningExample, ResponseVariant


def _extract_reasoning_span(reasoning: str, line_numbers: List[int]) -> str:
    if not line_numbers:
        return reasoning
    lines = reasoning.splitlines()
    keep: List[str] = []
    counter = 0
    target = max(line_numbers)
    for line in lines:
        keep.append(line)
        stripped = line.strip()
        if stripped and stripped not in {"<think>", "</think>"}:
            counter += 1
        if counter >= target:
            break
    return "\n".join(keep)


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
        enable_thinking: Optional[bool] = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reduction = reduction
        self.response_only = response_only
        self.include_answer = include_answer
        self.enable_thinking = enable_thinking
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
    def collect_dataset(
        self,
        dataset: OnPolicyDataset,
        *,
        progress: bool = True,
        limit: Optional[int] = None,
        clip_to_span: bool = False,
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
            if clip_to_span:
                self._collect_with_spans(example, store)
            else:
                for kind in ("on", "off"):
                    variants = list(example.all_variants(kind))
                    for idx, variant in enumerate(variants):
                        acts = self._extract_for_variant(example.prompt, variant)
                        self._record(store, kind, acts)
        return store

    # ------------------------------------------------------------------
    def _extract_for_variant(
        self,
        prompt: str,
        variant: ResponseVariant,
        *,
        reasoning_override: Optional[str] = None,
    ) -> Dict[int, torch.Tensor]:
        reasoning_text = reasoning_override or variant.reasoning
        temp_variant = ResponseVariant(
            reasoning=reasoning_text,
            final_answer=variant.final_answer,
            metadata=variant.metadata,
        )
        response_text = temp_variant.compose(include_answer=self.include_answer)

        # Build chat template text to ensure tokenisation matches generation.
        user_messages = [{"role": "user", "content": prompt}]
        chat_kwargs: Dict[str, object] = {"tokenize": False}
        if self.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self.enable_thinking

        prompt_text = self.tokenizer.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            **chat_kwargs,
        )

        assistant_messages = user_messages + [{"role": "assistant", "content": response_text}]
        full_text = self.tokenizer.apply_chat_template(
            assistant_messages,
            add_generation_prompt=False,
            **chat_kwargs,
        )

        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}

        full_inputs = self.tokenizer(full_text, return_tensors="pt")
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        prompt_ids = prompt_inputs["input_ids"]

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

    # ------------------------------------------------------------------
    def _collect_with_spans(
        self,
        example: ReasoningExample,
        store: Dict[str, Dict[int, List[torch.Tensor]]],
    ) -> None:
        off_variants = list(example.off_policy or [])
        if not off_variants:
            return

        for variant in off_variants:
            line_numbers = self._line_numbers_for_variant(variant)
            if not line_numbers:
                continue
            on_idx = int(variant.metadata.get("source_rollout", 0)) if variant.metadata else 0
            if on_idx >= len(example.on_policy):
                continue

            on_variant = example.on_policy[on_idx]
            on_override = _extract_reasoning_span(on_variant.reasoning, line_numbers)
            off_override = _extract_reasoning_span(variant.reasoning, line_numbers)

            on_acts = self._extract_for_variant(example.prompt, on_variant, reasoning_override=on_override)
            off_acts = self._extract_for_variant(example.prompt, variant, reasoning_override=off_override)
            self._record(store, "on", on_acts)
            self._record(store, "off", off_acts)

    def _record(
        self,
        store: Dict[str, Dict[int, List[torch.Tensor]]],
        kind: str,
        acts: Dict[int, torch.Tensor],
    ) -> None:
        for layer, tensor in acts.items():
            store[kind][layer].append(tensor.cpu())

    def _line_numbers_for_variant(self, variant: ResponseVariant) -> Optional[List[int]]:
        if not variant.metadata:
            return None
        for key in ("line_numbers", "edited_line_numbers", "paired_line_numbers"):
            numbers = variant.metadata.get(key)
            if isinstance(numbers, list) and numbers:
                return [int(num) for num in numbers]
        return None
