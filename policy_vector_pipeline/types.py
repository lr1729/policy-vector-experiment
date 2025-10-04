from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ResponseVariant:
    """A single reasoning rollout (reasoning + final answer)."""

    reasoning: str
    final_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compose(self, include_answer: bool = True, separator: str = "\n") -> str:
        """Return reasoning text (optionally including the final answer)."""
        parts: List[str] = []
        reason = self.reasoning.strip()
        if reason:
            parts.append(reason)
        if include_answer:
            answer = self.final_answer.strip()
            if answer:
                parts.append(answer)
        return separator.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ResponseVariant":
        return ResponseVariant(
            reasoning=raw.get("reasoning", ""),
            final_answer=raw.get("final_answer", ""),
            metadata=raw.get("metadata", {}),
        )


@dataclass
class ReasoningExample:
    """Stores on-policy and off-policy variants for the same prompt."""

    example_id: str
    prompt: str
    on_policy: List[ResponseVariant]
    off_policy: List[ResponseVariant]
    correct_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: Optional[float] = None

    def all_variants(self, variant_type: str) -> Iterable[ResponseVariant]:
        if variant_type == "on":
            return self.on_policy
        if variant_type == "off":
            return self.off_policy
        raise ValueError("variant_type must be 'on' or 'off'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "on_policy": [v.to_dict() for v in self.on_policy],
            "off_policy": [v.to_dict() for v in self.off_policy],
            "correct_answer": self.correct_answer,
            "metadata": self.metadata,
            "similarity": self.similarity,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ReasoningExample":
        return ReasoningExample(
            example_id=raw["example_id"],
            prompt=raw["prompt"],
            on_policy=[ResponseVariant.from_dict(v) for v in raw.get("on_policy", [])],
            off_policy=[ResponseVariant.from_dict(v) for v in raw.get("off_policy", [])],
            correct_answer=raw.get("correct_answer"),
            metadata=raw.get("metadata", {}),
            similarity=raw.get("similarity"),
        )


@dataclass
class OnPolicyDataset:
    """Collection of reasoning examples with optional metadata."""

    examples: List[ReasoningExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, example: ReasoningExample) -> None:
        self.examples.append(example)

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def filter_by_similarity(self, threshold: float) -> "OnPolicyDataset":
        filtered = [ex for ex in self.examples if ex.similarity is None or ex.similarity >= threshold]
        return OnPolicyDataset(filtered, metadata=self.metadata.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "examples": [ex.to_dict() for ex in self.examples],
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "OnPolicyDataset":
        meta = raw.get("metadata", {})
        examples = [ReasoningExample.from_dict(ex) for ex in raw.get("examples", [])]
        return OnPolicyDataset(examples=examples, metadata=meta)
