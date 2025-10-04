from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .generation import parse_reasoning


@dataclass
class EvaluationRecord:
    prompt: str
    completion: str
    is_correct: bool
    metadata: dict


def simple_answer_checker(target: str) -> Callable[[str], bool]:
    target = target.strip()

    def _check(text: str) -> bool:
        parsed = parse_reasoning(text)
        answer = parsed.final_answer.strip()
        return target in answer

    return _check


def evaluate_accuracy(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    answer_checker: Callable[[str], bool],
) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    for prompt, completion in zip(prompts, completions):
        is_correct = answer_checker(completion)
        record = EvaluationRecord(
            prompt=prompt,
            completion=completion,
            is_correct=is_correct,
            metadata={},
        )
        records.append(record)
    return records


def accuracy(records: Iterable[EvaluationRecord]) -> float:
    records = list(records)
    if not records:
        return 0.0
    correct = sum(1 for record in records if record.is_correct)
    return correct / len(records)
