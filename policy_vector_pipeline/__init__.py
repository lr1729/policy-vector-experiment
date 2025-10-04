"""Utilities for building an on-policy steering dataset and vectors."""

from .types import ResponseVariant, ReasoningExample, OnPolicyDataset
from .activation import ActivationCollector
from .vector import MeanDifferenceVector
from .steering import SteeringRunner
from .generation import (
    GenerationSettings,
    build_off_policy_variants,
    generate_on_policy_variants,
    parse_reasoning,
    run_generation,
)
from .evaluation import EvaluationRecord, accuracy, evaluate_accuracy, simple_answer_checker
from .io import load_dataset, save_dataset
from .prompt_sources import load_prompts

__all__ = [
    "ResponseVariant",
    "ReasoningExample",
    "OnPolicyDataset",
    "ActivationCollector",
    "MeanDifferenceVector",
    "SteeringRunner",
    "GenerationSettings",
    "generate_on_policy_variants",
    "build_off_policy_variants",
    "parse_reasoning",
    "run_generation",
    "EvaluationRecord",
    "accuracy",
    "evaluate_accuracy",
    "simple_answer_checker",
    "load_dataset",
    "save_dataset",
    "load_prompts",
]
