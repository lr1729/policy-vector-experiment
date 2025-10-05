"""Utilities for building an on-policy steering dataset and vectors."""

from .types import ResponseVariant, ReasoningExample, OnPolicyDataset, load_dataset, save_dataset
from .activation import ActivationCollector
from .vector import MeanDifferenceVector
from .steering import SteeringRunner
from .generation import (
    GenerationSettings,
    build_off_policy_variants,
    generate_on_policy_variants,
    run_generation,
)
from .utils import get_default_device, load_model_and_tokenizer, EmbeddingSimilarity

__all__ = [
    "ResponseVariant",
    "ReasoningExample",
    "OnPolicyDataset",
    "load_dataset",
    "save_dataset",
    "ActivationCollector",
    "MeanDifferenceVector",
    "SteeringRunner",
    "GenerationSettings",
    "generate_on_policy_variants",
    "build_off_policy_variants",
    "run_generation",
    "get_default_device",
    "load_model_and_tokenizer",
    "EmbeddingSimilarity",
]
