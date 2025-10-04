from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from .types import ResponseVariant

THINK_START = "<think>"
THINK_END = "</think>"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 8192
    temperature: float = 0.6  # Qwen3 thinking mode: 0.6
    top_p: float = 0.95       # Qwen3 thinking mode: 0.95
    top_k: int = 20           # Qwen3 thinking mode: 20
    do_sample: bool = True    # NEVER use greedy for thinking mode!

    def to_kwargs(self) -> Dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
        }


def _ensure_generation_config(model: PreTrainedModel, settings: GenerationSettings) -> GenerationConfig:
    cfg = model.generation_config
    for key, value in settings.to_kwargs().items():
        setattr(cfg, key, value)
    return cfg


def run_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    settings: Optional[GenerationSettings] = None,
    num_samples: int = 1,
    enable_thinking: bool = True,
) -> List[str]:
    """Generate raw completions for a prompt using Qwen3's chat template.

    Args:
        prompt: User question/problem
        enable_thinking: Enable Qwen3's thinking mode (default True)
    """
    settings = settings or GenerationSettings()
    cfg = _ensure_generation_config(model, settings)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use chat template for proper Qwen3 thinking mode
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate (may take a while on MPS with num_samples > 1)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=cfg,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Get input length in tokens
    input_len = inputs.input_ids.shape[1]

    texts: List[str] = []
    for output in outputs:
        # Extract only the generated tokens (not the prompt)
        output_ids = output[input_len:].tolist()

        # Decode WITHOUT skipping special tokens to preserve <think> tags
        completion = tokenizer.decode(output_ids, skip_special_tokens=False)

        # Clean up any chat template artifacts but keep <think> tags
        # Remove any trailing end tokens like <|im_end|>
        completion = completion.replace("<|im_end|>", "").strip()

        texts.append(completion)
    return texts


def parse_reasoning(text: str) -> ResponseVariant:
    """Split a completion into reasoning and final answer.

    Expected Qwen3 format: <think>{internal_reasoning}</think>{final_answer}

    Handles incomplete generations gracefully (e.g., generation cut off before </think>).
    """
    reasoning = ""
    answer = ""

    text = text.strip()

    # Case 1: Both tags present (complete thinking)
    if THINK_START in text and THINK_END in text:
        try:
            before, rest = text.split(THINK_START, 1)
            think_content, after = rest.split(THINK_END, 1)

            reasoning = f"{THINK_START}\n{think_content.strip()}\n{THINK_END}"
            answer = after.strip()
        except ValueError:
            # Multiple tags or malformed - take first occurrence
            reasoning = f"{THINK_START}\n{text}\n{THINK_END}"

    # Case 2: Only start tag (incomplete generation)
    elif THINK_START in text:
        _, rest = text.split(THINK_START, 1)
        reasoning = f"{THINK_START}\n{rest.strip()}\n{THINK_END}"
        # No final answer (generation was cut off)

    # Case 3: No tags at all (model didn't use thinking mode)
    else:
        reasoning = f"{THINK_START}\n{text}\n{THINK_END}"

    return ResponseVariant(reasoning=reasoning, final_answer=answer)


def generate_on_policy_variants(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    settings: Optional[GenerationSettings] = None,
    num_samples: int = 4,
) -> List[ResponseVariant]:
    """Sample the model to obtain on-policy reasoning traces."""
    raw = run_generation(model, tokenizer, prompt, settings=settings, num_samples=num_samples)
    return [parse_reasoning(chunk) for chunk in raw]


def build_off_policy_variants(
    edited_texts: Iterable[str],
    *,
    ensure_think_tags: bool = False,
) -> List[ResponseVariant]:
    """Convert manually authored off-policy text into response variants."""
    variants: List[ResponseVariant] = []
    for text in edited_texts:
        variant = parse_reasoning(text)
        if ensure_think_tags and THINK_START not in variant.reasoning:
            variant.reasoning = f"{THINK_START}\n{variant.reasoning}\n{THINK_END}"
        variants.append(variant)
    return variants
