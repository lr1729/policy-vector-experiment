from __future__ import annotations

import os
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rollouts import RolloutsClient
except ImportError:  # pragma: no cover - optional dependency
    RolloutsClient = None


PARAPHRASE_PROMPT = """You are paraphrasing a model's internal reasoning. Below is numbered reasoning text.

Your task: Rewrite ONLY lines {targets} to have the same meaning but different wording.

Numbered reasoning:
{reasoning}

Return ONLY a JSON object like {{"1": "rewritten text", "2": "rewritten text"}}.
Do NOT include "[1]" prefixes in your rewritten text.
Do NOT add explanatory text before or after the JSON.
"""


@dataclass
class Paraphraser:
    model_name: str
    device_map: str = "auto"
    torch_dtype: Optional[torch.dtype] = torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    provider: Optional[str] = None
    requests_per_minute: int = 100
    device_preference: Optional[str] = None
    def __post_init__(self) -> None:
        if self.model_name.startswith("openrouter:"):
            if RolloutsClient is None:
                raise ImportError(
                    "rollouts package is required for openrouter paraphrasing. Install with `pip install rollouts`."
                )
            self.backend = "openrouter"
            self.api_model = self.model_name.split(":", 1)[1]
            self.provider = self.provider or "deepinfra"
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable is required for openrouter paraphraser")
            provider_cfg = None
            if self.provider:
                provider_cfg = {"only": [self.provider]}
            self.client = RolloutsClient(
                model=self.api_model,
                provider=provider_cfg,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                requests_per_minute=self.requests_per_minute,
                api_key=api_key,
                progress_bar=False,
            )
        else:
            self.backend = "local"
            device = self.device_preference or (
                "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            )
            if device == "cpu":
                device_map = None
            elif device == "mps":
                device_map = {"": "mps"}
            else:
                device_map = "auto"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def rewrite(
        self,
        reasoning_text: str,
        num_lines_to_rewrite: int | Sequence[int],
    ) -> Optional[str]:
        """Rewrite the first N non-empty lines of reasoning.

        Args:
            reasoning_text: Text with <think> tags: "<think>\ncontent\n</think>".
            num_lines_to_rewrite: Either an integer count **or** a non-string
                sequence whose length specifies how many leading non-empty
                lines should be paraphrased.

        Returns:
            Paraphrased text with <think> tags, or None if paraphrasing failed.
        """
        if isinstance(num_lines_to_rewrite, (str, bytes)):
            raise TypeError("num_lines_to_rewrite must not be a string")

        if isinstance(num_lines_to_rewrite, SequenceABC):
            seq = list(num_lines_to_rewrite)
            requested_count = len(seq)
        else:
            requested_count = int(num_lines_to_rewrite)

        if requested_count <= 0:
            return reasoning_text

        # Extract content from <think> tags
        content, has_tags = self._extract_content(reasoning_text)
        lines = content.splitlines()

        # Build mapping: only number non-empty lines
        numbered_lines, line_num_to_idx = self._number_nonempty_lines(lines)

        # We want to paraphrase the first N numbered lines
        target_numbers = list(range(1, min(requested_count + 1, len(line_num_to_idx) + 1)))

        if not target_numbers:
            return reasoning_text  # No non-empty lines to paraphrase

        # Generate paraphrases via API
        targets_str = ", ".join(str(num) for num in target_numbers)
        prompt = PARAPHRASE_PROMPT.format(
            targets=targets_str,
            reasoning="\n".join(numbered_lines),
        )

        try:
            replacements = self._generate_replacements(prompt, target_numbers)
        except Exception:
            return None

        # Apply replacements to original lines
        rewritten_lines = lines[:]
        for line_num, new_text in replacements.items():
            if line_num in line_num_to_idx:
                idx = line_num_to_idx[line_num]
                rewritten_lines[idx] = new_text

        # Validate changes
        changed_indices = {line_num_to_idx[num] for num in target_numbers if num in line_num_to_idx}
        if not self._validate_changes(lines, rewritten_lines, changed_indices):
            return None

        result = "\n".join(rewritten_lines)

        # Re-wrap with tags if they were present
        if has_tags:
            result = f"<think>\n{result}\n</think>"

        return result

    def _extract_content(self, reasoning_text: str) -> tuple[str, bool]:
        """Extract content from <think> tags.

        Returns: (content, has_tags)
        """
        text = reasoning_text.strip()

        if not text.startswith("<think>"):
            return text, False

        try:
            _, rest = text.split("<think>", 1)
            content = rest.split("</think>", 1)[0] if "</think>" in rest else rest
            return content.strip(), True
        except ValueError:
            return text, False

    def _number_nonempty_lines(self, lines: list[str]) -> tuple[list[str], dict[int, int]]:
        """Number only non-empty lines for paraphraser prompt.

        Returns:
            (numbered_lines, line_num_to_idx):
                - numbered_lines: Lines with [1], [2] prefixes on non-empty lines
                - line_num_to_idx: Mapping from line number → original index
        """
        numbered = []
        line_num_to_idx = {}
        counter = 1

        for idx, line in enumerate(lines):
            if line.strip():
                numbered.append(f"[{counter}] {line}")
                line_num_to_idx[counter] = idx
                counter += 1
            else:
                numbered.append(line)

        return numbered, line_num_to_idx

    def _validate_changes(self, original: list[str], rewritten: list[str], changed_indices: set[int]) -> bool:
        """Validate that paraphrasing changed the right lines."""
        if len(original) != len(rewritten):
            return False

        for idx in changed_indices:
            if idx >= len(original):
                return False
            # Changed lines must be different and non-empty
            if not rewritten[idx].strip() or original[idx].strip() == rewritten[idx].strip():
                return False

        # Non-changed lines must stay the same
        for idx in range(len(original)):
            if idx not in changed_indices:
                if original[idx].strip() != rewritten[idx].strip():
                    return False

        return True

    # ------------------------------------------------------------------
    def _generate_replacements(self, prompt: str, targets: Sequence[int]) -> Dict[int, str]:
        if self.backend == "openrouter":
            return self._generate_openrouter(prompt, targets)
        return self._generate_local(prompt, targets)

    def _generate_local(self, prompt: str, targets: Sequence[int]) -> Dict[int, str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_block = decoded[len(prompt):].strip()
        return _parse_json_replacements(json_block, targets)

    def _generate_openrouter(self, prompt: str, targets: Sequence[int]) -> Dict[int, str]:
        system_instruction = (
            "You edit numbered internal reasoning lines for a language model. Rewrite only the requested line"
            " numbers, ensure the tone stays internal (not user-facing), preserve meaning, and respond ONLY"
            " with a JSON object mapping each requested line number (e.g., '3') to its rewritten text."
        )

        prompt_text = (
            f"{system_instruction}\n\n"
            f"Numbered reasoning:\n{prompt}\n\n"
            "Return JSON only."
        )

        # DEBUG: Show what we're sending to API
        print(f"[Paraphraser DEBUG] Sending to API - targets: {targets}", flush=True)
        print(f"[Paraphraser DEBUG] Prompt (first 300 chars): {prompt_text[:300]}", flush=True)

        try:
            responses = self.client.generate(
                prompt_text,
                n_samples=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            )
        except Exception as e:
            print(f"[Paraphraser] OpenRouter API error: {e}")
            raise RuntimeError(f"OpenRouter generation failed: {e}") from e

        if not responses:
            raise RuntimeError("OpenRouter returned no responses for paraphrasing request")
        response = responses[0]
        text = (response.content or response.full or response.reasoning or "").strip()

        if not text:
            raise RuntimeError("OpenRouter response was empty")

        # DEBUG: Show what API returned
        print(f"[Paraphraser DEBUG] API response (first 200 chars): {text[:200]}", flush=True)

        replacements = _parse_json_replacements(text, targets)

        # DEBUG: Show parsed replacements
        print(f"[Paraphraser DEBUG] Parsed replacements: {replacements}", flush=True)

        return replacements


# ----------------------------------------------------------------------
def _enumerate_lines(lines: Sequence[str]) -> (List[str], Dict[int, int]):
    enumerated: List[str] = []
    index_to_number: Dict[int, int] = {}
    counter = 1
    for idx, line in enumerate(lines):
        if line.strip():
            enumerated.append(f"[{counter}] {line}")
            index_to_number[idx] = counter
            counter += 1
        else:
            enumerated.append(line)
    return enumerated, index_to_number


def _validate_rewrite(
    original: Sequence[str], rewritten: Sequence[str], edited_indices: set[int]
) -> bool:
    if len(original) != len(rewritten):
        return False
    for idx, (orig_line, rew_line) in enumerate(zip(original, rewritten)):
        if idx in edited_indices:
            if not rew_line.strip():
                return False
            if orig_line.strip() == rew_line.strip():
                return False
            continue
        if orig_line.strip() != rew_line.strip():
            return False
    return True


def _parse_json_replacements(text: str, targets: Sequence[int]) -> Dict[int, str]:
    import json
    import re

    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        inner = text.strip("`")
        lines = inner.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines)

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse paraphraser JSON: {text}") from exc

    replacements: Dict[int, str] = {}
    for num in targets:
        value = None
        for key in (str(num), f"line_{num}", f"Line {num}"):
            if key in raw:
                value = raw[key]
                break
        if value is None or not isinstance(value, str):
            continue

        # Strip line number tags like "[1] " from the beginning
        value = value.strip()
        value = re.sub(r'^\[\d+\]\s*', '', value)

        replacements[num] = value
    return replacements
