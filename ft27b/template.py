"""The prompt format, in one place.

Training, evaluation and inference all render prompts through `prompt_text`,
with the system prompt and thinking mode from project.toml. If any of them
built prompts their own way, the model would quietly get worse and no metric
would say why. That's the reason this module exists; there's a test for it.

The tokenizer is passed in; nothing here imports transformers.
"""
from __future__ import annotations

TRUNC_MARK = "\n[... truncated ...]"


def chat_messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_text(tok, system: str, user: str, *, enable_thinking: bool = False) -> str:
    """The rendered prompt, ending where the model's answer begins."""
    return tok.apply_chat_template(
        chat_messages(system, user), tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
