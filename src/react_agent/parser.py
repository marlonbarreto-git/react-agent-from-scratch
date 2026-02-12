"""LLM output parser for the ReAct agent."""

import re
from dataclasses import dataclass

ERROR_PREVIEW_CHARS = 100


@dataclass
class ParsedAction:
    """Parsed result when the LLM decides to invoke a tool."""

    thought: str
    action: str
    action_input: str


@dataclass
class ParsedFinal:
    """Parsed result when the LLM produces a final answer."""

    thought: str
    answer: str


def parse_llm_output(text: str) -> ParsedAction | ParsedFinal:
    """Parse raw LLM text into a ``ParsedAction`` or ``ParsedFinal``.

    Raises ``ValueError`` if neither a Final Answer nor an Action can be
    extracted from the text.
    """
    thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return ParsedFinal(thought=thought, answer=final_match.group(1).strip())

    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

    if not action_match:
        raise ValueError(f"Could not parse action from LLM output: {text[:ERROR_PREVIEW_CHARS]}")

    return ParsedAction(
        thought=thought,
        action=action_match.group(1).strip(),
        action_input=input_match.group(1).strip() if input_match else "",
    )
