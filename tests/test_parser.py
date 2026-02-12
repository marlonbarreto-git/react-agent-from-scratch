"""Tests for parse_llm_output."""

import pytest

from react_agent.parser import ParsedAction, ParsedFinal, parse_llm_output


class TestParseLLMOutput:
    def test_parse_action(self):
        text = (
            "Thought: I need to search for Python tutorials\n"
            "Action: search\n"
            "Action Input: python tutorial"
        )
        result = parse_llm_output(text)
        assert isinstance(result, ParsedAction)
        assert result.thought == "I need to search for Python tutorials"
        assert result.action == "search"
        assert result.action_input == "python tutorial"

    def test_parse_final_answer(self):
        text = (
            "Thought: I now know the answer\n"
            "Final Answer: The capital of France is Paris."
        )
        result = parse_llm_output(text)
        assert isinstance(result, ParsedFinal)
        assert result.thought == "I now know the answer"
        assert result.answer == "The capital of France is Paris."

    def test_parse_missing_action_raises(self):
        text = "Thought: I need to think about this"
        with pytest.raises(ValueError, match="Could not parse action"):
            parse_llm_output(text)

    def test_parse_action_without_input(self):
        text = (
            "Thought: Let me check\n"
            "Action: get_time"
        )
        result = parse_llm_output(text)
        assert isinstance(result, ParsedAction)
        assert result.action == "get_time"
        assert result.action_input == ""

    def test_parse_multiline_final_answer(self):
        text = (
            "Thought: I have all the information\n"
            "Final Answer: Here are the results:\n"
            "1. First item\n"
            "2. Second item"
        )
        result = parse_llm_output(text)
        assert isinstance(result, ParsedFinal)
        assert "Here are the results:" in result.answer
        assert "1. First item" in result.answer
        assert "2. Second item" in result.answer
