"""Tests for the ReAct agent loop."""

from unittest.mock import MagicMock

from react_agent.agent import ReActAgent
from react_agent.tools import ToolRegistry


def _make_registry_with_tool(name="search", description="Search the web", result="tool_result"):
    registry = ToolRegistry()
    registry.register(name, description, lambda x: result)
    return registry


def test_agent_returns_final_answer_immediately():
    llm_fn = MagicMock(return_value="Thought: I know\nFinal Answer: 42")
    registry = _make_registry_with_tool()
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    result = agent.run("What is the answer?")

    assert result.answer == "42"
    assert result.steps == []
    assert result.success is True
    assert llm_fn.call_count == 1


def test_agent_executes_tool_and_gets_answer():
    llm_fn = MagicMock(
        side_effect=[
            "Thought: I need to search\nAction: search\nAction Input: query",
            "Thought: I now know the answer\nFinal Answer: found it",
        ]
    )
    registry = _make_registry_with_tool(result="tool_result")
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    result = agent.run("Find something")

    assert result.answer == "found it"
    assert result.success is True
    assert len(result.steps) == 1
    assert result.steps[0].thought == "I need to search"
    assert result.steps[0].action == "search"
    assert result.steps[0].action_input == "query"
    assert result.steps[0].observation == "tool_result"


def test_agent_max_iterations_reached():
    llm_fn = MagicMock(
        return_value="Thought: still thinking\nAction: search\nAction Input: q"
    )
    registry = _make_registry_with_tool()
    agent = ReActAgent(llm_fn=llm_fn, tools=registry, max_iterations=2)

    result = agent.run("Infinite question")

    assert result.success is False
    assert result.answer == "Max iterations reached"
    assert len(result.steps) == 2


def test_agent_handles_tool_error():
    llm_fn = MagicMock(
        side_effect=[
            "Thought: try unknown tool\nAction: nonexistent_tool\nAction Input: test",
            "Thought: I now know\nFinal Answer: recovered",
        ]
    )
    registry = _make_registry_with_tool()
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    result = agent.run("Try broken tool")

    assert result.success is True
    assert len(result.steps) == 1
    assert result.steps[0].observation.startswith("Error:")


def test_agent_passes_tool_descriptions_in_prompt():
    llm_fn = MagicMock(return_value="Thought: done\nFinal Answer: ok")
    registry = ToolRegistry()
    registry.register("calculator", "Perform math calculations", lambda x: x)
    registry.register("search", "Search the web", lambda x: x)
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    agent.run("test")

    prompt = llm_fn.call_args[0][0]
    assert "calculator" in prompt
    assert "Perform math calculations" in prompt
    assert "search" in prompt
    assert "Search the web" in prompt


def test_agent_accumulates_observation_in_prompt():
    llm_fn = MagicMock(
        side_effect=[
            "Thought: first step\nAction: search\nAction Input: q1",
            "Thought: done\nFinal Answer: answer",
        ]
    )
    registry = _make_registry_with_tool(result="first_observation")
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    agent.run("test accumulation")

    second_call_prompt = llm_fn.call_args_list[1][0][0]
    assert "Observation: first_observation" in second_call_prompt


def test_agent_multiple_steps():
    llm_fn = MagicMock(
        side_effect=[
            "Thought: step 1\nAction: search\nAction Input: q1",
            "Thought: step 2\nAction: search\nAction Input: q2",
            "Thought: step 3\nAction: search\nAction Input: q3",
            "Thought: done\nFinal Answer: final",
        ]
    )
    registry = _make_registry_with_tool()
    agent = ReActAgent(llm_fn=llm_fn, tools=registry)

    result = agent.run("multi-step question")

    assert result.success is True
    assert result.answer == "final"
    assert len(result.steps) == 3
    assert result.steps[0].action_input == "q1"
    assert result.steps[1].action_input == "q2"
    assert result.steps[2].action_input == "q3"
