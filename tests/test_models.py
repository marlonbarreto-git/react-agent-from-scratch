"""Tests for AgentStep and AgentResult dataclasses."""

from dataclasses import fields

import pytest

from react_agent.models import AgentResult, AgentStep


class TestAgentStepInstantiation:
    def test_create_with_all_fields(self):
        step = AgentStep(
            thought="I need to search",
            action="search",
            action_input="python",
            observation="Found results",
        )
        assert step.thought == "I need to search"
        assert step.action == "search"
        assert step.action_input == "python"
        assert step.observation == "Found results"

    def test_positional_arguments(self):
        step = AgentStep("think", "act", "input", "observe")
        assert step.thought == "think"
        assert step.action == "act"
        assert step.action_input == "input"
        assert step.observation == "observe"

    def test_missing_thought_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentStep(action="search", action_input="q", observation="r")

    def test_missing_action_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentStep(thought="t", action_input="q", observation="r")

    def test_missing_action_input_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentStep(thought="t", action="a", observation="r")

    def test_missing_observation_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentStep(thought="t", action="a", action_input="q")

    def test_no_arguments_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentStep()


class TestAgentStepEdgeCases:
    def test_empty_strings(self):
        step = AgentStep(thought="", action="", action_input="", observation="")
        assert step.thought == ""
        assert step.action == ""
        assert step.action_input == ""
        assert step.observation == ""

    def test_multiline_thought(self):
        thought = "Line one\nLine two\nLine three"
        step = AgentStep(thought=thought, action="a", action_input="i", observation="o")
        assert "\n" in step.thought

    def test_long_observation(self):
        obs = "x" * 50_000
        step = AgentStep(thought="t", action="a", action_input="i", observation=obs)
        assert len(step.observation) == 50_000

    def test_unicode_content(self):
        step = AgentStep(
            thought="Necesito buscar",
            action="buscar",
            action_input="consulta",
            observation="resultados",
        )
        assert step.thought == "Necesito buscar"

    def test_special_characters_in_action(self):
        step = AgentStep(
            thought="t",
            action="tool-name_v2",
            action_input='{"key": "value"}',
            observation="done",
        )
        assert step.action == "tool-name_v2"
        assert step.action_input == '{"key": "value"}'


class TestAgentStepMutability:
    def test_thought_can_be_updated(self):
        step = AgentStep(thought="old", action="a", action_input="i", observation="o")
        step.thought = "new"
        assert step.thought == "new"

    def test_observation_can_be_updated(self):
        step = AgentStep(thought="t", action="a", action_input="i", observation="old")
        step.observation = "new"
        assert step.observation == "new"


class TestAgentStepDataclassFeatures:
    def test_has_four_fields(self):
        assert len(fields(AgentStep)) == 4

    def test_field_names(self):
        names = [f.name for f in fields(AgentStep)]
        assert names == ["thought", "action", "action_input", "observation"]

    def test_equality(self):
        a = AgentStep("t", "a", "i", "o")
        b = AgentStep("t", "a", "i", "o")
        assert a == b

    def test_inequality(self):
        a = AgentStep("t1", "a", "i", "o")
        b = AgentStep("t2", "a", "i", "o")
        assert a != b

    def test_repr_contains_class_name(self):
        step = AgentStep("t", "a", "i", "o")
        assert "AgentStep" in repr(step)


class TestAgentResultInstantiation:
    def test_create_with_answer_only(self):
        result = AgentResult(answer="The answer is 42")
        assert result.answer == "The answer is 42"
        assert result.steps == []
        assert result.success is True

    def test_create_with_all_fields(self):
        step = AgentStep("t", "a", "i", "o")
        result = AgentResult(answer="done", steps=[step], success=False)
        assert result.answer == "done"
        assert result.steps == [step]
        assert result.success is False

    def test_default_steps_is_empty_list(self):
        result = AgentResult(answer="test")
        assert result.steps == []
        assert isinstance(result.steps, list)

    def test_default_success_is_true(self):
        result = AgentResult(answer="test")
        assert result.success is True

    def test_missing_answer_raises_type_error(self):
        with pytest.raises(TypeError):
            AgentResult()

    def test_explicit_success_false(self):
        result = AgentResult(answer="failed", success=False)
        assert result.success is False


class TestAgentResultDefaultStepsNotShared:
    def test_default_steps_are_independent(self):
        r1 = AgentResult(answer="a")
        r2 = AgentResult(answer="b")
        r1.steps.append(AgentStep("t", "a", "i", "o"))
        assert len(r2.steps) == 0

    def test_each_instance_gets_new_list(self):
        r1 = AgentResult(answer="a")
        r2 = AgentResult(answer="b")
        assert r1.steps is not r2.steps


class TestAgentResultEdgeCases:
    def test_empty_answer(self):
        result = AgentResult(answer="")
        assert result.answer == ""

    def test_multiline_answer(self):
        answer = "Line 1\nLine 2\nLine 3"
        result = AgentResult(answer=answer)
        assert "\n" in result.answer

    def test_multiple_steps(self):
        steps = [
            AgentStep(f"thought-{i}", f"action-{i}", f"input-{i}", f"obs-{i}")
            for i in range(10)
        ]
        result = AgentResult(answer="final", steps=steps)
        assert len(result.steps) == 10
        assert result.steps[0].thought == "thought-0"
        assert result.steps[9].thought == "thought-9"

    def test_long_answer(self):
        answer = "y" * 100_000
        result = AgentResult(answer=answer)
        assert len(result.answer) == 100_000


class TestAgentResultMutability:
    def test_answer_can_be_updated(self):
        result = AgentResult(answer="old")
        result.answer = "new"
        assert result.answer == "new"

    def test_success_can_be_toggled(self):
        result = AgentResult(answer="test", success=True)
        result.success = False
        assert result.success is False

    def test_steps_can_be_appended(self):
        result = AgentResult(answer="test")
        result.steps.append(AgentStep("t", "a", "i", "o"))
        assert len(result.steps) == 1


class TestAgentResultDataclassFeatures:
    def test_has_three_fields(self):
        assert len(fields(AgentResult)) == 3

    def test_field_names(self):
        names = [f.name for f in fields(AgentResult)]
        assert names == ["answer", "steps", "success"]

    def test_equality_same_values(self):
        a = AgentResult(answer="x", steps=[], success=True)
        b = AgentResult(answer="x", steps=[], success=True)
        assert a == b

    def test_inequality_different_answer(self):
        a = AgentResult(answer="x")
        b = AgentResult(answer="y")
        assert a != b

    def test_inequality_different_success(self):
        a = AgentResult(answer="x", success=True)
        b = AgentResult(answer="x", success=False)
        assert a != b

    def test_inequality_different_steps(self):
        step = AgentStep("t", "a", "i", "o")
        a = AgentResult(answer="x", steps=[step])
        b = AgentResult(answer="x", steps=[])
        assert a != b

    def test_repr_contains_class_name(self):
        result = AgentResult(answer="test")
        assert "AgentResult" in repr(result)
