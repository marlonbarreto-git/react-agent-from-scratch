"""Main ReAct agent loop."""

from typing import Callable

from react_agent.models import AgentStep, AgentResult
from react_agent.tools import ToolRegistry
from react_agent.parser import parse_llm_output, ParsedAction, ParsedFinal

DEFAULT_MAX_ITERATIONS = 10


class ReActAgent:
    """Agent that follows the ReAct (Reasoning + Acting) pattern.

    Alternates between reasoning (Thought) and acting (Action) steps,
    using an LLM to decide which tool to call and when to produce a
    final answer.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        tools: ToolRegistry,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.llm_fn = llm_fn
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, question: str) -> AgentResult:
        """Execute the ReAct loop for a given question.

        Repeatedly prompts the LLM and executes tool calls until a final
        answer is produced or *max_iterations* is reached.
        """
        steps = []
        prompt = self._build_initial_prompt(question)

        for _ in range(self.max_iterations):
            llm_response = self.llm_fn(prompt)
            parsed = parse_llm_output(llm_response)

            if isinstance(parsed, ParsedFinal):
                return AgentResult(answer=parsed.answer, steps=steps, success=True)

            try:
                observation = self.tools.execute(parsed.action, parsed.action_input)
            except Exception as e:
                observation = f"Error: {e}"

            step = AgentStep(
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input,
                observation=observation,
            )
            steps.append(step)

            prompt += f"\n{llm_response}\nObservation: {observation}\n"

        return AgentResult(
            answer="Max iterations reached",
            steps=steps,
            success=False,
        )

    def _build_initial_prompt(self, question: str) -> str:
        """Build the initial prompt with tool descriptions and the user question."""
        tool_descriptions = self.tools.get_tool_descriptions()
        return (
            f"Answer the following question using the available tools.\n\n"
            f"Tools:\n{tool_descriptions}\n\n"
            f"Use this format:\n"
            f"Thought: reason about what to do\n"
            f"Action: tool_name\n"
            f"Action Input: input for the tool\n"
            f"Observation: tool result\n"
            f"... (repeat as needed)\n"
            f"Thought: I now know the answer\n"
            f"Final Answer: the final answer\n\n"
            f"Question: {question}\n"
        )
