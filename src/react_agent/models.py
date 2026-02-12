"""Data models for the ReAct agent."""

from dataclasses import dataclass, field


@dataclass
class AgentStep:
    """A single Thought-Action-Observation step in the ReAct loop."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """Final outcome of an agent run, including all intermediate steps."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True
