"""Data models for the ReAct agent."""

from dataclasses import dataclass, field


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True
