"""Tool registry for the ReAct agent."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[[str], str]) -> None:
        self._tools[name] = Tool(name=name, description=description, func=func)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, name: str, input: str) -> str:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool.func(input)

    def get_tool_descriptions(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
