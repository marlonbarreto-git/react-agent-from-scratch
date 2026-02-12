"""Tool registry for the ReAct agent."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


class ToolRegistry:
    """Registry that stores tools and dispatches execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[[str], str]) -> None:
        """Register a tool with its name, description, and callable."""
        self._tools[name] = Tool(name=name, description=description, func=func)

    def get(self, name: str) -> Tool | None:
        """Return a tool by name, or ``None`` if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute(self, name: str, tool_input: str) -> str:
        """Execute a tool by name with the given input string."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool.func(tool_input)

    def get_tool_descriptions(self) -> str:
        """Return a formatted multi-line string describing every registered tool."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
