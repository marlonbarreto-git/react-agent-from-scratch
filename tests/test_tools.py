"""Tests for ToolRegistry."""

import pytest

from react_agent.tools import Tool, ToolRegistry


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_tool(self):
        self.registry.register("search", "Search the web", lambda q: f"results for {q}")
        tool = self.registry.get("search")
        assert tool is not None
        assert tool.name == "search"
        assert tool.description == "Search the web"

    def test_get_nonexistent_returns_none(self):
        assert self.registry.get("nonexistent") is None

    def test_list_tools_returns_all_registered(self):
        self.registry.register("search", "Search the web", lambda q: q)
        self.registry.register("calc", "Calculator", lambda q: q)
        tools = self.registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"search", "calc"}

    def test_execute_calls_function(self):
        self.registry.register("calc", "Calculator", lambda q: str(eval(q)))
        result = self.registry.execute("calc", "2 + 2")
        assert result == "4"

    def test_execute_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Tool 'unknown' not found"):
            self.registry.execute("unknown", "input")

    def test_get_tool_descriptions_format(self):
        self.registry.register("search", "Search the web", lambda q: q)
        self.registry.register("calc", "Do math", lambda q: q)
        desc = self.registry.get_tool_descriptions()
        assert "- search: Search the web" in desc
        assert "- calc: Do math" in desc

    def test_register_overwrites_existing(self):
        self.registry.register("search", "Old description", lambda q: "old")
        self.registry.register("search", "New description", lambda q: "new")
        tool = self.registry.get("search")
        assert tool.description == "New description"
        assert tool.func("x") == "new"
        assert len(self.registry.list_tools()) == 1
