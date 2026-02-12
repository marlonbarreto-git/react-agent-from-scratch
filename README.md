# ReAct Agent from Scratch

ReAct agent implementation from scratch with Thought-Action-Observation loop and dynamic tool registry.

## Overview

This project implements the ReAct (Reasoning + Acting) pattern from scratch without relying on agent frameworks. The agent follows a structured Thought-Action-Observation loop, using regex-based parsing to extract structured actions from LLM output. It supports dynamic tool registration and automatic prompt construction, running iteratively until it reaches a final answer or hits the maximum iteration limit.

## Architecture

```
Question
    |
    v
+------------------+
|   ReActAgent     |
| (loop controller,|
|  prompt builder) |
+------------------+
    |         ^
    v         |
+--------+  +-------------+
| LLM fn |  | Observation |
+--------+  +-------------+
    |              ^
    v              |
+------------------+
|  parse_llm_output|
| (regex parser)   |
+------------------+
    |
    v
+------------------+
|  ToolRegistry    |
| (register, exec, |
|  descriptions)   |
+------------------+
```

## Features

- Full ReAct loop: Thought -> Action -> Observation -> repeat
- Regex-based LLM output parser for structured extraction
- Dynamic tool registry with description generation
- Pluggable LLM function (any callable)
- Configurable max iterations with graceful termination
- Step-by-step execution trace (AgentStep history)
- Final Answer detection and extraction

## Tech Stack

- Python 3.11+
- Pydantic (data validation)

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/react-agent-from-scratch.git
cd react-agent-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/react_agent/
  __init__.py
  models.py    # AgentStep and AgentResult dataclasses
  tools.py     # Tool dataclass and ToolRegistry
  parser.py    # Regex parser for LLM output
  agent.py     # ReActAgent loop controller
tests/
  test_parser.py
  test_tools.py
  test_agent.py
```

## Testing

```bash
pytest -v --cov=src/react_agent
```

19 tests covering LLM output parsing, tool registry operations, and agent loop execution.

## License

MIT
