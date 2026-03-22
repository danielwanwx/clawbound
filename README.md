# ClawBound

Bounded sparse-context execution engine for AI agents.

## Install

```bash
pip install clawbound
```

With provider support:

```bash
pip install clawbound[providers]
```

## Quick Start

```python
from clawbound import create_engine, EngineConfig, EngineRequest

engine = create_engine(EngineConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="sk-...",
))

response = await engine.run(EngineRequest(message="What is 2+2?"))
print(response.content)
```

## CLI

```bash
clawbound --provider anthropic --model claude-sonnet-4-20250514 --message "hello"
```

## Architecture

```
TaskCompiler → PolicyEngine → PromptBuilder → ExecutionLoop → SessionStore
```

- **TaskCompiler**: Classifies user input → TaskSpec (type, complexity, risk)
- **PolicyEngine**: Resolves TaskSpec → RuntimePolicy (budget, tools, iterations)
- **PromptBuilder**: Budget-aware segment admission + system prompt rendering
- **ExecutionLoop**: Async model interaction with tool calls, signals, action gating
- **SessionStore**: Multi-turn continuity with deterministic compaction

### Providers

| Provider | Adapter | Endpoint |
|----------|---------|----------|
| Anthropic | Native Messages API | `api.anthropic.com` |
| Google Gemini | OpenAI-compatible | `generativelanguage.googleapis.com` |
| MiniMax | OpenAI-compatible | `api.minimax.io` / `api.minimaxi.com` (CN) |

## Development

```bash
uv sync --all-extras
uv run pytest              # 371 tests, ~0.3s
uv run mypy src/clawbound  # strict mode
uv run ruff check src tests
```

## License

MIT
