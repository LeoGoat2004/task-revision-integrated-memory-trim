"""Infrastructure layer: IO-bound adapters.

- `infrastructure/llm.py`        : OpenAI chat client with retry
- `infrastructure/embed.py`      : OpenAI embedding client
- `infrastructure/sqlite.py`     : SQLite persistence + migration

This layer is the only place where the project touches the network or the
filesystem. All higher layers depend on the abstractions declared here.
"""
