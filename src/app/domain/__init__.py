"""Domain layer (pure business logic; no IO).

Cross-cutting rules:
- Functions in this layer MUST be pure (no network, no filesystem, no globals).
- All prompts live in `prompts.py`; if you add a prompt, write it in English.
- Type hints are mandatory on every public function.
"""
