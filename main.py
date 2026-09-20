"""
Composition root for PaperLens.

The only place in the codebase that knows about every concrete adapter.
The CLI layer delegates here; nothing in domain/ or application/ imports
a specific library directly.
"""
from __future__ import annotations

from app.interfaces.cli.commands import app

if __name__ == "__main__":
    app()