"""
cli/__main__.py — Entry point for the NLGraph CLI.

Run with:
    python -m cli
    nlgraph              (after pip install -e .)
"""

from __future__ import annotations

import asyncio
import sys


def main():
    """Entry point registered in pyproject.toml [project.scripts]."""
    # Ensure data directories exist
    try:
        from config import settings
        settings.ensure_dirs()
    except Exception as e:
        print(f"[startup] Could not initialise data directories: {e}")
        sys.exit(1)

    from cli.interactive import NLGraphCLI
    cli = NLGraphCLI()

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
