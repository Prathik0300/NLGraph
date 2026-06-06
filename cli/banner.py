"""
cli/banner.py — NLGraph ASCII banner and startup display.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align


# ── ASCII art — NLGraph ───────────────────────────────────────────────────
# Font: Big / ANSI Shadow style

NLGRAPH_ASCII = r"""
███╗   ██╗██╗      ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
████╗  ██║██║     ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
██╔██╗ ██║██║     ██║  ███╗██████╔╝███████║██████╔╝███████║
██║╚██╗██║██║     ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝"""

VERSION     = "v0.1.0"
TAGLINE     = "Query Decomposition · Multi-Agent Orchestration"
AUTHOR_LINE = "Vector-native  ·  Framework-agnostic  ·  Self-improving"


def render_banner(console: Console):
    """Render the full NLGraph startup banner."""

    # ── ASCII art with gradient-style coloring ──────────────────────────
    art = Text(NLGRAPH_ASCII)
    art.stylize("bold magenta")

    console.print()
    console.print(Align.center(art))

    # ── Subtitle row ────────────────────────────────────────────────────
    version_text = Text()
    version_text.append(f"  {VERSION}  ", style="bold white on magenta")
    version_text.append(f"  {TAGLINE}  ", style="dim white")

    console.print(Align.center(version_text))
    console.print(Align.center(Text(AUTHOR_LINE, style="dim magenta")))
    console.print()

    # ── Capability legend ────────────────────────────────────────────────
    caps = [
        ("🔍 Gather",      "cyan"),
        ("🧠 Analyze",     "blue"),
        ("📐 Plan",        "magenta"),
        ("⚙️  Execute",    "yellow"),
        ("✅ Verify",      "green"),
        ("✨ Refine",      "bright_cyan"),
        ("📝 Communicate", "white"),
    ]

    legend_parts = []
    for label, color in caps:
        t = Text(f" {label} ", style=f"bold {color}")
        legend_parts.append(t)

    legend = Text("  ").join(legend_parts)
    console.print(Align.center(legend))
    console.print()

    # ── Separator ────────────────────────────────────────────────────────
    console.rule(style="dim magenta")
    console.print()


def render_system_status(
    console:         Console,
    ollama_ok:       bool,
    embed_ok:        bool,
    qdrant_ok:       bool,
    centroids_ok:    bool,
    spacy_ok:        bool,
    ollama_model:    str = "",
    embed_model:     str = "",
):
    """Render system dependency status panel."""

    def status_line(label: str, ok: bool, detail: str = ""):
        t = Text()
        if ok:
            t.append("  ● ", style="bold green")
            t.append(f"{label:<22}", style="white")
            t.append("ready", style="bold green")
        else:
            t.append("  ✗ ", style="bold red")
            t.append(f"{label:<22}", style="white")
            t.append("not available", style="bold red")
        if detail:
            t.append(f"  [{detail}]", style="dim white")
        return t

    lines = [
        status_line("Ollama (LLM)",    ollama_ok,    ollama_model),
        status_line("bge-m3 (Embed)",  embed_ok,     embed_model),
        status_line("Qdrant (Memory)", qdrant_ok,    "embedded"),
        status_line("Centroids",       centroids_ok, "capability vectors"),
        status_line("spaCy (NLP)",     spacy_ok,     "phrase splitting"),
    ]

    content = Text("\n").join(lines)

    panel = Panel(
        content,
        title="[bold cyan]System Status[/bold cyan]",
        border_style="dim cyan",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


def render_ready_prompt(console: Console):
    """Render the 'ready' line before the first query prompt."""
    t = Text()
    t.append("  NLGraph", style="bold magenta")
    t.append(" is ready. ", style="white")
    t.append("Type your query below, or ", style="dim white")
    t.append("/help", style="bold cyan")
    t.append(" for commands.", style="dim white")
    console.print(t)
    console.print()
