"""
cli/display.py — All Rich-based display components for NLGraph CLI.

Every piece of output that a user sees goes through a function here.
No raw print() calls in other modules — always call display functions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn,
)
from rich.tree import Tree
from rich.rule import Rule
from rich.align import Align
from rich.live import Live
from rich.padding import Padding

from core.models import (
    IntentMap, Agent, DependencyEdge, ExecutionPreview,
    ExecutionResult, AgentOutput, AgentStatus, Capability,
    DependencyType, Fact,
)
from cli.theme import (
    CAPABILITY_COLORS, STATUS_COLORS, STATUS_ICONS,
    DEP_COLORS, confidence_color, confidence_bar,
)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _conf_text(score: float):
    t = Text()
    bar = confidence_bar(score)
    t.append(f"{bar} ", style=confidence_color(score))
    t.append(f"{score:.2f}", style=confidence_color(score))
    return t


def _cap_text(cap: Capability, bold: bool = True):
    color = CAPABILITY_COLORS.get(cap.value, "white")
    style = f"bold {color}" if bold else color
    t = Text()
    t.append(cap.emoji, style=style)
    t.append(f" {cap.value.upper()}", style=style)
    return t


# ══════════════════════════════════════════════════════════════════════════
# SPINNERS / PROGRESS
# ══════════════════════════════════════════════════════════════════════════

def make_spinner(description: str):
    """Single-task indeterminate spinner."""
    return Progress(
        SpinnerColumn(style="bold magenta"),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    )


def make_step_progress(total_steps: int):
    """Determinate progress bar for multi-step operations."""
    return Progress(
        SpinnerColumn(style="bold magenta"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            bar_width=30,
            style="magenta",
            complete_style="bold green",
            finished_style="bold green",
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


@contextmanager
def spinner(console: Console, message: str):
    """Context manager: show a spinner while code runs inside the block."""
    p = make_spinner(message)
    with p:
        p.add_task(message)
        yield
    console.print(f"  [bold green]✓[/bold green] [white]{message}[/white]")


# ══════════════════════════════════════════════════════════════════════════
# QUERY INPUT
# ══════════════════════════════════════════════════════════════════════════

def print_query_received(console: Console, query: str):
    t = Text()
    t.append("\n  Query  ", style="bold white on magenta")
    t.append("  ", style="white")
    t.append(query, style="bold magenta")
    console.print(t)
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# PHASE HEADERS
# ══════════════════════════════════════════════════════════════════════════

def print_phase(console: Console, phase_num: int, label: str):
    t = Text()
    t.append(f"  [{phase_num}] ", style="bold cyan")
    t.append(label, style="bold white")
    console.print(t)


def print_phase_done(console: Console, label: str, detail: str = ""):
    t = Text()
    t.append("      ✓ ", style="bold green")
    t.append(label, style="green")
    if detail:
        t.append(f"  {detail}", style="dim white")
    console.print(t)


def print_phase_warn(console: Console, label: str, detail: str = ""):
    t = Text()
    t.append("      ⚠ ", style="bold yellow")
    t.append(label, style="yellow")
    if detail:
        t.append(f"  {detail}", style="dim yellow")
    console.print(t)


def print_phase_error(console: Console, label: str, detail: str = ""):
    t = Text()
    t.append("      ✗ ", style="bold red")
    t.append(label, style="red")
    if detail:
        t.append(f"  {detail}", style="dim red")
    console.print(t)


# ══════════════════════════════════════════════════════════════════════════
# DECOMPOSITION INTERNALS
# ══════════════════════════════════════════════════════════════════════════

def print_phrases(console: Console, phrases: list):
    """Show the phrases the splitter identified."""
    console.print()
    console.print("  [bold cyan]Phrases identified[/bold cyan]")
    for i, phrase in enumerate(phrases):
        t = Text()
        t.append(f"    [{i+1}] ", style="dim white")
        t.append(phrase.text, style="white")
        if phrase.connective_signal:
            t.append(f"  ← {phrase.connective_signal}", style="dim cyan")
        if phrase.dep_relation:
            t.append(f"  [{phrase.dep_relation}]", style="dim blue")
        console.print(t)
    console.print()


def print_projection_scores(
    console: Console,
    scores: dict[str, float],
    threshold: float,
):
    """Show all 7 capability projection scores as a mini bar chart."""
    console.print("  [bold cyan]Capability projection[/bold cyan]")
    for cap_val, score in sorted(scores.items(), key=lambda x: -x[1]):
        try:
            cap = Capability(cap_val)
        except ValueError:
            continue
        color   = CAPABILITY_COLORS.get(cap_val, "white")
        bar     = confidence_bar(score, width=12)
        active  = score >= threshold

        t = Text()
        t.append(f"    {cap.emoji} ", style=f"bold {color}")
        t.append(f"{cap_val.upper():<12}", style=f"bold {color}" if active else "dim white")
        t.append(f"  {bar} ", style=f"bold {color}" if active else "dim white")
        t.append(f"{score:.3f}", style=confidence_color(score))
        if active:
            t.append("  ✓ ACTIVE", style="bold green")
        else:
            t.append(f"  (threshold {threshold:.2f})", style="dim white")
        console.print(t)
    console.print()


def print_dependency_signals(
    console: Console,
    edges: list[DependencyEdge],
):
    """Show detected dependency edges with signal breakdown."""
    if not edges:
        return
    console.print("  [bold cyan]Dependency detection[/bold cyan]")
    for edge in edges:
        dep_color = DEP_COLORS.get(edge.dep_type.value, "white")
        t = Text()
        t.append(f"    {edge.from_capability.value.upper():<12}", style="bold cyan")
        t.append(f" {edge.dep_type.arrow} ", style=f"bold {dep_color}")
        t.append(f"{edge.to_capability.value.upper():<12}", style="bold cyan")
        t.append(f"  [{edge.dep_type.value}]", style=dep_color)
        t.append(f"  conf: {edge.confidence:.2f}", style=confidence_color(edge.confidence))
        console.print(t)

        if edge.signal_breakdown:
            for sig, val in edge.signal_breakdown.items():
                sub = Text()
                sub.append(f"             {sig:<14}", style="dim white")
                sub.append(f"{val:.3f}", style="dim cyan")
                console.print(sub)
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# EXECUTION PLAN (the main preview panel)
# ══════════════════════════════════════════════════════════════════════════

def print_execution_plan(console: Console, preview: ExecutionPreview):
    """Render the full execution plan that the user approves."""
    intent = preview.intent_map

    # ── Header ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Execution Plan[/bold cyan]", style="cyan")
    console.print()

    # ── Query summary ────────────────────────────────────────────────────
    t = Text()
    t.append("  Query  ", style="bold white on magenta")
    t.append("  ", style="white")
    t.append(intent.original_query, style="bold magenta")
    console.print(t)

    t2 = Text()
    t2.append(f"  Domain: ", style="dim white")
    t2.append(intent.detected_domain.upper(), style="bold cyan")
    t2.append(f"   Method: ", style="dim white")
    t2.append(intent.decomposition_method.value, style="cyan")
    if intent.cache_similarity:
        t2.append(f"   Cache similarity: ", style="dim white")
        t2.append(f"{intent.cache_similarity:.3f}", style="bold green")
    console.print(t2)
    console.print()

    # ── Agent plan table ─────────────────────────────────────────────────
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#",           width=3,  justify="right")
    table.add_column("Role",        width=20)
    table.add_column("Capability",  width=14)
    table.add_column("Confidence",  width=16)
    table.add_column("Depends on",  width=20)
    table.add_column("Objective",   width=45)

    parallel_groups = intent.parallel_groups()
    gen_map: dict[str, int] = {}
    for gi, gen in enumerate(parallel_groups):
        for a in gen:
            gen_map[a.agent_id] = gi

    for i, agent in enumerate(intent.agents, start=1):
        cap_color = CAPABILITY_COLORS.get(agent.capability.value, "white")
        conf_txt  = f"[{confidence_color(agent.confidence)}]{confidence_bar(agent.confidence, 8)} {agent.confidence:.2f}[/]"

        # depends on → resolve to role labels
        dep_labels = []
        agent_map = {a.agent_id: a for a in intent.agents}
        for dep_id in agent.depends_on:
            if dep_id in agent_map:
                dep_labels.append(agent_map[dep_id].role_label)
        dep_str = ", ".join(dep_labels) if dep_labels else "—"

        # implicit badge
        role_display = agent.role_label
        implicit_badge = ""
        if agent.is_implicit:
            implicit_badge = " [bold yellow]⚑ implicit[/bold yellow]"

        table.add_row(
            f"[dim white]{i}[/dim white]",
            f"[bold white]{role_display}[/bold white]{implicit_badge}",
            f"[bold {cap_color}]{agent.capability.emoji} {agent.capability.value.upper()}[/]",
            conf_txt,
            f"[dim white]{dep_str}[/dim white]",
            f"[white]{agent.objective[:43]}{'…' if len(agent.objective) > 43 else ''}[/white]",
        )

    console.print(Padding(table, (0, 2)))

    # ── Parallel groups note ─────────────────────────────────────────────
    parallel_count = sum(1 for g in parallel_groups if len(g) > 1)
    if parallel_count:
        note = Text()
        note.append(f"  ⚡ ", style="bold yellow")
        note.append(f"{parallel_count} parallel group(s) detected — agents will run simultaneously where possible.", style="yellow")
        console.print(note)

    # ── Implicit additions ───────────────────────────────────────────────
    if intent.implicit_additions:
        console.print()
        console.print("  [bold yellow]⚑ Implicit additions[/bold yellow]  [dim white](system detected these were needed)[/dim white]")
        for imp in intent.implicit_additions:
            t = Text()
            t.append(f"    • {imp.capability.value.upper()}", style="bold yellow")
            t.append(f"  (confidence {imp.confidence:.2f})", style="yellow")
            t.append(f"  — {imp.reason}", style="dim white")
            console.print(t)

    # ── Dependency graph ─────────────────────────────────────────────────
    if intent.edges:
        console.print()
        console.print("  [bold cyan]Dependency graph[/bold cyan]")
        for edge in intent.edges:
            dep_color = DEP_COLORS.get(edge.dep_type.value, "white")
            t = Text()
            t.append(f"    {edge.from_capability.value.upper():<12}", style="cyan")
            t.append(f" {edge.dep_type.arrow} ", style=f"bold {dep_color}")
            t.append(f"{edge.to_capability.value.upper():<12}", style="cyan")
            t.append(f"  {edge.dep_type.value}", style=dep_color)
            t.append(f"  (conf: {edge.confidence:.2f})", style="dim white")
            console.print(t)

    # ── Warnings ─────────────────────────────────────────────────────────
    if preview.warnings:
        console.print()
        for w in preview.warnings:
            console.print(f"  [bold red]⚠  {w}[/bold red]")

    console.print()
    console.rule(style="dim cyan")
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# EXECUTION OUTPUT
# ══════════════════════════════════════════════════════════════════════════

def print_agent_start(console: Console, agent: Agent, gen_num: int):
    cap_color = CAPABILITY_COLORS.get(agent.capability.value, "white")
    t = Text()
    t.append(f"\n  {STATUS_ICONS['running']} ", style="bold yellow")
    t.append(f"[Gen {gen_num}] ", style="dim white")
    t.append(f"{agent.role_label}", style="bold white")
    t.append(f"  [{agent.capability.emoji} {agent.capability.value.upper()}]", style=f"bold {cap_color}")
    t.append(f"  id:{agent.agent_id}", style="dim white")
    console.print(t)
    sub = Text()
    sub.append(f"    ↳ ", style="dim white")
    sub.append(agent.objective, style="white")
    console.print(sub)


def print_agent_complete(
    console: Console,
    agent: Agent,
    output: AgentOutput,
):
    cap_color = CAPABILITY_COLORS.get(agent.capability.value, "white")
    t = Text()
    t.append(f"  {STATUS_ICONS['complete']} ", style="bold green")
    t.append(f"{agent.role_label}", style="bold green")
    t.append(f"  done", style="green")
    t.append(f"  ({output.latency_ms:.0f}ms)", style="dim white")
    console.print(t)

    # Show truncated output preview
    preview_text = output.content[:300].replace("\n", " ")
    if len(output.content) > 300:
        preview_text += "…"

    panel = Panel(
        f"[white]{preview_text}[/white]",
        title=f"[{cap_color}]{agent.capability.emoji} {agent.role_label}[/{cap_color}]",
        border_style=f"dim {cap_color}",
        padding=(0, 1),
    )
    console.print(Padding(panel, (0, 4)))


def print_agent_failed(console: Console, agent: Agent, error: str):
    t = Text()
    t.append(f"  {STATUS_ICONS['failed']} ", style="bold red")
    t.append(f"{agent.role_label}", style="bold red")
    t.append(f"  FAILED", style="bold red")
    t.append(f"  — {error[:80]}", style="dim red")
    console.print(t)


def print_agent_skipped(console: Console, agent: Agent, reason: str):
    t = Text()
    t.append(f"  {STATUS_ICONS['skipped']} ", style="dim yellow")
    t.append(f"{agent.role_label}", style="dim yellow")
    t.append(f"  skipped  — {reason}", style="dim yellow")
    console.print(t)


def print_generation_separator(console: Console, gen_num: int, count: int):
    console.print()
    label = f"  Generation {gen_num}  ·  {count} agent{'s' if count != 1 else ''}"
    if count > 1:
        label += "  [parallel]"
    console.print(f"[dim cyan]{label}[/dim cyan]")
    console.print(f"  [dim cyan]{'─' * 50}[/dim cyan]")


# ══════════════════════════════════════════════════════════════════════════
# FINAL RESULT
# ══════════════════════════════════════════════════════════════════════════

def print_final_result(console: Console, result: ExecutionResult):
    console.print()
    console.rule("[bold green]Execution Complete[/bold green]", style="green")
    console.print()

    # ── Timing and coherence ─────────────────────────────────────────────
    t = Text()
    t.append(f"  Total time: ", style="dim white")
    t.append(f"{result.total_latency_ms/1000:.1f}s", style="bold white")
    t.append(f"   Agents: ", style="dim white")
    t.append(f"{len(result.outputs)}", style="bold white")
    t.append(f"   Coherence: ", style="dim white")
    t.append(f"{result.coherence_score:.2f}", style=confidence_color(result.coherence_score))
    console.print(t)
    console.print()

    # ── Final answer ─────────────────────────────────────────────────────
    panel = Panel(
        f"[bright_white]{result.final_answer}[/bright_white]",
        title="[bold green]Final Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(Padding(panel, (0, 2)))

    # ── Open questions (if any) ───────────────────────────────────────────
    if result.open_questions:
        console.print()
        console.print("  [bold yellow]Open questions not resolved:[/bold yellow]")
        for q in result.open_questions:
            console.print(f"    [yellow]• {q}[/yellow]")

    # ── Contradictions (if any) ──────────────────────────────────────────
    if result.contradictions:
        console.print()
        console.print("  [bold red]Possible contradictions detected:[/bold red]")
        for c in result.contradictions:
            console.print(f"    [red]• {c}[/red]")

    console.print()


# ══════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════

def print_feedback_prompt(console: Console):
    console.print()
    console.rule(style="dim magenta")
    t = Text()
    t.append("  How did this go? ", style="white")
    t.append("[1–5]", style="bold magenta")
    t.append("  or press Enter to skip: ", style="dim white")
    console.print(t)


def print_feedback_stored(console: Console, score: float):
    stars = "★" * int(score) + "☆" * (5 - int(score))
    t = Text()
    t.append("  Feedback stored  ", style="bold green")
    t.append(stars, style="bold yellow")
    t.append(f"  ({score:.1f}/5)", style="dim white")
    console.print(t)
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# ERRORS / WARNINGS
# ══════════════════════════════════════════════════════════════════════════

def print_error(console: Console, title: str, detail: str = ""):
    content = Text()
    content.append(detail, style="red") if detail else None
    panel = Panel(
        content if detail else Text(title, style="bold red"),
        title=f"[bold red]✗  {title}[/bold red]" if detail else "[bold red]✗  Error[/bold red]",
        border_style="red",
        padding=(0, 1),
    )
    console.print(panel)


def print_warning(console: Console, message: str):
    t = Text()
    t.append("  ⚠  ", style="bold yellow")
    t.append(message, style="yellow")
    console.print(t)


def print_clarification_question(console: Console, question: str):
    console.print()
    panel = Panel(
        f"[bold yellow]{question}[/bold yellow]",
        title="[bold cyan]Clarification needed[/bold cyan]",
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(Padding(panel, (0, 2)))


# ══════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════

def print_help(console: Console):
    console.print()
    table = Table(
        show_header=False,
        border_style="dim cyan",
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Command", style="bold cyan", width=20)
    table.add_column("Description", style="white")

    commands = [
        ("/help",         "Show this help"),
        ("/history",      "Show last 10 queries"),
        ("/stats",        "Show system performance stats"),
        ("/clear",        "Clear the screen"),
        ("/internals on", "Show projection scores and dependency signals"),
        ("/internals off","Hide internal details (cleaner output)"),
        ("/setup",        "Re-run centroid setup"),
        ("/quit or /exit","Exit NLGraph"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(Padding(table, (0, 2)))
    console.print()
