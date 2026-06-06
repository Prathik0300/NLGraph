"""
cli/interactive.py — Main interactive CLI loop for NLGraph.

This is the conversation layer — it manages the full user interaction cycle:
  prompt → decompose → preview → confirm → execute → feedback → loop
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from config import settings
from core.models import ExecutionPreview, FeedbackSubmission, AgentEdit
from core.exceptions import (
    NLGraphError, OllamaUnavailableError, CentroidNotFoundError,
    StaleCentroidError, SpacyModelError,
)
from cli.banner import render_banner, render_system_status, render_ready_prompt
from cli.display import (
    print_query_received, print_phase, print_phase_done, print_phase_warn,
    print_phase_error, print_phrases, print_projection_scores,
    print_dependency_signals, print_execution_plan, print_agent_start,
    print_agent_complete, print_agent_failed, print_agent_skipped,
    print_generation_separator, print_final_result, print_feedback_prompt,
    print_feedback_stored, print_error, print_warning, print_help,
    print_clarification_question, spinner,
)
from cli.theme import NLGRAPH_THEME


# ── Prompt Toolkit style (input line only) ────────────────────────────────
PT_STYLE = PTStyle.from_dict({
    "prompt":         "#cc00cc bold",
    "prompt.query":   "#ffffff",
})


class NLGraphCLI:
    """
    Main CLI class. Owns the console, the prompt session,
    and the orchestrator reference. Everything the user sees
    goes through this class.
    """

    def __init__(self):
        self.console       = Console(theme=NLGRAPH_THEME, highlight=False)
        self.session: PromptSession = PromptSession()
        self.show_internals: bool   = settings.CLI_SHOW_INTERNALS
        self.history: list[str]     = []

        # Orchestrator injected after startup checks
        self._orchestrator = None

    # ──────────────────────────────────────────────────────────────────────
    # STARTUP
    # ──────────────────────────────────────────────────────────────────────

    async def startup(self):
        """
        Run all startup checks. Returns True if the system is ready to
        accept queries, False if a critical dependency is missing.
        """
        self.console.clear()
        render_banner(self.console)

        # ── Check each dependency ────────────────────────────────────────
        ollama_ok    = False
        embed_ok     = False
        qdrant_ok    = False
        centroids_ok = False
        spacy_ok     = False
        ollama_model = settings.OLLAMA_LLM_MODEL
        embed_model  = settings.EMBED_MODEL

        with self.console.status("[bold cyan]Checking system dependencies…[/bold cyan]", spinner="dots"):
            await asyncio.sleep(0.1)   # let the spinner render

            # Ollama
            try:
                from core.execution.llm_providers import OllamaProvider
                provider = OllamaProvider()
                ollama_ok = await provider.health_check()
            except Exception:
                ollama_ok = False

            # Embedding model
            try:
                from core.intent.embedding_engine import EmbeddingEngine
                eng = EmbeddingEngine()
                embed_ok = await eng.health_check()
            except Exception:
                embed_ok = False

            # Qdrant — use the singleton so we don't open a second client on the same path
            try:
                from core.memory.global_store import get_global_store
                store = get_global_store()
                await store.initialize()
                qdrant_ok = True
            except Exception:
                qdrant_ok = False

            # Centroids
            try:
                from core.intent.capability_projector import CentroidStore
                cs = CentroidStore(str(settings.CENTROID_PATH))
                cs.load()
                centroids_ok = True
            except Exception:
                centroids_ok = False

            # spaCy + benepar
            try:
                import spacy
                nlp = spacy.load(settings.SPACY_MODEL)
                spacy_ok = True
            except Exception:
                spacy_ok = False

        render_system_status(
            self.console,
            ollama_ok=ollama_ok,
            embed_ok=embed_ok,
            qdrant_ok=qdrant_ok,
            centroids_ok=centroids_ok,
            spacy_ok=spacy_ok,
            ollama_model=ollama_model,
            embed_model=embed_model,
        )

        # ── Handle missing centroids ─────────────────────────────────────
        if not centroids_ok:
            print_warning(
                self.console,
                "Centroids not found. Run: python -m setup.centroid_builder"
            )

        # ── Critical check: embedding must work ─────────────────────────
        if not embed_ok:
            print_error(
                self.console,
                "Embedding model unavailable",
                f"Run: python -c \"from FlagEmbedding import BGEM3FlagModel; "
                f"BGEM3FlagModel('{settings.EMBED_MODEL}')\" to download the model.",
            )
            return False

        # ── Build orchestrator ───────────────────────────────────────────
        try:
            from core.execution.orchestrator import Orchestrator
            self._orchestrator = Orchestrator()
            await self._orchestrator.initialize()
        except Exception as e:
            print_error(self.console, "Failed to initialize orchestrator", str(e))
            return False

        render_ready_prompt(self.console)
        return True

    # ──────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────

    async def run(self):
        """Main interactive loop. Runs until the user exits."""
        ready = await self.startup()
        if not ready:
            self.console.print("\n[bold red]NLGraph could not start. Fix the issues above and try again.[/bold red]\n")
            return

        while True:
            try:
                raw = await self._prompt()
            except (KeyboardInterrupt, EOFError):
                self._farewell()
                break

            query = raw.strip()
            if not query:
                continue

            # ── Built-in commands ────────────────────────────────────────
            if query.startswith("/"):
                handled = await self._handle_command(query)
                if handled == "quit":
                    self._farewell()
                    break
                continue

            # ── Normal query flow ────────────────────────────────────────
            self.history.append(query)
            await self._query_cycle(query)

    # ──────────────────────────────────────────────────────────────────────
    # QUERY CYCLE
    # ──────────────────────────────────────────────────────────────────────

    async def _query_cycle(self, query: str):
        """
        Full cycle for a single query:
        submit → decompose → preview → confirm → execute → feedback
        """
        print_query_received(self.console, query)

        # ── Phase 1: Decompose ────────────────────────────────────────────
        print_phase(self.console, 1, "Decomposing query…")
        t_start = time.perf_counter()

        try:
            preview: ExecutionPreview = await self._orchestrator.submit(
                query=query,
                show_internals_callback=self._show_internals_callback if self.show_internals else None,
            )
        except NLGraphError as e:
            print_phase_error(self.console, "Decomposition failed", str(e))
            return

        t_decomp = (time.perf_counter() - t_start) * 1000
        method = preview.intent_map.decomposition_method.value
        cache_note = ""
        if preview.intent_map.cache_similarity:
            cache_note = f"cache hit ({preview.intent_map.cache_similarity:.3f})"

        print_phase_done(
            self.console,
            f"Decomposed into {preview.estimated_agent_count} agents",
            f"{t_decomp:.0f}ms  ·  {method}  {cache_note}",
        )

        # ── Phase 2: Show clarification if needed ─────────────────────────
        if preview.clarification_needed:
            print_clarification_question(self.console, preview.clarification_needed)
            answer = await self._prompt_inline("  Your answer: ")
            if answer.strip():
                # Re-submit with clarification appended
                clarified_query = f"{query}. Context: {answer.strip()}"
                await self._query_cycle(clarified_query)
                return

        # ── Phase 3: Show execution plan ─────────────────────────────────
        print_execution_plan(self.console, preview)

        # ── Phase 4: User confirmation ────────────────────────────────────
        confirmed, edits = await self._confirm_plan(preview)
        if not confirmed:
            self.console.print("  [dim white]Cancelled.[/dim white]\n")
            return

        # ── Phase 5: Execute ─────────────────────────────────────────────
        print_phase(self.console, 2, "Executing plan…")
        self.console.print()

        try:
            result = await self._orchestrator.confirm(
                preview_id=preview.preview_id,
                edits=edits,
                agent_start_callback=self._on_agent_start,
                agent_done_callback=self._on_agent_done,
                agent_fail_callback=self._on_agent_fail,
                agent_skip_callback=self._on_agent_skip,
                generation_start_callback=self._on_generation_start,
                question_callback=self._on_blocking_question,
            )
        except NLGraphError as e:
            print_phase_error(self.console, "Execution failed", str(e))
            return

        # ── Phase 6: Show result ─────────────────────────────────────────
        print_final_result(self.console, result)

        # ── Phase 7: Feedback ─────────────────────────────────────────────
        await self._collect_feedback(result.execution_id)

    # ──────────────────────────────────────────────────────────────────────
    # CONFIRM / EDIT PLAN
    # ──────────────────────────────────────────────────────────────────────

    async def _confirm_plan(
        self, preview: ExecutionPreview
    ):
        """
        Ask user to approve, edit, or reject the plan.
        Returns (confirmed, edits_or_None).
        """
        self.console.print("  [bold white]Confirm plan?[/bold white]")
        self.console.print("  [bold green]  y / Enter[/bold green]  [dim white]— run it[/dim white]")
        self.console.print("  [bold yellow]  e[/bold yellow]          [dim white]— edit (remove/add agents)[/dim white]")
        self.console.print("  [bold red]  n[/bold red]          [dim white]— cancel[/dim white]")
        self.console.print()

        answer = await self._prompt_inline("  > ")
        answer = answer.strip().lower()

        if answer in ("n", "no", "cancel"):
            return False, None

        if answer in ("e", "edit"):
            edits = await self._edit_plan(preview)
            return True, edits

        # y / Enter / anything else → run it
        return True, None

    async def _edit_plan(self, preview: ExecutionPreview):
        """
        Edit flow — supports both removing existing agents and adding new ones.

        Remove: enter agent numbers (e.g. "1, 3")
        Add:    enter capability names (e.g. "gather, analyze, verify")
        Valid capability names: gather, analyze, plan, execute, verify, refine, communicate
        """
        from core.models import Capability

        agents  = preview.intent_map.agents
        query   = preview.intent_map.original_query
        edits: list[AgentEdit] = []

        _VALID_CAPS = {c.value for c in Capability}
        _CAP_EMOJIS = {
            "gather": "🔍", "analyze": "🧠", "plan": "📐",
            "execute": "⚡", "verify": "✅", "refine": "🔧", "communicate": "📝",
        }

        self.console.print()
        self.console.print("  [bold cyan]Agents in current plan:[/bold cyan]")
        for i, agent in enumerate(agents, start=1):
            emoji = _CAP_EMOJIS.get(agent.capability.value, "•")
            self.console.print(
                f"    [white][[/white][cyan]{i}[/cyan][white]][/white] "
                f"{agent.role_label}  "
                f"[dim white]{emoji} {agent.capability.value.upper()}[/dim white]"
            )

        # ── Remove step ────────────────────────────────────────────────────
        self.console.print()
        self.console.print(
            "  [dim white]Remove agents by number (e.g. [/dim white][white]1, 3[/white]"
            "[dim white]) or Enter to keep all:[/dim white]"
        )
        raw_remove = (await self._prompt_inline("  remove › ")).strip()
        if raw_remove:
            try:
                indices = [int(x.strip()) - 1 for x in raw_remove.split(",")]
                for idx in indices:
                    if 0 <= idx < len(agents):
                        edits.append(AgentEdit(
                            action="remove",
                            agent_id=agents[idx].agent_id,
                        ))
                        self.console.print(
                            f"  [red]  − Removed:[/red] {agents[idx].role_label} "
                            f"[dim white]({agents[idx].capability.value})[/dim white]"
                        )
            except ValueError:
                print_warning(self.console, "Could not parse removal input — skipping removals.")

        # ── Add step ───────────────────────────────────────────────────────
        self.console.print()
        caps_list = "  ".join(
            f"[cyan]{c}[/cyan]" for c in sorted(_VALID_CAPS)
        )
        self.console.print(
            f"  [dim white]Add capabilities (e.g. [/dim white]"
            f"[white]gather, analyze, verify[/white]"
            f"[dim white]) or Enter to skip:[/dim white]"
        )
        self.console.print(f"  [dim white]  Available: {caps_list}[/dim white]")
        raw_add = (await self._prompt_inline("  add    › ")).strip()
        if raw_add:
            tokens = [t.strip().lower() for t in raw_add.replace(",", " ").split()]
            added_any = False
            for token in tokens:
                if token in _VALID_CAPS:
                    edits.append(AgentEdit(
                        action="add",
                        payload={
                            "capability": token,
                            "query":      query,
                            "objective":  query,
                        },
                    ))
                    emoji = _CAP_EMOJIS.get(token, "•")
                    self.console.print(
                        f"  [green]  + Added:[/green] {emoji} {token.upper()} agent"
                    )
                    added_any = True
                else:
                    print_warning(
                        self.console,
                        f"Unknown capability '{token}' — skipped. "
                        f"Use: {', '.join(sorted(_VALID_CAPS))}"
                    )
            if not added_any and raw_add:
                print_warning(self.console, "No valid capabilities found in input.")

        self.console.print()
        return edits

    # ──────────────────────────────────────────────────────────────────────
    # FEEDBACK
    # ──────────────────────────────────────────────────────────────────────

    async def _collect_feedback(self, execution_id: str):
        print_feedback_prompt(self.console)
        raw = await self._prompt_inline("  > ")
        raw = raw.strip()

        if not raw:
            return

        try:
            score = float(raw)
            if not 1.0 <= score <= 5.0:
                raise ValueError
        except ValueError:
            print_warning(self.console, f"Invalid score '{raw}'. Use 1–5.")
            return

        correction = None
        if score < 3.0:
            self.console.print("  [dim white]Optional: describe what was wrong (or Enter to skip):[/dim white]")
            correction = await self._prompt_inline("  > ")
            correction = correction.strip() or None

        await self._orchestrator.feedback(
            execution_id=execution_id,
            score=score,
            correction=correction,
        )
        print_feedback_stored(self.console, score)

    # ──────────────────────────────────────────────────────────────────────
    # AGENT CALLBACKS (called by orchestrator during execution)
    # ──────────────────────────────────────────────────────────────────────

    def _on_agent_start(self, agent, generation: int):
        print_agent_start(self.console, agent, generation)

    def _on_agent_done(self, agent, output):
        print_agent_complete(self.console, agent, output)

    def _on_agent_fail(self, agent, error: str):
        print_agent_failed(self.console, agent, error)

    def _on_agent_skip(self, agent, reason: str):
        print_agent_skipped(self.console, agent, reason)

    def _on_generation_start(self, gen_num: int, count: int):
        print_generation_separator(self.console, gen_num, count)

    async def _on_blocking_question(self, question: str):
        """Called by orchestrator when a blocking mid-execution question arises."""
        print_clarification_question(self.console, question)
        answer = await self._prompt_inline("  Answer: ")
        return answer.strip()

    def _show_internals_callback(self, event: str, data: dict):
        """Called by orchestrator to show internal pipeline steps."""
        if not self.show_internals:
            return
        if event == "phrases":
            print_phrases(self.console, data["phrases"])
        elif event == "projection":
            print_projection_scores(
                self.console,
                data["scores"],
                data["threshold"],
            )
        elif event == "dependencies":
            print_dependency_signals(self.console, data["edges"])

    # ──────────────────────────────────────────────────────────────────────
    # COMMANDS
    # ──────────────────────────────────────────────────────────────────────

    async def _handle_command(self, cmd: str):
        cmd = cmd.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            return "quit"

        elif cmd == "/help":
            print_help(self.console)

        elif cmd == "/clear":
            self.console.clear()
            render_banner(self.console)

        elif cmd == "/history":
            self.console.print()
            self.console.print("  [bold cyan]Recent queries[/bold cyan]")
            last10 = self.history[-10:]
            for i, q in enumerate(reversed(last10), start=1):
                self.console.print(f"  [dim white]{i:2}.[/dim white]  [white]{q}[/white]")
            self.console.print()

        elif cmd == "/stats":
            await self._show_stats()

        elif cmd == "/internals on":
            self.show_internals = True
            self.console.print("  [green]Internal details ON — projection scores and dependency signals will be shown.[/green]\n")

        elif cmd == "/internals off":
            self.show_internals = False
            self.console.print("  [dim white]Internal details OFF — clean output mode.[/dim white]\n")

        elif cmd == "/setup":
            self.console.print("  [yellow]Run: python -m setup.centroid_builder[/yellow]\n")

        else:
            self.console.print(f"  [bold red]Unknown command:[/bold red] [white]{cmd}[/white]  — type [bold cyan]/help[/bold cyan]")
            self.console.print()

        return None

    async def _show_stats(self):
        try:
            report = await self._orchestrator.get_stats()
            self.console.print()
            self.console.print("  [bold cyan]System stats[/bold cyan]")
            self.console.print(f"  Total queries:    [white]{report.total_queries}[/white]")
            self.console.print(f"  Avg quality:      [{_conf(report.avg_quality_score/5)}]{report.avg_quality_score:.2f}/5[/]")
            self.console.print(f"  Cache hit rate:   [white]{report.cache_hit_rate*100:.1f}%[/white]")
            self.console.print(f"  Intent library:   [white]{report.intent_library_size} entries[/white]")
            self.console.print(f"  Avg latency:      [white]{report.avg_latency_ms:.0f}ms[/white]")
            self.console.print(f"  Feedback count:   [white]{report.feedback_count}[/white]")
            self.console.print()
        except Exception as e:
            print_warning(self.console, f"Could not fetch stats: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # PROMPT HELPERS
    # ──────────────────────────────────────────────────────────────────────

    async def _prompt(self):
        """Main query prompt with magenta NLGraph > prefix."""
        # Use prompt_async() — prompt_toolkit is asyncio-native.
        # The sync session.prompt() wrapped in run_in_executor can return
        # stale buffered input on the second and subsequent calls.
        return await self.session.prompt_async(
            HTML('<ansi fg="magenta"><b>NLGraph</b></ansi> <ansi fg="white">›</ansi> '),
            style=PT_STYLE,
        )

    async def _prompt_inline(self, prompt_text: str):
        """Secondary inline prompt (for confirmations, edits, feedback)."""
        try:
            return await self.session.prompt_async(
                HTML(f'<ansi fg="cyan">{prompt_text}</ansi>'),
                style=PT_STYLE,
            )
        except (KeyboardInterrupt, EOFError):
            return ""

    # ──────────────────────────────────────────────────────────────────────
    # MISC
    # ──────────────────────────────────────────────────────────────────────

    def _farewell(self):
        self.console.print()
        self.console.print("  [bold magenta]NLGraph[/bold magenta] [dim white]— session ended.[/dim white]")
        self.console.print()


def _conf(v: float):
    """Helper for stats color."""
    if v >= 0.8:
        return "bold green"
    elif v >= 0.6:
        return "yellow"
    return "red"
