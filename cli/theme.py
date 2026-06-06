"""
cli/theme.py — Color palette and style constants for all CLI output.
Every color decision is made here. Other modules import from here.
"""

from rich.theme import Theme
from rich.style import Style


# ── Semantic color mapping ─────────────────────────────────────────────────
#   white   → default text, descriptions, body content
#   green   → success, completed, high confidence (≥ 0.80)
#   yellow  → running, in-progress, medium confidence (0.60–0.79), implicit
#   red     → error, failure, low confidence (< 0.60), warnings
#   cyan    → structural info, capability names, section headers
#   magenta → NLGraph brand, query text, special highlights
#   blue    → dependency info, graph edges, memory operations
#   dim     → metadata, timestamps, secondary info

NLGRAPH_THEME = Theme({
    # ── Status states ──────────────────────────────────────────────────
    "success":     "bold green",
    "warning":     "bold yellow",
    "error":       "bold red",
    "info":        "cyan",
    "dim_info":    "dim white",
    "running":     "bold yellow",
    "pending":     "dim white",
    "complete":    "bold green",
    "failed":      "bold red",
    "skipped":     "dim yellow",

    # ── Confidence levels ─────────────────────────────────────────────
    "conf_high":   "bold green",      # ≥ 0.80
    "conf_med":    "yellow",           # 0.60–0.79
    "conf_low":    "bold red",         # < 0.60
    "implicit":    "bold yellow",      # implicitly added capabilities

    # ── Structural elements ───────────────────────────────────────────
    "capability":  "bold cyan",
    "agent_name":  "bold white",
    "agent_id":    "dim white",
    "dependency":  "blue",
    "query_text":  "bold magenta",
    "phrase":      "white",
    "domain":      "dim cyan",

    # ── NLGraph brand ─────────────────────────────────────────────────
    "brand":       "bold magenta",
    "brand_dim":   "magenta",
    "subtitle":    "dim white",

    # ── Section headers ───────────────────────────────────────────────
    "header":      "bold cyan",
    "subheader":   "cyan",
    "separator":   "dim white",

    # ── Memory / internals ────────────────────────────────────────────
    "cache_hit":   "bold green",
    "cache_miss":  "dim white",
    "memory_op":   "dim blue",
    "score":       "white",

    # ── Execution ─────────────────────────────────────────────────────
    "output_text": "white",
    "agent_output":"bright_white",
    "feedback":    "bold magenta",
})


# ── Capability → color mapping ────────────────────────────────────────────
CAPABILITY_COLORS: dict[str, str] = {
    "gather":      "cyan",
    "analyze":     "blue",
    "plan":        "magenta",
    "execute":     "yellow",
    "verify":      "green",
    "refine":      "bright_cyan",
    "communicate": "white",
}

# ── Agent status → color mapping ─────────────────────────────────────────
STATUS_COLORS: dict[str, str] = {
    "pending":  "dim white",
    "running":  "bold yellow",
    "complete": "bold green",
    "failed":   "bold red",
    "skipped":  "dim yellow",
}

STATUS_ICONS: dict[str, str] = {
    "pending":  "○",
    "running":  "◉",
    "complete": "●",
    "failed":   "✗",
    "skipped":  "⊘",
}

# ── Dependency type → display ─────────────────────────────────────────────
DEP_COLORS: dict[str, str] = {
    "sequential":   "blue",
    "prerequisite": "yellow",
    "parallel":     "green",
    "feeds_into":   "cyan",
    "none":         "dim white",
}


def confidence_color(score: float):
    """Return the appropriate color for a confidence score."""
    if score >= 0.80:
        return "bold green"
    elif score >= 0.60:
        return "yellow"
    else:
        return "bold red"


def confidence_bar(score: float, width: int = 10):
    """Return a filled bar string for a confidence score."""
    filled = int(score * width)
    empty  = width - filled
    return "█" * filled + "░" * empty
