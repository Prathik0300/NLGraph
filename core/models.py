"""
core/models.py — Complete Pydantic v2 data contracts for NLGraph.
Every data structure that crosses module boundaries lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════

class Capability(str, Enum):
    GATHER      = "gather"
    ANALYZE     = "analyze"
    PLAN        = "plan"
    EXECUTE     = "execute"
    VERIFY      = "verify"
    REFINE      = "refine"
    COMMUNICATE = "communicate"

    @property
    def emoji(self):
        return {
            "gather":      "🔍",
            "analyze":     "🧠",
            "plan":        "📐",
            "execute":     "⚙️",
            "verify":      "✅",
            "refine":      "✨",
            "communicate": "📝",
        }[self.value]

    @property
    def color(self):
        """Rich color for CLI display."""
        return {
            "gather":      "cyan",
            "analyze":     "blue",
            "plan":        "magenta",
            "execute":     "yellow",
            "verify":      "green",
            "refine":      "bright_cyan",
            "communicate": "white",
        }[self.value]


class DependencyType(str, Enum):
    SEQUENTIAL   = "sequential"
    PREREQUISITE = "prerequisite"
    PARALLEL     = "parallel"
    FEEDS_INTO   = "feeds_into"
    NONE         = "none"

    @property
    def arrow(self):
        return {
            "sequential":   "→",
            "prerequisite": "⟹",
            "parallel":     "∥",
            "feeds_into":   "↣",
            "none":         "·",
        }[self.value]


class AgentStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class DecompositionMethod(str, Enum):
    VECTOR        = "vector"
    CACHE_HIT     = "cache_hit"
    FALLBACK_LINEAR = "fallback_linear"


# ══════════════════════════════════════════════════════════════════════════
# PHRASE SPLITTING
# ══════════════════════════════════════════════════════════════════════════

class Phrase(BaseModel):
    text:               str
    start_char:         int
    end_char:           int
    phrase_index:       int
    connective_signal:Optional[str] = None   # from PDTB lexicon
    dep_relation:Optional[str] = None   # spaCy dep_ that triggered split
    constituency_node:Optional[str] = None   # benepar node type
    confidence:         float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def text_nonempty(cls, v: str):
        if not v.strip():
            raise ValueError("Phrase text must not be empty")
        return v.strip()


# ══════════════════════════════════════════════════════════════════════════
# INTENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════

class DependencyEdge(BaseModel):
    from_capability:     Capability
    to_capability:       Capability
    dep_type:            DependencyType
    confidence:          float = Field(ge=0.0, le=1.0)
    source_phrase_indices: tuple[int, int] = (0, 1)
    signal_breakdown:    dict[str, float] = Field(default_factory=dict)
    # {"lexical": 0.90, "parse": 0.75, "cross_encoder": 0.82}


class ImplicitAddition(BaseModel):
    capability:    Capability
    reason:        str
    confidence:    float = Field(ge=0.0, le=1.0)
    triggered_by:  list[Capability] = Field(default_factory=list)


class Agent(BaseModel):
    agent_id:             str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    capability:           Capability
    role_label:           str
    objective:            str
    depends_on:           list[str] = Field(default_factory=list)   # agent_ids
    can_parallelize_with: list[str] = Field(default_factory=list)
    reads_from_blackboard:list[str] = Field(default_factory=list)
    writes_to_blackboard: list[str] = Field(default_factory=list)
    confidence:           float = Field(ge=0.0, le=1.0)
    is_implicit:          bool  = False
    phrase_indices:       list[int] = Field(default_factory=list)
    llm_model:            str  = ""   # resolved at construction time

    @field_validator("objective")
    @classmethod
    def objective_nonempty(cls, v: str):
        if not v.strip():
            raise ValueError("Agent objective must not be empty")
        return v.strip()


class IntentMap(BaseModel):
    query_id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query:      str
    agents:              list[Agent]
    edges:               list[DependencyEdge]
    activation_scores:   dict[str, float] = Field(default_factory=dict)
    implicit_additions:  list[ImplicitAddition] = Field(default_factory=list)
    total_phrases:       int = 0
    decomposition_method: DecompositionMethod = DecompositionMethod.VECTOR
    cache_similarity:Optional[float] = None
    detected_domain:     str = "general"
    created_at:          datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def agents_nonempty(self):
        if not self.agents:
            raise ValueError("IntentMap must contain at least one agent")
        return self

    def parallel_groups(self):
        """Return agents grouped by their topological generation (parallel groups)."""
        import networkx as nx
        g = nx.DiGraph()
        for agent in self.agents:
            g.add_node(agent.agent_id)
        for agent in self.agents:
            for dep_id in agent.depends_on:
                g.add_edge(dep_id, agent.agent_id)
        agent_map = {a.agent_id: a for a in self.agents}
        return [
            [agent_map[aid] for aid in gen if aid in agent_map]
            for gen in nx.topological_generations(g)
        ]


# ══════════════════════════════════════════════════════════════════════════
# BLACKBOARD
# ══════════════════════════════════════════════════════════════════════════

class Fact(BaseModel):
    fact_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    content:      str
    source_agent: str
    confidence:   float = Field(ge=0.0, le=1.0)
    timestamp:    datetime = Field(default_factory=datetime.utcnow)
    embedding:Optional[list[float]] = None   # stored as list; np.ndarray in memory

    @field_validator("content")
    @classmethod
    def content_nonempty(cls, v: str):
        if not v.strip():
            raise ValueError("Fact content must not be empty")
        return v.strip()


class Decision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content:     str
    made_by:     str        # agent_id
    rationale:   str
    timestamp:   datetime = Field(default_factory=datetime.utcnow)


class Artifact(BaseModel):
    artifact_id:  str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_key: str                   # matches Agent.writes_to_blackboard entry
    content:      str
    produced_by:  str                   # agent_id
    format:       str = "text"          # "text" | "json" | "code" | "table"
    timestamp:    datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    agent_id:     str
    status:       AgentStatus = AgentStatus.PENDING
    started_at:Optional[datetime] = None
    completed_at:Optional[datetime] = None
    error:Optional[str] = None
    output_artifact_id:Optional[str] = None
    latency_ms:Optional[float] = None


class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level:     str = "info"     # "info" | "warning" | "error" | "debug"
    agent_id:Optional[str] = None
    event:     str
    details:   dict[str, Any] = Field(default_factory=dict)


class BlackboardSnapshot(BaseModel):
    """Serializable snapshot of a Blackboard — no FAISS, no locks."""
    query_id:       str
    original_query: str
    intent_map:     IntentMap
    facts:          list[Fact]
    decisions:      list[Decision]
    artifacts:      list[Artifact]
    open_questions: list[str]
    agent_states:   dict[str, AgentState]
    execution_log:  list[LogEntry]
    created_at:     datetime = Field(default_factory=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════════════
# AGENT EXECUTION
# ══════════════════════════════════════════════════════════════════════════

class AgentContext(BaseModel):
    agent_id:            str
    objective:           str
    relevant_facts:      list[Fact] = Field(default_factory=list)
    upstream_artifacts:  list[Artifact] = Field(default_factory=list)
    global_examples:     list[str] = Field(default_factory=list)
    open_questions:      list[str] = Field(default_factory=list)
    token_count_estimate: int = 0


class AgentOutput(BaseModel):
    agent_id:     str
    capability:   Capability
    content:      str
    quality_score:Optional[float] = None
    latency_ms:   float
    artifact_id:  str


# ══════════════════════════════════════════════════════════════════════════
# EXECUTION PREVIEW & RESULT
# ══════════════════════════════════════════════════════════════════════════

class ExecutionPreview(BaseModel):
    preview_id:               str = Field(default_factory=lambda: str(uuid.uuid4()))
    query:                    str
    intent_map:               IntentMap
    estimated_agent_count:    int
    estimated_parallel_groups: int
    confidence_summary:       dict[str, float] = Field(default_factory=dict)
    warnings:                 list[str] = Field(default_factory=list)
    clarification_needed:Optional[str] = None
    expires_at:               datetime = Field(default_factory=datetime.utcnow)


class AgentEdit(BaseModel):
    action:    str   # "add" | "remove" | "modify_objective" | "change_dependency"
    agent_id:Optional[str] = None
    payload:   dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    execution_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    preview_id:      str
    query:           str
    outputs:         list[AgentOutput]
    final_answer:    str
    coherence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    contradictions:  list[str] = Field(default_factory=list)
    open_questions:  list[str] = Field(default_factory=list)
    total_latency_ms: float
    completed_at:    datetime = Field(default_factory=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════

class FeedbackSubmission(BaseModel):
    execution_id:    str
    score:           float = Field(ge=1.0, le=5.0)
    correction:Optional[str] = None
    corrected_plan:Optional[IntentMap] = None
    correction_type: str = "confirm"   # "confirm" | "edit" | "reject"


# ══════════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORTING
# ══════════════════════════════════════════════════════════════════════════

class CapabilityStats(BaseModel):
    capability:       str
    avg_quality:      float
    avg_latency_ms:   float
    invocation_count: int
    top_domains:      list[str] = Field(default_factory=list)


class GlobalPerformanceReport(BaseModel):
    total_queries:      int
    avg_quality_score:  float
    capability_stats:   dict[str, CapabilityStats]
    cache_hit_rate:     float
    avg_latency_ms:     float
    feedback_count:     int
    intent_library_size: int
    last_calibration:Optional[datetime] = None


# ══════════════════════════════════════════════════════════════════════════
# HYBRID VECTOR (bge-m3 output)
# ══════════════════════════════════════════════════════════════════════════

class HybridVector(BaseModel):
    dense:  list[float]                 # 1024-dim
    sparse: dict[str, float]            # token_id → weight (non-zero only)

    class Config:
        arbitrary_types_allowed = True
