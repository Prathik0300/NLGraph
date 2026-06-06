"""
config.py — Single source of truth for all tunable constants.
All other modules import the `settings` singleton.
No business logic lives here.
"""

from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


BASE_DIR = Path(__file__).parent


class OrchestratorConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR:          Path = BASE_DIR / "data"
    MODEL_CACHE_DIR:   Path = BASE_DIR / "data" / "models"
    CENTROID_PATH:     Path = BASE_DIR / "data" / "centroids.npz"
    QDRANT_PATH:       Path = BASE_DIR / "data" / "qdrant"
    SQLITE_PATH:       Path = BASE_DIR / "data" / "nlgraph.db"
    SETFIT_MODEL_DIR:  Path = BASE_DIR / "data" / "models" / "setfit"

    # ── Embedding model ────────────────────────────────────────────────────
    EMBED_MODEL:       str  = "BAAI/bge-m3"
    EMBED_DIM:         int  = 1024
    EMBED_BATCH_SIZE:  int  = 32
    LOCAL_FILES_ONLY:  bool = False          # set True after first download

    # ── Capability projection ──────────────────────────────────────────────
    ACTIVATION_THRESHOLD:     float = 0.55   # global default (per-cap overrides in SQLite)
    AMBIGUITY_THRESHOLD:      float = 0.08   # gap < this → invoke cross-encoder
    PROTOTYPE_K:              int   = 3      # K-means clusters per capability
    CONTRASTIVE_ALPHA:        float = 0.70   # repulsion strength for contrastive centroid adjustment
                                             # prototype += α*(prototype − neg_mean), then re-norm
    IMPLICATION_MIN_SCORE:    float = 0.70   # min projector score to fire implication rules

    IMPLICATION_CONF_EXECUTE_PLAN:     float = 0.85
    IMPLICATION_CONF_VERIFY_EXECUTE:   float = 0.90
    IMPLICATION_CONF_GATHER_ANALYZE:   float = 0.78

    # ── Dependency detection ───────────────────────────────────────────────
    DEP_LEXICAL_WEIGHT:       float = 0.45
    DEP_PARSE_WEIGHT:         float = 0.30
    DEP_CROSSENC_WEIGHT:      float = 0.25
    DEP_MIN_CONFIDENCE:       float = 0.55   # below this → default SEQUENTIAL
    CROSS_ENC_MODEL:          str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Global memory (Qdrant) ─────────────────────────────────────────────
    CACHE_HIT_THRESHOLD:      float = 0.97   # cosine similarity → return cached plan
                                             # (dense-only; RRF scores are NOT used for cache hits
                                             #  because RRF top-1 always scores ~1.0 regardless of
                                             #  actual semantic equivalence)
    AGENT_OUTPUT_MIN_QUALITY: float = 4.0    # only store outputs ≥ this score

    # ── Blackboard (FAISS in-memory) ───────────────────────────────────────
    FAISS_MAX_FACTS:          int   = 10000
    CONTEXT_TOP_K_FACTS:      int   = 8
    CONTEXT_TOKEN_BUDGET:     int   = 3000

    # ── Embed cache ────────────────────────────────────────────────────────
    EMBED_CACHE_MAX_SIZE:     int   = 50000
    EMBED_CACHE_TTL_SECONDS:  int   = 86400  # 24 hours

    # ── DAG validation ────────────────────────────────────────────────────
    MAX_DAG_REPAIR_ATTEMPTS:  int   = 3

    # ── LLM execution (Ollama) ────────────────────────────────────────────
    OLLAMA_BASE_URL:          str   = "http://localhost:11434"
    OLLAMA_LLM_MODEL:         str   = "mistral:7b"
    OLLAMA_FALLBACK_MODEL:    str   = "mistral:7b"
    OLLAMA_TIMEOUT_SECONDS:   int   = 120
    OLLAMA_AUTO_PULL:         bool  = True
    CAPABILITY_MODEL_MAP:     dict  = Field(default_factory=lambda: {
        "gather":      "qwen2.5:14b",
        "analyze":     "deepseek-r1:14b",
        "plan":        "deepseek-r1:14b",
        "execute":     "qwen2.5:14b",
        "verify":      "qwen2.5:14b",
        "refine":      "qwen2.5:14b",
        "communicate": "mistral:7b",
    })

    # ── Phrase splitting ──────────────────────────────────────────────────
    SPACY_MODEL:              str   = "en_core_web_trf"
    MIN_PHRASE_TOKENS:        int   = 3
    MAX_PHRASES:              int   = 20
    LONG_QUERY_THRESHOLD:     int   = 50    # tokens → try NeuralEDUSeg fallback

    # ── Feedback / learning ───────────────────────────────────────────────
    SETFIT_RETRAIN_THRESHOLD: int   = 20    # new labeled examples → retrain
    RECALIB_THRESHOLD:        int   = 50    # low-score entries → recalibrate thresholds

    # ── CLI display ───────────────────────────────────────────────────────
    CLI_SHOW_INTERNALS:       bool  = True  # show projection scores, dep signals, etc.
    CLI_COMPACT_MODE:         bool  = False # hide internal details

    def ensure_dirs(self):
        """Create all data directories if they don't exist."""
        for p in [self.DATA_DIR, self.MODEL_CACHE_DIR,
                  self.QDRANT_PATH, self.SETFIT_MODEL_DIR]:
            p.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = OrchestratorConfig()
