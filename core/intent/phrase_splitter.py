"""
core/intent/phrase_splitter.py — 3-layer phrase splitting pipeline.

Layer 1: spaCy dependency parse  (conj / advcl / mark relations)
Layer 2: Benepar constituency    (VP / SBAR node confirmation)
Layer 3: PDTB connective lexicon (explicit discourse signals, O(1) lookup)

Fallback: sentence-boundary split when the query is very short or very long.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from config import settings
from core.exceptions import PhraseSplitError, SpacyModelError
from core.models import Phrase


# ── Load PDTB lexicon once at module import ───────────────────────────────
_PDTB_PATH = Path(__file__).parent.parent.parent / "setup" / "pdtb_connectives.json"

def _load_pdtb():
    if not _PDTB_PATH.exists():
        return {}
    raw = json.loads(_PDTB_PATH.read_text())
    flat = {}
    for _category, entries in raw.items():
        if _category.startswith("_"):
            continue
        for phrase, meta in entries.items():
            flat[phrase.lower()] = meta
    return flat

_PDTB_LEXICON: dict = _load_pdtb()

# Dependency relations that signal a clause boundary
_SPLIT_DEPS = {"conj", "advcl", "acl"}

# Minimum token length for a phrase to be kept
_MIN_TOKENS = settings.MIN_PHRASE_TOKENS


# ══════════════════════════════════════════════════════════════════════════
# PHRASE SPLITTER
# ══════════════════════════════════════════════════════════════════════════

class PhraseSplitter:
    """
    Splits a natural language query into semantically coherent phrase units.

    Lazy-loads spaCy + benepar on first call so startup is fast.
    """

    def __init__(self):
        self._nlp       = None
        self._has_benepar = False

    # ── Lazy init ─────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(settings.SPACY_MODEL)
        except OSError:
            raise SpacyModelError(
                f"spaCy model '{settings.SPACY_MODEL}' not installed.\n"
                f"Run: python -m spacy download {settings.SPACY_MODEL}"
            )

        # Benepar is optional — conflicts with transformers>=4.36, so may not
        # be installed. The dep-parse layer alone handles the common cases well.
        try:
            import benepar
            if "benepar" not in self._nlp.pipe_names:
                try:
                    self._nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                    self._has_benepar = True
                except Exception:
                    pass
            else:
                self._has_benepar = True
        except Exception:
            self._has_benepar = False

    # ── Public API ────────────────────────────────────────────────────────

    def split(self, query: str):
        """
        Main entry point. Returns an ordered list of Phrase objects.
        Raises PhraseSplitError if splitting produces nothing usable.
        """
        self._ensure_loaded()
        query = query.strip()
        if not query:
            raise PhraseSplitError("Empty query passed to phrase splitter.")

        doc = self._nlp(query)

        # ── Pre-annotate with PDTB lexicon ───────────────────────────────
        pdtb_hits = self._scan_pdtb(query)

        # ── Layer 1: dependency-based boundaries ─────────────────────────
        boundaries = self._dep_boundaries(doc)

        # ── Layer 2: benepar VP/SBAR confirmation ────────────────────────
        if self._has_benepar:
            boundaries = self._benepar_confirm(doc, boundaries)

        # ── Build phrase spans from boundaries ───────────────────────────
        phrases = self._build_phrases(doc, boundaries, pdtb_hits)

        # ── Fallback: sentence split if too few phrases ──────────────────
        if len(phrases) < 2:
            phrases = self._sentence_fallback(doc, pdtb_hits)

        # ── Merge fragments that are too short ───────────────────────────
        phrases = self._merge_short(phrases)

        # ── Final guard ───────────────────────────────────────────────────
        if not phrases:
            raise PhraseSplitError(f"Could not extract any phrases from: '{query}'")

        # Re-index
        for i, p in enumerate(phrases):
            p.phrase_index = i

        return phrases

    # ── Layer 1: dependency parse ─────────────────────────────────────────

    def _dep_boundaries(self, doc):
        """
        Find token indices where a new clause starts.
        Signals: conj / advcl / acl pointing to a verb-headed subtree.
        """
        boundaries: set[int] = set()

        for token in doc:
            if token.dep_ in _SPLIT_DEPS and token.head != token:
                # Only split on VP-level coordination/subordination
                if token.pos_ in ("VERB", "AUX") or any(
                    c.pos_ in ("VERB", "AUX") for c in token.subtree
                ):
                    boundaries.add(token.left_edge.i)

        return sorted(boundaries)

    # ── Layer 2: benepar ─────────────────────────────────────────────────

    def _benepar_confirm(self, doc, dep_boundaries):
        """
        Use constituency parse to add VP/SBAR boundaries missed by dep parse.
        Returns merged + sorted boundary list.
        """
        boundaries = set(dep_boundaries)

        try:
            for sent in doc.sents:
                for token in sent:
                    if hasattr(token._, "parse_string") and token._.parse_string:
                        ps = token._.parse_string
                        # SBAR = subordinate clause — strong boundary signal
                        if ps.startswith("(SBAR") or ps.startswith("(S "):
                            boundaries.add(token.i)
        except Exception:
            pass  # benepar failed on this doc — fall back to dep only

        return sorted(boundaries)

    # ── PDTB lexicon scan ─────────────────────────────────────────────────

    def _scan_pdtb(self, query):
        """
        Scan query text for known PDTB connectives.
        Returns dict: {char_offset: connective_meta}
        """
        hits = {}
        lower = query.lower()
        for phrase, meta in _PDTB_LEXICON.items():
            idx = lower.find(phrase)
            while idx != -1:
                hits[idx] = {"phrase": phrase, **meta}
                idx = lower.find(phrase, idx + 1)
        return hits

    # ── Build phrase spans ────────────────────────────────────────────────

    def _build_phrases(self, doc, boundaries, pdtb_hits):
        """
        Slice the doc into Phrase objects using boundary token indices.
        """
        if not boundaries:
            # Entire query is one phrase
            text = doc.text.strip()
            return [Phrase(
                text=text,
                start_char=0,
                end_char=len(text),
                phrase_index=0,
                connective_signal=self._pdtb_signal_at(0, pdtb_hits),
            )]

        phrases = []
        prev = 0

        for boundary in boundaries:
            token = doc[boundary]
            end = token.left_edge.idx  # char offset of this token's left edge

            chunk = doc.text[prev:end].strip().strip(",;")
            if len(chunk.split()) >= _MIN_TOKENS:
                # Find connective in this chunk's char range
                signal = self._pdtb_signal_at(prev, pdtb_hits)
                dep_rel = _get_dep_at_boundary(doc, boundary)
                phrases.append(Phrase(
                    text=chunk,
                    start_char=prev,
                    end_char=end,
                    phrase_index=len(phrases),
                    connective_signal=signal,
                    dep_relation=dep_rel,
                ))
            prev = token.left_edge.idx

        # Last segment
        tail = doc.text[prev:].strip().strip(",;")
        if len(tail.split()) >= _MIN_TOKENS:
            phrases.append(Phrase(
                text=tail,
                start_char=prev,
                end_char=len(doc.text),
                phrase_index=len(phrases),
            ))

        return phrases

    # ── Sentence-boundary fallback ────────────────────────────────────────

    def _sentence_fallback(self, doc, pdtb_hits):
        """
        Split on sentence boundaries + comma/semicolon for simple queries.
        """
        phrases = []

        # First try spaCy sentence boundaries
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text.split()) >= _MIN_TOKENS:
                phrases.append(Phrase(
                    text=text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    phrase_index=len(phrases),
                    connective_signal=self._pdtb_signal_at(sent.start_char, pdtb_hits),
                ))

        # If still only one, split on commas
        if len(phrases) <= 1:
            raw_parts = re.split(r"\s*[,;]\s*", doc.text)
            phrases = []
            pos = 0
            for part in raw_parts:
                part = part.strip()
                if len(part.split()) >= _MIN_TOKENS:
                    start = doc.text.find(part, pos)
                    phrases.append(Phrase(
                        text=part,
                        start_char=start,
                        end_char=start + len(part),
                        phrase_index=len(phrases),
                    ))
                    pos = start + len(part)

        return phrases

    # ── Merge short fragments ─────────────────────────────────────────────

    def _merge_short(self, phrases):
        """Merge phrases below MIN_PHRASE_TOKENS into their nearest neighbour."""
        if not phrases:
            return phrases

        result = []
        i = 0
        while i < len(phrases):
            p = phrases[i]
            if len(p.text.split()) < _MIN_TOKENS and result:
                # Merge into previous
                prev = result[-1]
                result[-1] = Phrase(
                    text=(prev.text + " " + p.text).strip(),
                    start_char=prev.start_char,
                    end_char=p.end_char,
                    phrase_index=prev.phrase_index,
                    connective_signal=prev.connective_signal or p.connective_signal,
                    dep_relation=prev.dep_relation or p.dep_relation,
                )
            else:
                result.append(p)
            i += 1

        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pdtb_signal_at(self, char_offset, pdtb_hits):
        """Return the PDTB signal type closest to a char offset, if any."""
        for offset, meta in pdtb_hits.items():
            if abs(offset - char_offset) < 40:
                return meta.get("type")
        return None


def _get_dep_at_boundary(doc, token_idx):
    try:
        return doc[token_idx].dep_
    except Exception:
        return None


# ── Module-level singleton ────────────────────────────────────────────────
phrase_splitter = PhraseSplitter()


if __name__ == "__main__":
    queries = [
        "research transformer architectures, understand attention deeply, then implement it",
        "find relevant papers on quantum computing and analyze the key algorithms",
        "build a REST API with authentication, test it thoroughly, and document everything",
        "summarize this document",
        "research the topic, understand it, plan the approach, implement the solution, verify it works, and write it up",
    ]
    splitter = PhraseSplitter()
    for q in queries:
        print(f"\nQuery: {q}")
        try:
            phrases = splitter.split(q)
            for p in phrases:
                sig = f" [{p.connective_signal}]" if p.connective_signal else ""
                dep = f" ({p.dep_relation})" if p.dep_relation else ""
                print(f"  [{p.phrase_index}] {p.text}{sig}{dep}")
        except Exception as e:
            print(f"  ERROR: {e}")
