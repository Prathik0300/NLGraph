"""
setup/centroid_builder.py — Builds and validates multi-prototype centroids.

Run this ONCE before using NLGraph:
    python -m setup.centroid_builder

What it does:
  1. Verifies the embedding engine is available
  2. Embeds all seed phrases (capability + dependency)
  3. Runs K-means per capability → K=3 prototypes each
  4. Validates centroid separation (warns if two capabilities overlap)
  5. Saves centroids.npz with metadata
  6. Prints a calibration report
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.asyncio import tqdm as atqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.models import Capability, DependencyType
from core.intent.embedding_engine import EmbeddingEngine
from setup.capability_seeds import CAPABILITY_SEEDS, CAPABILITY_CONTRASTIVE_SEEDS
from setup.dependency_seeds import DEPENDENCY_SEEDS, DEPENDENCY_CONTRASTIVE_SEEDS

console = Console(highlight=False)


# ══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CapabilityCentroidSet:
    capability:  str
    prototypes:  np.ndarray            # shape (K, DIM)
    seed_count:  int
    k_used:      int
    silhouette:  float
    intra_variance: float


@dataclass
class DependencyCentroid:
    dep_type:   str
    centroid:   np.ndarray             # shape (DIM,)
    seed_count: int


@dataclass
class CentroidValidationReport:
    capability_centroids: list[CapabilityCentroidSet]
    dependency_centroids: list[DependencyCentroid]
    pairwise_similarities: dict[tuple[str, str], float] = field(default_factory=dict)
    problematic_pairs: list[tuple[str, str]] = field(default_factory=list)
    passed: bool = True


# ══════════════════════════════════════════════════════════════════════════
# CENTROID BUILDER
# ══════════════════════════════════════════════════════════════════════════

class CentroidBuilder:

    def __init__(self, embed_engine: EmbeddingEngine):
        self._engine = embed_engine

    # ── Capability centroids ──────────────────────────────────────────────

    async def build_capability_centroids(
        self,
        seeds: dict[str, list[str]],
        contrastive_seeds: dict[str, dict[str, list[str]]] | None = None,
        k: int = settings.PROTOTYPE_K,
        contrastive_alpha: float = settings.CONTRASTIVE_ALPHA,
    ):
        """
        For each capability, embed all seeds and run K-means to produce K prototypes.
        Picks the best K via silhouette score if auto_k=True.

        If contrastive_seeds is provided, each prototype is adjusted via:
            prototype += alpha * (prototype - neg_mean)   then L2-renormalised
        This pushes each prototype away from its negative space, improving
        inter-capability separation without altering the positive seed set.
        """
        results = []
        console.print(f"\n  [bold cyan]Building capability centroids (K={k}, α={contrastive_alpha})[/bold cyan]")
        console.print()

        for cap_val, seed_list in seeds.items():
            console.print(f"  [cyan]{cap_val.upper():<14}[/cyan] embedding {len(seed_list)} seeds…")

            # Embed all seeds
            try:
                embeddings = await self._engine.embed_batch(seed_list)
            except Exception as e:
                console.print(f"  [bold red]  ✗ embedding failed for {cap_val}: {e}[/bold red]")
                continue

            # Auto-select best K via silhouette score
            best_k      = k
            best_score  = -1.0
            best_labels = None
            best_km     = None

            for candidate_k in range(2, min(k + 2, len(seed_list) // 5 + 1)):
                try:
                    km = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
                    labels = km.fit_predict(embeddings)
                    score  = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
                    if score > best_score:
                        best_score  = score
                        best_k      = candidate_k
                        best_labels = labels
                        best_km     = km
                except Exception:
                    continue

            if best_km is None:
                # Fallback: use mean of all seeds as single prototype
                centroid = _l2_norm(embeddings.mean(axis=0))
                prototypes = centroid[np.newaxis, :]
                best_k = 1
                best_score = 0.0
                best_labels = np.zeros(len(embeddings), dtype=int)
            else:
                # L2-normalise each prototype
                prototypes = np.array([
                    _l2_norm(c) for c in best_km.cluster_centers_
                ])

            # ── Contrastive repulsion ──────────────────────────────────────
            # Embed the negative examples for this capability and push each
            # prototype away from their mean in embedding space.
            if contrastive_seeds and cap_val in contrastive_seeds:
                neg_phrases: list[str] = []
                for neg_list in contrastive_seeds[cap_val].values():
                    neg_phrases.extend(neg_list)

                if neg_phrases:
                    try:
                        neg_embeddings = await self._engine.embed_batch(neg_phrases)
                        neg_mean = _l2_norm(neg_embeddings.mean(axis=0))

                        # prototype += α * (prototype − neg_mean), then re-normalise
                        prototypes = np.array([
                            _l2_norm(p + contrastive_alpha * (p - neg_mean))
                            for p in prototypes
                        ])
                        console.print(
                            f"  [dim white]    ↳ contrastive repulsion: "
                            f"{len(neg_phrases)} negatives, α={contrastive_alpha}[/dim white]"
                        )
                    except Exception as e:
                        console.print(
                            f"  [yellow]    ⚠ contrastive repulsion skipped for {cap_val}: {e}[/yellow]"
                        )

            # Intra-cluster variance (avg distance to nearest prototype)
            intra_var = float(np.mean([
                1.0 - float(embeddings[i] @ prototypes[best_labels[i]])
                for i in range(len(embeddings))
            ])) if best_labels is not None else 0.0

            cs = CapabilityCentroidSet(
                capability=cap_val,
                prototypes=prototypes,
                seed_count=len(seed_list),
                k_used=best_k,
                silhouette=best_score,
                intra_variance=intra_var,
            )
            results.append(cs)

            sil_color = "green" if best_score > 0.40 else "yellow" if best_score > 0.25 else "red"
            console.print(
                f"  [bold green]  ✓[/bold green] "
                f"K={best_k}  "
                f"silhouette=[{sil_color}]{best_score:.3f}[/{sil_color}]  "
                f"intra-var={intra_var:.3f}"
            )

        return results

    # ── Dependency centroids ──────────────────────────────────────────────

    async def build_dependency_centroids(
        self,
        seeds: dict[str, list[str]],
        contrastive_seeds: dict[str, dict[str, list[str]]] | None = None,
        contrastive_alpha: float = settings.CONTRASTIVE_ALPHA,
    ):
        """
        Simple mean centroid per dependency type (no clustering needed).

        If contrastive_seeds is provided, applies the same repulsion step as
        for capability centroids: centroid += α*(centroid − neg_mean), re-norm.
        """
        results = []
        console.print(f"\n  [bold cyan]Building dependency centroids (α={contrastive_alpha})[/bold cyan]")
        console.print()

        for dep_val, seed_list in seeds.items():
            console.print(f"  [cyan]{dep_val.upper():<16}[/cyan] embedding {len(seed_list)} seeds…")
            try:
                embeddings = await self._engine.embed_batch(seed_list)
            except Exception as e:
                console.print(f"  [bold red]  ✗ failed: {e}[/bold red]")
                continue

            centroid = _l2_norm(embeddings.mean(axis=0))

            # ── Contrastive repulsion ──────────────────────────────────────
            if contrastive_seeds and dep_val in contrastive_seeds:
                neg_phrases: list[str] = []
                for neg_list in contrastive_seeds[dep_val].values():
                    neg_phrases.extend(neg_list)

                if neg_phrases:
                    try:
                        neg_embeddings = await self._engine.embed_batch(neg_phrases)
                        neg_mean = _l2_norm(neg_embeddings.mean(axis=0))
                        centroid = _l2_norm(centroid + contrastive_alpha * (centroid - neg_mean))
                        console.print(
                            f"  [dim white]    ↳ contrastive repulsion: "
                            f"{len(neg_phrases)} negatives, α={contrastive_alpha}[/dim white]"
                        )
                    except Exception as e:
                        console.print(
                            f"  [yellow]    ⚠ contrastive repulsion skipped for {dep_val}: {e}[/yellow]"
                        )

            results.append(DependencyCentroid(
                dep_type=dep_val,
                centroid=centroid,
                seed_count=len(seed_list),
            ))
            console.print(f"  [bold green]  ✓[/bold green] centroid built from {len(seed_list)} seeds")

        return results

    # ── Validation ───────────────────────────────────────────────────────

    def validate(
        self,
        cap_sets: list[CapabilityCentroidSet],
    ):
        """
        Compute pairwise cosine similarities between all capability centroids
        (using the primary/first prototype of each).
        Flags pairs with similarity > 0.75 as problematic.
        """
        report = CentroidValidationReport(
            capability_centroids=cap_sets,
            dependency_centroids=[],
        )

        # Use first prototype as representative centroid for separation check
        reps: dict[str, np.ndarray] = {
            cs.capability: cs.prototypes[0]
            for cs in cap_sets
        }

        names = list(reps.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sim = float(reps[a] @ reps[b])
                report.pairwise_similarities[(a, b)] = sim
                if sim > 0.75:
                    report.problematic_pairs.append((a, b))
                    report.passed = False

        return report

    # ── Save ─────────────────────────────────────────────────────────────

    def save(
        self,
        cap_sets: list[CapabilityCentroidSet],
        dep_centroids: list[DependencyCentroid],
        path: Path,
    ):
        """
        Save all centroids to a .npz file with metadata.
        Metadata includes model name + build timestamp so stale centroids
        can be detected at load time.
        """
        save_dict: dict[str, np.ndarray] = {}

        # Capability prototypes: key = "cap_{name}_prototypes"
        for cs in cap_sets:
            save_dict[f"cap_{cs.capability}_prototypes"] = cs.prototypes
            save_dict[f"cap_{cs.capability}_meta"] = np.array(
                [cs.seed_count, cs.k_used, cs.silhouette, cs.intra_variance],
                dtype=np.float32,
            )

        # Dependency centroids: key = "dep_{name}"
        for dc in dep_centroids:
            save_dict[f"dep_{dc.dep_type}"] = dc.centroid
            save_dict[f"dep_{dc.dep_type}_meta"] = np.array(
                [dc.seed_count], dtype=np.float32
            )

        # Metadata array (model name encoded as bytes)
        model_bytes = settings.EMBED_MODEL.encode("utf-8")
        save_dict["_model_name"] = np.frombuffer(model_bytes, dtype=np.uint8)
        save_dict["_embed_dim"]  = np.array([settings.EMBED_DIM], dtype=np.int32)
        save_dict["_build_time"] = np.array([time.time()], dtype=np.float64)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **save_dict)
        console.print(f"\n  [bold green]✓ Centroids saved →[/bold green] [white]{path}[/white]")


# ══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════

def print_validation_report(report: CentroidValidationReport):
    console.print()
    console.print("  [bold cyan]Centroid separation validation[/bold cyan]")
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim cyan",
        padding=(0, 1),
    )
    table.add_column("Pair",        width=24)
    table.add_column("Similarity",  width=12)
    table.add_column("Status",      width=14)

    for (a, b), sim in sorted(report.pairwise_similarities.items(), key=lambda x: -x[1]):
        pair = f"{a.upper()} ↔ {b.upper()}"
        if sim > 0.75:
            status = "[bold red]⚠  CLOSE[/bold red]"
        elif sim > 0.60:
            status = "[yellow]moderate[/yellow]"
        else:
            status = "[green]well separated[/green]"
        sim_color = "red" if sim > 0.75 else "yellow" if sim > 0.60 else "green"
        table.add_row(pair, f"[{sim_color}]{sim:.4f}[/{sim_color}]", status)

    console.print(table)

    if report.problematic_pairs:
        console.print()
        console.print("  [bold yellow]⚠  Problematic pairs (similarity > 0.75):[/bold yellow]")
        for a, b in report.problematic_pairs:
            console.print(
                f"    [yellow]• {a.upper()} ↔ {b.upper()}[/yellow]  "
                f"[dim white]→ add contrastive seeds to push them apart[/dim white]"
            )
    else:
        console.print()
        console.print("  [bold green]✓ All capability centroids are well separated.[/bold green]")


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _l2_norm(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

async def run_build():
    settings.ensure_dirs()
    console.print()
    console.rule("[bold magenta]NLGraph — Centroid Builder[/bold magenta]", style="magenta")
    console.print()

    # ── Step 1: Check embedding engine ──────────────────────────────────
    console.print("  [bold cyan][1] Loading embedding engine…[/bold cyan]")
    engine = EmbeddingEngine()
    try:
        await engine.initialize()
        console.print("  [bold green]  ✓ bge-m3 loaded[/bold green]")
    except Exception as e:
        console.print(f"  [bold red]  ✗ Failed to load embedding model: {e}[/bold red]")
        console.print("  [dim white]  Make sure bge-m3 is downloaded:[/dim white]")
        console.print(f"  [yellow]  python -c \"from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('{settings.EMBED_MODEL}')\"[/yellow]")
        sys.exit(1)

    builder = CentroidBuilder(engine)

    # ── Step 2: Build capability centroids ───────────────────────────────
    console.print(f"\n  [bold cyan][2] Building capability centroids…[/bold cyan]")
    cap_sets = await builder.build_capability_centroids(
        CAPABILITY_SEEDS,
        contrastive_seeds=CAPABILITY_CONTRASTIVE_SEEDS,
    )

    # ── Step 3: Build dependency centroids ───────────────────────────────
    console.print(f"\n  [bold cyan][3] Building dependency centroids…[/bold cyan]")
    dep_centroids = await builder.build_dependency_centroids(
        DEPENDENCY_SEEDS,
        contrastive_seeds=DEPENDENCY_CONTRASTIVE_SEEDS,
    )

    # ── Step 4: Validate ─────────────────────────────────────────────────
    console.print(f"\n  [bold cyan][4] Validating centroid separation…[/bold cyan]")
    report = builder.validate(cap_sets)
    report.dependency_centroids = dep_centroids
    print_validation_report(report)

    # ── Step 5: Save ─────────────────────────────────────────────────────
    console.print(f"\n  [bold cyan][5] Saving centroids…[/bold cyan]")
    builder.save(cap_sets, dep_centroids, settings.CENTROID_PATH)

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Build complete[/bold green]", style="green")
    console.print()
    console.print(f"  [white]Capability centroids:[/white]  [bold green]{len(cap_sets)}[/bold green]")
    console.print(f"  [white]Dependency centroids:[/white]  [bold green]{len(dep_centroids)}[/bold green]")
    console.print(f"  [white]Saved to:[/white]              [dim white]{settings.CENTROID_PATH}[/dim white]")
    if not report.passed:
        console.print(f"\n  [bold yellow]⚠  {len(report.problematic_pairs)} centroid pair(s) are too close.[/bold yellow]")
        console.print("  [dim white]  Add contrastive seeds to the problematic capabilities and rebuild.[/dim white]")
    console.print()


if __name__ == "__main__":
    asyncio.run(run_build())
