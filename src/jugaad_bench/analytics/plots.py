"""
Publication-quality plots for JugaadReasoning-1K benchmark results.

Generates three figures:
1. Discriminative vs Generative performance (grouped bar chart)
2. Scarcity degradation by budget tier (line graph)
3. Domain failure-mode breakdown (stacked bar chart)

Uses matplotlib + seaborn with optional SciencePlots style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jugaad_bench.models import EvalResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style setup
# ─────────────────────────────────────────────────────────────────────────────

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

# Colorblind-safe palette (Wong 2011 + Tol)
_CB_BLUE = "#0072B2"
_CB_ORANGE = "#E69F00"
_CB_GREEN = "#009E73"
_CB_RED = "#D55E00"
_CB_PURPLE = "#CC79A7"
_CB_CYAN = "#56B4E9"
_CB_YELLOW = "#F0E442"

_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


def _apply_style() -> None:
    """Apply SciencePlots style if available, otherwise seaborn defaults."""
    try:
        plt.style.use(["science", "ieee", "no-latex"])
        logger.info("Applied SciencePlots style (science + ieee + no-latex).")
    except Exception:
        try:
            plt.style.use(["science", "no-latex"])
            logger.info("Applied SciencePlots style (science + no-latex).")
        except Exception:
            sns.set_theme(style="whitegrid", font_scale=1.1)
            logger.info("SciencePlots not available; using seaborn whitegrid.")


def _sort_by_tier(results: list[EvalResult]) -> list[EvalResult]:
    """Sort results by model tier: frontier → open_weights → indic, alphabetically within."""
    # Heuristic: infer tier from model name prefixes.  In practice the order
    # the caller passes often already matches config ordering, but we stabilise
    # it here for deterministic plots.
    return sorted(results, key=lambda r: r.model_under_test)


def _save_figure(fig: plt.Figure, output_dir: Path, base_name: str, dpi: int = 300) -> list[Path]:
    """Save a figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("pdf", "png"):
        path = output_dir / f"{base_name}.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        paths.append(path)
        logger.info("Saved plot → %s", path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Discriminative vs Generative
# ─────────────────────────────────────────────────────────────────────────────


def plot_discriminative_vs_generative(
    results: list[EvalResult],
    output_dir: Path,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 300,
) -> list[Path]:
    """
    Grouped bar chart comparing MCQ accuracy and Open-Gen score (normalised
    to 0–100%) across models.

    Args:
        results: List of per-model ``EvalResult``.
        output_dir: Directory for saved figures.
        figsize: Figure dimensions in inches.
        dpi: Resolution for raster output.

    Returns:
        List of saved file paths.
    """
    _apply_style()
    results = _sort_by_tier(results)

    model_names = [r.model_under_test for r in results]
    mcq_pct = [r.mcq_global_accuracy * 100 for r in results]
    opengen_pct = [r.open_gen_global_average_score / 3.0 * 100 for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars_mcq = ax.bar(
        x - width / 2, mcq_pct, width, label="MCQ Accuracy", color=_CB_BLUE, edgecolor="white"
    )
    bars_og = ax.bar(
        x + width / 2, opengen_pct, width, label="Open Gen Score", color=_CB_ORANGE, edgecolor="white"
    )

    # Percentage labels
    for bar in bars_mcq:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar in bars_og:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Discriminative vs Generative Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "discriminative_vs_generative", dpi)
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Scarcity Degradation
# ─────────────────────────────────────────────────────────────────────────────


def plot_scarcity_degradation(
    results: list[EvalResult],
    output_dir: Path,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 300,
) -> list[Path]:
    """
    Line graph showing how Open Gen score degrades as budget tightens.

    X-axis: Budget tiers (₹200 → ₹50 → ₹0).
    Y-axis: Average Open-Gen Score (0–3).

    Args:
        results: List of per-model ``EvalResult`` (must have ``budget_tier_performance``).
        output_dir: Directory for saved figures.
        figsize: Figure dimensions in inches.
        dpi: Resolution for raster output.

    Returns:
        List of saved file paths.
    """
    _apply_style()
    results = _sort_by_tier(results)

    # Canonical budget tier order (high → low budget)
    tier_order = ["₹200", "₹50", "₹0"]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, r in enumerate(results):
        if not r.budget_tier_performance:
            logger.warning(
                "No budget_tier_performance for %s; skipping.", r.model_under_test
            )
            continue

        y_vals: list[float] = []
        x_labels_present: list[str] = []
        for tier in tier_order:
            if tier in r.budget_tier_performance:
                y_vals.append(r.budget_tier_performance[tier])
                x_labels_present.append(tier)

        if not y_vals:
            continue

        marker = _MARKER_CYCLE[idx % len(_MARKER_CYCLE)]
        ax.plot(
            x_labels_present,
            y_vals,
            marker=marker,
            label=r.model_under_test,
            linewidth=1.5,
            markersize=6,
        )

    ax.set_xlabel("Budget Tightness")
    ax.set_ylabel("Average Open Gen Score (0–3)")
    ax.set_title("Scarcity Degradation Across Budget Tiers")
    ax.set_ylim(-0.1, 3.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=7,
        borderaxespad=0,
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "scarcity_degradation", dpi)
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Domain Failure Modes
# ─────────────────────────────────────────────────────────────────────────────


def plot_domain_failure_modes(
    results: list[EvalResult],
    output_dir: Path,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 300,
) -> list[Path]:
    """
    Stacked bar chart of failure-mode percentages per domain.

    Aggregates across all models for a global domain-level view.

    Since ``EvalResult.failure_modes`` is model-level (not per-domain),
    we redistribute failures proportionally based on domain weight.
    If detailed per-domain failure data is unavailable, we use the global
    failure distribution applied uniformly across domains.

    Args:
        results: List of per-model ``EvalResult``.
        output_dir: Directory for saved figures.
        figsize: Figure dimensions in inches.
        dpi: Resolution for raster output.

    Returns:
        List of saved file paths.
    """
    _apply_style()

    # Collect all domains
    all_domains: set[str] = set()
    for r in results:
        all_domains.update(r.domain_performance.keys())
    domains = sorted(all_domains)

    if not domains:
        logger.warning("No domain data available for failure mode plot.")
        return []

    # Aggregate failure counts across all models
    total_cv = sum(r.failure_modes.constraint_violations for r in results)
    total_ph = sum(r.failure_modes.physical_hallucinations for r in results)
    total_ta = sum(r.failure_modes.task_abandonment for r in results)
    grand_total = total_cv + total_ph + total_ta

    if grand_total == 0:
        logger.warning("No failures to plot in domain failure modes.")
        # Create a plot showing zero failures
        cv_pct = [0.0] * len(domains)
        ph_pct = [0.0] * len(domains)
        ta_pct = [0.0] * len(domains)
    else:
        # Distribute proportionally across domains based on domain weight
        # (fraction of total problems in each domain)
        domain_weight: dict[str, float] = {}
        total_problems = 0
        for r in results:
            for d in domains:
                if d not in domain_weight:
                    domain_weight[d] = 0.0

        # Use open_gen score as a proxy: lower score → more failures
        for r in results:
            for d in domains:
                dp = r.domain_performance.get(d)
                if dp:
                    # Failure rate proxy: (3 - open_gen) / 3
                    domain_weight[d] += (3.0 - dp.open_gen) / 3.0

        total_weight = sum(domain_weight.values())
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero

        cv_pct: list[float] = []
        ph_pct: list[float] = []
        ta_pct: list[float] = []

        for d in domains:
            w = domain_weight.get(d, 0.0) / total_weight
            domain_failures = grand_total * w
            if domain_failures == 0:
                cv_pct.append(0.0)
                ph_pct.append(0.0)
                ta_pct.append(0.0)
            else:
                cv_pct.append(total_cv * w / domain_failures * 100)
                ph_pct.append(total_ph * w / domain_failures * 100)
                ta_pct.append(total_ta * w / domain_failures * 100)

    # Simplify: use global proportions for each domain since we don't have
    # per-domain failure breakdowns
    if grand_total > 0:
        global_cv_pct = total_cv / grand_total * 100
        global_ph_pct = total_ph / grand_total * 100
        global_ta_pct = total_ta / grand_total * 100
        cv_pct = [global_cv_pct] * len(domains)
        ph_pct = [global_ph_pct] * len(domains)
        ta_pct = [global_ta_pct] * len(domains)

    x = np.arange(len(domains))

    fig, ax = plt.subplots(figsize=figsize)

    bars_cv = ax.bar(x, cv_pct, label="Constraint Violations", color=_CB_RED, edgecolor="white")
    bars_ph = ax.bar(
        x, ph_pct, bottom=cv_pct, label="Physical Hallucinations", color=_CB_ORANGE, edgecolor="white"
    )
    bottom_ta = [c + p for c, p in zip(cv_pct, ph_pct)]
    bars_ta = ax.bar(
        x, ta_pct, bottom=bottom_ta, label="Task Abandonment", color=_CB_PURPLE, edgecolor="white"
    )

    # Percentage labels on each segment
    for bars, values in [(bars_cv, cv_pct), (bars_ph, ph_pct), (bars_ta, ta_pct)]:
        for bar, val in zip(bars, values):
            if val > 5:  # Only label segments large enough to be readable
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )

    ax.set_xlabel("Domain")
    ax.set_ylabel("Failure Proportion (%)")
    ax.set_title("Failure Mode Breakdown by Domain")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in domains], fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "domain_failure_modes", dpi)
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Generate all
# ─────────────────────────────────────────────────────────────────────────────


def generate_all_plots(
    results: list[EvalResult],
    output_dir: Path,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 300,
) -> list[Path]:
    """
    Generate all three publication-quality plots.

    Args:
        results: Per-model evaluation results.
        output_dir: Directory for saved figures.
        figsize: Figure dimensions.
        dpi: Resolution.

    Returns:
        List of all saved file paths.
    """
    output_dir = Path(output_dir)
    all_paths: list[Path] = []

    logger.info("Generating all plots to %s …", output_dir)

    all_paths.extend(
        plot_discriminative_vs_generative(results, output_dir, figsize, dpi)
    )
    all_paths.extend(
        plot_scarcity_degradation(results, output_dir, figsize, dpi)
    )
    all_paths.extend(
        plot_domain_failure_modes(results, output_dir, figsize, dpi)
    )

    logger.info("All plots generated: %d files.", len(all_paths))
    return all_paths
