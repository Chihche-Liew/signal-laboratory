#!/usr/bin/env python3
"""Generate the Section 3 framework figure.

Three bands: the human-designed research environment (top), the generation
loop drawn as a closed cycle around the archive (middle), and the three
post-discovery populations with the tests applied to each (bottom). Labels use
paper vocabulary rather than code identifiers. Run from any working directory:

    python paper/scripts/section3_generate_framework_figure.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "signallab-paper-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"

C = {
    "ink": "#23383f",
    "muted": "#64777d",
    "paper": "#ffffff",
    "env": "#eef1f2",      # human-designed environment
    "model": "#dfe9f8",    # model roles
    "code": "#e4f1dc",     # deterministic code
    "state": "#fbeed0",    # durable evidence / state
    "posthoc": "#ece4f5",  # post-discovery tests
    "line": "#5f7279",
    "loop": "#9a6b18",
    "band": "#c8d1d3",
}


def box(ax, x, y, w, h, text, *, color, fs=11.5, lw=1.2, ls="-", weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor=C["ink"], linewidth=lw, linestyle=ls, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=C["ink"], linespacing=1.15, zorder=4, fontweight=weight)


def arrow(ax, a, b, *, color=None, lw=1.3, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, linewidth=lw,
                                 color=color or C["line"], linestyle=ls, shrinkA=2, shrinkB=2,
                                 zorder=2))


def route(ax, pts, *, color=None, lw=1.3, style="-|>"):
    path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle=style, mutation_scale=12, linewidth=lw,
                                 color=color or C["line"], joinstyle="round", capstyle="round",
                                 zorder=2))


def band(ax, y, text):
    ax.text(15.7, y, text, ha="right", va="center", fontsize=10.5, fontweight="bold", color=C["muted"])
    ax.plot([0.3, 15.7], [y - 0.22, y - 0.22], color=C["band"], lw=0.8, zorder=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 8.3))
    fig.patch.set_facecolor(C["paper"])
    ax.set_facecolor(C["paper"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Band 1: human-designed research environment
    # ------------------------------------------------------------------
    band(ax, 8.85, "HUMAN-DESIGNED RESEARCH ENVIRONMENT  (fixed for every hypothesis)")
    ey, eh = 7.38, 1.12
    box(ax, 0.4, ey, 4.6, eh,
        "Hypothesis language\n66 accounting variables · 32 operators\nrecursive composition, validated grammar",
        color=C["env"], fs=10.5)
    box(ax, 5.4, ey, 4.8, eh,
        "Cross-theme tasks\n8 themes · 28 pairs · economic objective\nboth-theme constraint · proposal schema",
        color=C["env"], fs=10.5)
    box(ax, 10.6, ey, 5.1, eh,
        "Empirical protocol\nCRSP-Compustat, July 1963-2024\nNYSE-breakpoint VW quintiles · Fama-MacBeth $t$ (NW 6)\nfactor alpha · Sharpe · coverage",
        color=C["env"], fs=10.0)

    # ------------------------------------------------------------------
    # Band 2: generation loop as a closed cycle
    # ------------------------------------------------------------------
    band(ax, 6.95, "GENERATION LOOP  (per task: 10 generations x 6 proposals; 2-3 model calls per generation)")
    ly, lh = 5.35, 0.95
    xs = [0.4, 3.05, 5.35, 7.65, 9.95, 12.55]
    ws = [2.35, 2.0, 2.0, 2.0, 2.3, 3.05]
    labels = [
        ("Context\nrendering", C["code"]),
        ("Proposer", C["model"]),
        ("Critic", C["model"]),
        ("Revision\n(flagged only)", C["model"]),
        ("Parser +\nvalidator", C["code"]),
        ("Deterministic\nempirical grader", C["code"]),
    ]
    for x, w, (t, col) in zip(xs, ws, labels):
        box(ax, x, ly, w, lh, t, color=col)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + ws[i], ly + lh / 2), (xs[i + 1], ly + lh / 2))

    # environment enters every generation as fixed context (single note, no crossing edges)
    ax.text(8.0, 7.2, "language, task, and protocol enter every generation unchanged",
            ha="center", va="center", fontsize=9.4, color=C["muted"], style="italic")

    # archive on the return lane, centred
    ay, ah = 3.55, 0.95
    box(ax, 4.7, ay, 6.6, ah,
        "Archive of evaluated hypotheses\nformula · predicted sign · $t$ · alpha · Sharpe · coverage · failures\n+ critic notes and validation diagnostics",
        color=C["state"], fs=10.3, lw=1.5)
    # grader -> archive (down and left)
    route(ax, [(14.1, ly), (14.1, ay + ah / 2), (11.3, ay + ah / 2)], color=C["loop"], lw=1.9)
    # archive -> context (left and up)
    route(ax, [(4.7, ay + ah / 2), (1.6, ay + ah / 2), (1.6, ly)], color=C["loop"], lw=1.9)
    ax.text(12.7, ay + ah / 2 + 0.17, "evaluation results", ha="center", fontsize=9.4, color=C["loop"])
    ax.text(3.15, ay + ay * 0 + ah / 2 + 0.17, "evidence for generation g+1", ha="center", fontsize=9.4,
            color=C["loop"])
    # critic also reads the archive (thin dashed up-arrow)
    arrow(ax, (6.35, ay + ah), (6.35, ly), ls="--", lw=1.0)
    ax.text(6.5, 4.95, "critic reads archive", ha="left", va="center", fontsize=8.8,
            color=C["muted"], style="italic")

    # ------------------------------------------------------------------
    # Band 3: three populations of post-discovery inference
    # ------------------------------------------------------------------
    band(ax, 2.85, "POST-DISCOVERY INFERENCE  (outside the loop; one family per run)")
    py, ph = 1.55, 0.85
    pops = [
        (0.4, 4.35, "All evaluated hypotheses\n(search universe)"),
        (5.85, 4.35, "Correct-sign first pass\n$|t|>1.96$ in predicted direction"),
        (11.25, 4.35, "Horse-race survivors\nincremental within run"),
    ]
    for x, w, t in pops:
        box(ax, x, py, w, ph, t, color=C["state"], fs=10.5)
    arrow(ax, (4.75, py + ph / 2), (5.85, py + ph / 2))
    arrow(ax, (10.2, py + ph / 2), (11.25, py + ph / 2))
    ax.text(5.3, py + ph / 2 + 0.17, "screen", ha="center", fontsize=9, color=C["muted"])
    ax.text(10.72, py + ph / 2 + 0.17, "select", ha="center", fontsize=9, color=C["muted"])
    # archive feeds the search universe
    route(ax, [(8.0, ay), (8.0, 3.0), (2.575, 3.0), (2.575, py + ph)], color=C["loop"], lw=1.9)
    ax.text(5.3, 3.12, "completed run", ha="center", fontsize=9.4, color=C["loop"])

    ty, th = 0.35, 0.8
    tests = [
        (0.4, 4.35, "Multiple testing (BHY, Holm, Bonferroni, Bayesian)\nsigned Harvey-Liu double bootstrap"),
        (5.85, 4.35, r"Near-duplicate removal ($|\rho| > 0.95$)" + "\njoint Fama-MacBeth, backward elimination"),
        (11.25, 4.35, "FF5 / FF6 / q-factor alpha · WLS Fama-MacBeth\nsubsample & decay · Chen-Zimmermann spanning"),
    ]
    for x, w, t in tests:
        box(ax, x, ty, w, th, t, color=C["posthoc"], fs=9.3)
        arrow(ax, (x + w / 2, py), (x + w / 2, ty + th))

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------
    handles = [
        Patch(facecolor=C["env"], edgecolor=C["ink"], label="human-designed environment"),
        Patch(facecolor=C["model"], edgecolor=C["ink"], label="model roles"),
        Patch(facecolor=C["code"], edgecolor=C["ink"], label="deterministic code"),
        Patch(facecolor=C["state"], edgecolor=C["ink"], label="evidence / populations"),
        Patch(facecolor=C["posthoc"], edgecolor=C["ink"], label="inference procedures"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=5,
              frameon=False, fontsize=9.2, handlelength=1.4, columnspacing=1.6)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
    fig.savefig(OUT_DIR / "fig0_workflow.pdf", bbox_inches="tight", facecolor=C["paper"])
    fig.savefig(OUT_DIR / "fig0_workflow.png", dpi=220, bbox_inches="tight", facecolor=C["paper"])
    plt.close(fig)


if __name__ == "__main__":
    main()
