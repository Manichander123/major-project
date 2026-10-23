"""
scripts/generate_paper_fig5.py
==============================
Publication-quality rendering of Figure 5: Pipeline Evolution & Performance Trajectory.
Generates: outputs/reports/paper_materials/figures/fig5_pipeline_evolution.png

All values are verified against authoritative primary JSON reports:
- Stage 1: 0.5511 (Phase 6D Pilot XGB_satellite_only, N=714, phase6d_model_benchmark.json)
- Stage 2: 0.5058 (Phase 6F Stage 4 Early-Season XGB, N=7,843, phase6f_stage4_benchmark_report.json)
- Stage 3: 0.6174 (Phase 6F Stage 4 Full-Season XGB, N=7,843, phase6f_stage4_benchmark_report.json)
- Stage 4: 0.7069 (Phase 6I Multi-Temporal Unweighted S+T, N=7,843, phase6i_robustness_audit.json)
- Stage 5: 0.7226 (Phase 6I Final Inverse-Weighted S+T, N=7,843, phase6i_robustness_audit.json)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig5_pipeline_evolution.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

stages = [
    "1. Pilot Baseline\n(Phase 6D, N=714)\nSingle-Window",
    "2. Early-Season\n(Phase 6F, N=7,843)\nSingle-Window (34f)",
    "3. Full-Season\n(Phase 6F, N=7,843)\nSingle-Window (34f)",
    "4. Multi-Temporal S+T\n(Phase 6I, N=7,843)\nUnweighted (140f)",
    "5. Final Weighted S+T\n(Phase 6I, N=7,843)\nInv-Weighted (140f)",
]

f1_scores = [0.5511, 0.5058, 0.6174, 0.7069, 0.7226]
colors = ["#78909c", "#42a5f5", "#1e88e5", "#1565c0", "#2e7d32"]

plt.figure(figsize=(10, 6), dpi=300)
ax = plt.subplot(111)

bars = ax.bar(range(len(stages)), f1_scores, color=colors, width=0.55, edgecolor="#263238", linewidth=1.2, zorder=3)

# Highlight cohort transition
ax.axvline(0.5, color="#d32f2f", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
ax.text(0.52, 0.25, "Cohort Expansion (11× scale-up)\nN=714 (Pilot) → N=7,843 (116 Mandals)", 
        color="#b71c1c", fontsize=9, fontweight="bold", verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d32f2f", alpha=0.9))

# Value labels and deltas
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.012, f"{height:.4f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1a237e")

# Progression callouts on full-scale cohort
ax.annotate("+11.16 pp\n(Season Maturity)",
            xy=(2, 0.6174), xytext=(1.5, 0.66),
            ha="center", fontsize=8.5, fontweight="semibold", color="#0d47a1",
            arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.2))

ax.annotate("+8.95 pp\n(4-Window Phenology)",
            xy=(3, 0.7069), xytext=(2.6, 0.75),
            ha="center", fontsize=8.5, fontweight="semibold", color="#0d47a1",
            arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.2))

ax.annotate("+1.57 pp\n(Class Weighting)",
            xy=(4, 0.7226), xytext=(3.7, 0.785),
            ha="center", fontsize=8.5, fontweight="semibold", color="#1b5e20",
            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.2))

ax.set_ylabel("Out-of-Fold Macro-F1 Score", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Figure 5: Stepwise Progression of Model Engineering Across Research Phases", 
             fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9.5, fontweight="medium")
ax.set_ylim(0, 0.85)
ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Footer note on primary evidence
plt.figtext(0.5, -0.02, 
            "Data Provenance: Primary JSON benchmarks (Phases 6D, 6F, 6I). Evaluated under 5-Fold StratifiedGroupKFold by Mandal.", 
            ha="center", fontsize=8.5, style="italic", color="#455a64")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Successfully generated Figure 5 at {OUTPUT_PATH}")
