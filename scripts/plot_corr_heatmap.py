"""
============================================================================
  UQ vs. Performance — Correlation Heatmap Grid (Per Model)
============================================================================

For each of the 15 COD-selected models, compute the correlation between
UQ metrics and performance metrics across the 36 SDP datasets, then
display as a grid of heatmaps (4 columns × 4 rows).

Three correlation methods are generated:
  1. Spearman  (rank-based, robust to outliers)
  2. Kendall   (rank-based, conservative)
  3. Pearson   (linear)

Data:    benchmark_results_IVDP_FullUQ.csv
Output:  heatmap_All_Models_Spearman.pdf
         heatmap_All_Models_Kendall.pdf
         heatmap_All_Models_Pearson.pdf
============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, kendalltau
import math
import os
import sys

# Project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')

# =========================================================================
#  Publication aesthetics
# =========================================================================
sns.set(style="white", context="talk")

plt.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'stix',
    'axes.unicode_minus': False,
})


# =========================================================================
#  Core plotting function
# =========================================================================
def plot_grid_heatmap(df, method_name, method_func):
    """
    For one correlation method, generate a grid of heatmaps
    (one sub-plot per model).
    """
    print(f"\n>>> Generating {method_name} correlation grid ...")

    # --- Define metrics ---
    perf_metrics = ['AUC', 'F1', 'MCC', 'Recall(PD)', 'FPR(PF)',
                    'Precision', 'CE@20%', 'ECE']
    uq_metrics   = ['Entropy', 'Confidence', 'LeastConf', 'Margin',
                    'DeepGini', 'Variance', 'ExpEntropy', 'BALD']

    # Filter to columns that actually exist
    perf_metrics = [c for c in perf_metrics if c in df.columns]
    uq_metrics   = [c for c in uq_metrics   if c in df.columns]

    if not perf_metrics or not uq_metrics:
        print(f"  Warning: insufficient metric columns for {method_name}, skipping.")
        return

    # --- Grid layout ---
    models = sorted(df['Model'].unique())
    n_models = len(models)
    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.2 * n_cols, 5.8 * n_rows + 0.5))
    axes = axes.flatten()

    # --- Per-model heatmap ---
    for i, model in enumerate(models):
        ax = axes[i]
        model_df = df[df['Model'] == model]

        # Compute correlation matrix
        corr_matrix = pd.DataFrame(index=uq_metrics, columns=perf_metrics,
                                   dtype=float)

        for uq in uq_metrics:
            for perf in perf_metrics:
                sub = model_df[[uq, perf]].dropna()
                if len(sub) > 5:
                    val, _ = method_func(sub[uq], sub[perf])
                    corr_matrix.loc[uq, perf] = val
                else:
                    corr_matrix.loc[uq, perf] = np.nan

        # Draw heatmap
        sns.heatmap(corr_matrix.astype(float), ax=ax,
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    annot=True, fmt='.2f', annot_kws={"size": 9},
                    cbar=False, linewidths=0.5, linecolor='white')

        ax.set_title(model, fontsize=15, fontweight='bold', pad=10)

        # Remove tick marks completely
        ax.tick_params(axis='both', which='both', length=0, width=0,
                       top=False, bottom=False, left=False, right=False)

        # X-axis: only show labels on the last row
        if i // n_cols == n_rows - 1 or i >= n_models - n_cols:
            ax.set_xticklabels(perf_metrics, rotation=45, ha='right',
                               fontsize=10)
            plt.setp(ax.get_xticklines(), visible=False)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Y-axis: only show labels in the first column
        if i % n_cols == 0:
            ax.set_yticklabels(uq_metrics, rotation=0, fontsize=10)
            plt.setp(ax.get_yticklines(), visible=False)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # --- Shared colour bar ---
    cbar_ax = fig.add_axes([0.93, 0.28, 0.013, 0.42])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax,
                 label=f'{method_name} Correlation Coefficient')

    # --- Suptitle ---
    fig.suptitle(
        f'{method_name} Correlation: UQ Metrics vs. Performance (Per Model)',
        fontsize=24, fontweight='bold', y=1.005)

    # --- Save ---
    out = os.path.join(FIG_DIR, f'heatmap_All_Models_{method_name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  -> Saved: {out}  /  .png")


# =========================================================================
#  Main
# =========================================================================
def main():
    filepath = os.path.join(DATA_DIR, "benchmark_results_IVDP_FullUQ.csv")

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print("=" * 60)
    print("  UQ vs. Performance Correlation Heatmap Grid")
    print(f"  Data   : {os.path.basename(filepath)}")
    print(f"  Models : {df['Model'].nunique()}")
    print(f"  Rows   : {len(df)}")
    print("=" * 60)

    methods = {
        'Spearman': spearmanr,
        'Kendall':  kendalltau,
        'Pearson':  pearsonr,
    }

    for name, func in methods.items():
        plot_grid_heatmap(df, name, func)

    print(f"\n{'=' * 60}")
    print("  Done. Three heatmap grids generated.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
