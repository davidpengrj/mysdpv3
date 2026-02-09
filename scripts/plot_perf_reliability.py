"""
============================================================================
  Performance vs. Reliability — Two Publication-Quality Plots
============================================================================

Plot 1:  Diverging Horizontal Bar Chart
         Left  ←  Higher ECE (Worse Reliability, z-score)
         Right →  Higher MCC (Better Performance, z-score)
         Each model has two overlapping bars + value annotations.

Plot 2:  ECE Box-Plot across Classifiers
         Shows the distribution of ECE across 36 datasets per model,
         sorted by median ECE (best calibrated on the right).

Data:    benchmark_results_IVDP_FullUQ.csv  (15 models × 36 datasets)
Output:  perf_vs_reliability_bars.pdf  /  ece_boxplot.pdf
============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')

# =========================================================================
#  Publication aesthetics
# =========================================================================
plt.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'stix',
    'axes.unicode_minus': False,
})

COLOR_PERF  = '#3C5488'   # dark teal-blue  (Performance / MCC)
COLOR_ERR   = '#B5577D'   # muted rose-pink (Error / ECE)
COLOR_TEXT  = '#333333'
SPINE_CLR   = '#C0C0C0'
GRID_CLR    = '#EBEBEB'


# =========================================================================
#  Load data
# =========================================================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


# =========================================================================
#  Plot 1 — Diverging Horizontal Bar Chart
# =========================================================================
def plot_diverging_bars(df, perf='MCC', err='ECE',
                        filename='perf_vs_reliability_bars.pdf'):
    """
    Z-score normalise both metrics across the 15 models, then draw:
      - Performance (MCC) bars growing to the RIGHT  (higher = better)
      - Error (ECE) bars growing to the LEFT (inverted: higher ECE = worse)
    Models sorted top-to-bottom by performance rank.
    """
    print(f"\n>>> Plot 1: Diverging bar chart → {filename}")

    # Aggregate across datasets
    agg = df.groupby('Model')[[perf, err]].mean().reset_index()

    # Z-score standardise
    agg['z_perf'] = (agg[perf] - agg[perf].mean()) / agg[perf].std()
    agg['z_err']  = (agg[err]  - agg[err].mean())  / agg[err].std()

    # Sort by performance (best at top)
    agg = agg.sort_values('z_perf', ascending=True).reset_index(drop=True)

    n = len(agg)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(10.5, 0.58 * n + 1.8))

    bar_h = 0.38

    # --- ECE bars (extend LEFT, so negate z-score) ---
    bars_err = ax.barh(y_pos + bar_h / 2, -agg['z_err'], height=bar_h,
                       color=COLOR_ERR, alpha=0.82, zorder=2,
                       label=f'Error ({err})')

    # --- MCC bars (extend RIGHT) ---
    bars_perf = ax.barh(y_pos - bar_h / 2, agg['z_perf'], height=bar_h,
                        color=COLOR_PERF, alpha=0.88, zorder=2,
                        label=f'Performance ({perf})')

    # --- Value annotations ---
    for i, row in agg.iterrows():
        idx = agg.index.get_loc(i) if isinstance(agg.index, pd.RangeIndex) else i

        # ECE value (on left side of its bar)
        ece_val = row[err]
        ece_z   = -row['z_err']
        x_ece   = ece_z - 0.06 if ece_z < -0.2 else ece_z + 0.06
        ha_ece  = 'right' if ece_z < -0.2 else 'left'
        ax.text(x_ece, idx + bar_h / 2, f'{ece_val:.2f}',
                ha=ha_ece, va='center', fontsize=8.5, color=COLOR_ERR,
                fontweight='medium')

        # MCC value (on right side of its bar)
        mcc_val = row[perf]
        mcc_z   = row['z_perf']
        x_mcc   = mcc_z + 0.06 if mcc_z > 0.2 else mcc_z - 0.06
        ha_mcc  = 'left' if mcc_z > 0.2 else 'right'
        ax.text(x_mcc, idx - bar_h / 2, f'{mcc_val:.2f}',
                ha=ha_mcc, va='center', fontsize=8.5, color=COLOR_PERF,
                fontweight='medium')

    # --- Centre line ---
    ax.axvline(x=0, color=SPINE_CLR, linewidth=1.0, zorder=1)

    # --- Y axis: model names ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['Model'], fontsize=11, fontweight='medium',
                       color=COLOR_TEXT)

    # --- X axis ---
    ax.set_xlabel('Standardized Score (Z-Score)', fontsize=12,
                  color=COLOR_TEXT, fontweight='medium', labelpad=8)
    # Secondary X label (direction hints)
    ax.text(0.5, -0.09,
            r'$\leftarrow$ Higher Error (Worse Reliability)  |  '
            r'Higher Performance (Better) $\rightarrow$',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color='#777777')

    ax.tick_params(axis='x', labelsize=10, colors=COLOR_TEXT)

    # --- Grid & spines ---
    ax.xaxis.grid(True, linestyle='--', alpha=0.25, color=GRID_CLR)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_color(SPINE_CLR)

    # --- Legend ---
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06),
              ncol=2, frameon=False, fontsize=11, handlelength=1.8,
              handletextpad=0.5)

    plt.tight_layout()
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=600)
    fig.savefig(filename.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {filename}  /  .png")


# =========================================================================
#  Plot 2 — ECE Box-Plot across Classifiers
# =========================================================================
def plot_ece_boxplot(df, metric='ECE', filename='ece_boxplot.pdf'):
    """
    Box-plot of ECE distribution (across 36 datasets) per model.
    Sorted left-to-right by **descending** median ECE (worst → best),
    so the best-calibrated model is on the right.
    """
    print(f"\n>>> Plot 2: ECE box-plot → {filename}")

    # Sort models by median ECE (descending — worst on left)
    medians = df.groupby('Model')[metric].median().sort_values(ascending=False)
    model_order = medians.index.tolist()

    # Prepare data in plot order
    data_list = [df[df['Model'] == m][metric].values for m in model_order]
    n = len(model_order)

    # Colour gradient: worst (left) → best (right)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'ece_grad', ['#4B2E6B', '#3C5488', '#2A7A8C', '#4EA774', '#A8C256', '#D4CC36'])
    colors = [cmap(i / (n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bp = ax.boxplot(data_list, patch_artist=True, widths=0.6,
                    medianprops=dict(color='#222222', linewidth=1.6),
                    whiskerprops=dict(color='#666666', linewidth=1.0),
                    capprops=dict(color='#666666', linewidth=1.0),
                    flierprops=dict(marker='o', markerfacecolor='#999999',
                                    markeredgecolor='none', markersize=4, alpha=0.5))

    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#555555')
        patch.set_linewidth(0.8)

    # --- X axis ---
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(model_order, rotation=45, ha='right',
                       fontsize=10, fontweight='medium', color=COLOR_TEXT)

    # --- Y axis ---
    ax.set_ylabel('ECE (Lower is Better)', fontsize=13,
                  fontweight='bold', color=COLOR_TEXT, labelpad=8)
    ax.tick_params(axis='y', labelsize=10, colors=COLOR_TEXT)

    # --- Title ---
    ax.set_title('Who Needs Calibration?\n'
                 '(Expected Calibration Error across Classifiers)',
                 fontsize=14, fontweight='bold', color=COLOR_TEXT, pad=14)

    # --- Grid & spines ---
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=GRID_CLR)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_color(SPINE_CLR)

    plt.tight_layout()
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=600)
    fig.savefig(filename.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {filename}  /  .png")


# =========================================================================
#  Main
# =========================================================================
if __name__ == "__main__":
    filepath = os.path.join(DATA_DIR, "benchmark_results_IVDP_FullUQ.csv")

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    df = load_data(filepath)

    print("=" * 60)
    print("  Performance vs. Reliability Plots")
    print(f"  Data  : {os.path.basename(filepath)}")
    print(f"  Models: {df['Model'].nunique()}")
    print(f"  Rows  : {len(df)}")
    print("=" * 60)

    plot_diverging_bars(df, perf='MCC', err='ECE',
                        filename=os.path.join(FIG_DIR, 'perf_vs_reliability_bars.pdf'))

    plot_ece_boxplot(df, metric='ECE',
                     filename=os.path.join(FIG_DIR, 'ece_boxplot.pdf'))

    print(f"\n{'=' * 60}")
    print("  Done. Two plots generated.")
    print(f"{'=' * 60}")
