"""
============================================================================
  Performance vs. Reliability Rank Difference Plot
============================================================================

Compare each model's **performance rank** (by MCC) against its
**reliability rank** (by ECE, lower is better).

Two complementary views:
  A.  Horizontal rank-difference chart  (wide, ideal for single-column)
  B.  Compact slope-graph               (vertical, shows crossings)

Data source:  benchmark_results_IVDP_FullUQ.csv
Models:       15 COD-selected representatives
Datasets:     36 SDP datasets (averaged per model)

Output:  rank_diff_horizontal.pdf  /  rank_diff_slopegraph.pdf
============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys

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

COLOR_BLUE  = '#3C5488'   # rank up   / performance
COLOR_RED   = '#DC0000'   # rank drop / risk
COLOR_GRAY  = '#B0B0B0'   # stable
COLOR_TEXT  = '#333333'
SPINE_CLR   = '#C0C0C0'
GRID_CLR    = '#EBEBEB'


# =========================================================================
#  Load & aggregate data
# =========================================================================
def get_data(filename, perf_metric='MCC', rel_metric='ECE'):
    """Read CSV → aggregate across datasets → return DataFrame with ranks."""
    df = pd.read_csv(filename)
    df_agg = df.groupby('Model')[[perf_metric, rel_metric]].mean().reset_index()

    # Performance rank: higher MCC → rank 1
    df_agg['Perf_Rank'] = df_agg[perf_metric].rank(ascending=False, method='min').astype(int)
    # Reliability rank: lower ECE → rank 1
    df_agg['Rel_Rank']  = df_agg[rel_metric].rank(ascending=True,  method='min').astype(int)

    # Sort by performance rank
    df_agg = df_agg.sort_values('Perf_Rank').reset_index(drop=True)
    return df_agg


# =========================================================================
#  Plot A — Horizontal Rank Difference Chart
# =========================================================================
def plot_horizontal_rank_diff(df, perf_metric='MCC', rel_metric='ECE',
                               filename='rank_diff_horizontal.pdf'):
    print(f"\n>>> Plot A (Horizontal Rank Difference): {filename} ...")

    n_models = len(df)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, row in df.iterrows():
        rank_perf = row['Perf_Rank']
        rank_rel  = row['Rel_Rank']
        x_pos     = i

        diff = rank_rel - rank_perf
        if diff > 3:
            color = COLOR_RED
        elif diff < -3:
            color = COLOR_BLUE
        else:
            color = COLOR_GRAY

        # Vertical connecting line
        ax.plot([x_pos, x_pos], [rank_perf, rank_rel],
                color=color, alpha=0.6, linewidth=1.8, zorder=1)

        # Performance rank (circle)
        ax.scatter(x_pos, rank_perf, color=COLOR_BLUE, s=90, zorder=3,
                   edgecolors='white', linewidths=0.6,
                   label='Performance Rank' if i == 0 else "")
        # Reliability rank (square)
        ax.scatter(x_pos, rank_rel, color=COLOR_RED, marker='s', s=70, zorder=3,
                   edgecolors='white', linewidths=0.6,
                   label='Reliability Rank' if i == 0 else "")

        # Rank numbers (offset to avoid overlap)
        offset = 0.18
        ax.text(x_pos + offset, rank_perf, str(int(rank_perf)),
                va='center', fontsize=8.5, color=COLOR_BLUE, fontweight='bold')
        if rank_rel != rank_perf:
            ax.text(x_pos + offset, rank_rel, str(int(rank_rel)),
                    va='center', fontsize=8.5, color=COLOR_RED, fontweight='bold')

    # --- X axis ---
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(df['Model'], rotation=45, ha='right',
                       fontsize=10, fontweight='medium', color=COLOR_TEXT)

    # --- Y axis: rank (1 at top) ---
    ax.set_ylim(n_models + 1.2, -0.5)
    ax.set_ylabel("Rank (Lower is Better)", fontsize=12,
                  fontweight='bold', color=COLOR_TEXT)
    ax.set_yticks(range(1, n_models + 1))
    ax.tick_params(axis='y', labelsize=10, colors=COLOR_TEXT)

    # Grid & spines
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_CLR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_color(SPINE_CLR)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BLUE,
               label=f'Performance Rank ({perf_metric})', markersize=9),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_RED,
               label=f'Reliability Rank ({rel_metric})', markersize=9),
        Line2D([0], [0], color=COLOR_RED,  lw=2, label='Rank Drop (High Risk)'),
        Line2D([0], [0], color=COLOR_BLUE, lw=2, label='Rank Rise (Better Reliability)'),
        Line2D([0], [0], color=COLOR_GRAY, lw=2, label='Stable'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False,
              fontsize=9.5, handletextpad=0.6, columnspacing=1.2)

    plt.tight_layout()
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=600)
    fig.savefig(filename.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {filename}  /  .png")


# =========================================================================
#  Plot B — Compact Slope-graph
# =========================================================================
def plot_compact_slopegraph(df, perf_metric='MCC', rel_metric='ECE',
                             filename='rank_diff_slopegraph.pdf'):
    print(f"\n>>> Plot B (Compact Slopegraph): {filename} ...")

    # Sorted position on each side (to avoid label overlap)
    df_perf = df.sort_values(['Perf_Rank', 'Model']).reset_index(drop=True)
    df_perf['Left_Pos'] = df_perf.index + 1

    df_rel = df.sort_values(['Rel_Rank', 'Model']).reset_index(drop=True)
    df_rel['Right_Pos'] = df_rel.index + 1

    plot_data = df.merge(df_perf[['Model', 'Left_Pos']], on='Model') \
                  .merge(df_rel[['Model', 'Right_Pos']], on='Model')

    n_models = len(plot_data)
    fig, ax = plt.subplots(figsize=(9, n_models * 0.48 + 2.0))

    x_left, x_right = 1.0, 2.8

    for _, row in plot_data.iterrows():
        rank_left  = int(row['Perf_Rank'])
        rank_right = int(row['Rel_Rank'])
        y_left     = row['Left_Pos']
        y_right    = row['Right_Pos']

        diff = rank_right - rank_left
        if diff > 3:
            color, alpha, lw, zorder, fw = COLOR_RED,  0.9, 2.2, 3, 'bold'
        elif diff < -3:
            color, alpha, lw, zorder, fw = COLOR_BLUE, 0.8, 1.8, 2, 'bold'
        elif abs(diff) <= 2:
            color, alpha, lw, zorder, fw = COLOR_GRAY, 0.45, 0.9, 1, 'normal'
        else:
            color, alpha, lw, zorder, fw = COLOR_GRAY, 0.6, 1.0, 1, 'normal'

        ax.plot([x_left, x_right], [y_left, y_right],
                c=color, alpha=alpha, lw=lw, marker='o', markersize=5,
                markeredgecolor='white', markeredgewidth=0.4, zorder=zorder)

        fs = 9.5

        # Left side: performance rank + model name
        ax.text(x_left - 0.06, y_left, f'#{rank_left}',
                ha='right', va='center', fontsize=fs, fontweight='bold', color=color)
        ax.text(x_left - 0.32, y_left, row['Model'],
                ha='right', va='center', fontsize=fs, fontweight=fw, color=COLOR_TEXT)

        # Right side: reliability rank + model name
        ax.text(x_right + 0.06, y_right, f'#{rank_right}',
                ha='left', va='center', fontsize=fs, fontweight='bold', color=color)
        ax.text(x_right + 0.32, y_right, row['Model'],
                ha='left', va='center', fontsize=fs, fontweight=fw, color=COLOR_TEXT)

    # Axis setup
    ax.set_ylim(n_models + 0.8, 0.2)
    ax.set_xlim(x_left - 3.5, x_right + 3.5)
    ax.axis('off')

    # Column headers
    ax.text(x_left, -0.3, f'Performance Rank ({perf_metric})',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color=COLOR_TEXT)
    ax.text(x_right, -0.3, f'Reliability Rank ({rel_metric})',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color=COLOR_TEXT)

    # Subtle separator lines
    ax.axvline(x=x_left,  ymin=0.02, ymax=0.97,
               color=SPINE_CLR, linewidth=0.6, linestyle='-')
    ax.axvline(x=x_right, ymin=0.02, ymax=0.97,
               color=SPINE_CLR, linewidth=0.6, linestyle='-')

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
    file_path = os.path.join(DATA_DIR, "benchmark_results_IVDP_FullUQ.csv")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    perf_metric = 'MCC'
    rel_metric  = 'ECE'

    df = get_data(file_path, perf_metric, rel_metric)

    print("=" * 60)
    print("  Performance vs. Reliability Rank Comparison")
    print(f"  Performance metric : {perf_metric} (higher is better)")
    print(f"  Reliability metric : {rel_metric} (lower is better)")
    print(f"  Models             : {len(df)}")
    print("=" * 60)

    # Print rank table
    print(f"\n  {'Model':<20s}  {perf_metric:>6s}  Perf#  {rel_metric:>6s}  Rel#  Diff")
    print(f"  {'-'*20}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*4}  {'-'*4}")
    for _, row in df.iterrows():
        diff = int(row['Rel_Rank'] - row['Perf_Rank'])
        sign = '+' if diff > 0 else ''
        print(f"  {row['Model']:<20s}  {row[perf_metric]:6.4f}  "
              f"{int(row['Perf_Rank']):>5d}  {row[rel_metric]:6.4f}  "
              f"{int(row['Rel_Rank']):>4d}  {sign}{diff}")

    # Generate both plots
    plot_horizontal_rank_diff(df, perf_metric, rel_metric,
                              filename=os.path.join(FIG_DIR, 'rank_diff_horizontal.pdf'))
    plot_compact_slopegraph(df, perf_metric, rel_metric,
                            filename=os.path.join(FIG_DIR, 'rank_diff_slopegraph.pdf'))

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")
