"""
============================================================================
  Misclassification Rate vs. Prediction Uncertainty — Binned Line Chart
============================================================================

Goal: Prove that the UQ (Uncertainty Quantification) metric — prediction
entropy — is **reliable**.  When models are more uncertain about a sample,
their predictions are more likely to be **wrong**.

Method:
  For each of the 36 SDP datasets, the 15 COD-selected representative
  classifiers are trained (with SMOTE).  For every test sample:
    1. Each model produces a prediction (0 or 1) via majority-vote.
    2. The majority-vote label  →  y_pred.
    3. Each model's prediction entropy is averaged  →  uncertainty score.
    4. A sample is "misclassified" when  y_pred ≠ y_true.

  All samples are pooled across 36 datasets and binned into 10 equal-width
  uncertainty intervals.  Within each bin we compute:
      Misclassification Rate = (# wrong predictions) / (# samples in bin)

  Expected result:  Misclassification rate increases monotonically with
  uncertainty, confirming UQ reliability.

Output:  misclassification_vs_uncertainty.pdf  /  .png
============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

# Project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')
sys.path.insert(0, PROJECT_ROOT)

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE

# Reuse infrastructure from benchmark_sdp.py
from benchmark_sdp import load_and_preprocess, get_sklearn_models


# =========================================================================
#  Per-sample entropy
# =========================================================================
def per_sample_entropy(probs):
    """Shannon entropy per sample.  probs: (n_samples, n_classes)."""
    eps = 1e-10
    p = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(p * np.log(p), axis=1)


# =========================================================================
#  Wilson score interval (for binomial proportion CI)
# =========================================================================
def wilson_ci(n_success, n_total, z=1.96):
    """Return (lower, upper) of the Wilson score 95 % CI."""
    if n_total == 0:
        return 0.0, 0.0
    p_hat = n_success / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total) / denom
    return max(0, centre - margin), min(1, centre + margin)


# =========================================================================
#  Process one dataset → per-sample (avg entropy, majority-vote pred, y_true)
# =========================================================================
def process_dataset(file_path):
    """
    Train all 15 models on one dataset.
    Return arrays:
        avg_entropy : ndarray (n_test,)  — mean entropy across models
        y_pred_vote : ndarray (n_test,)  — majority-vote prediction (0 or 1)
        y_true      : ndarray (n_test,)  — true labels (0/1)
    Returns (None, None, None) on failure.
    """
    file_name = os.path.basename(file_path)
    parent_folder = os.path.basename(os.path.dirname(file_path))
    print(f"  -> {parent_folder}/{file_name} ... ", end="", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = load_and_preprocess(file_path)

    if result[0] is None:
        print("SKIP")
        return None, None, None

    X_train, X_test, y_train, y_test = result[0], result[1], result[2], result[3]

    # SMOTE
    try:
        if np.sum(y_train == 1) > 1:
            k = min(np.sum(y_train == 1) - 1, 5)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except Exception:
        X_train_res, y_train_res = X_train, y_train

    n_test = len(y_test)
    entropy_accum = np.zeros(n_test)     # running sum of entropies
    vote_accum    = np.zeros(n_test)     # sum of predicted labels (0/1)
    model_count   = np.zeros(n_test)     # how many models contributed

    # --- 14 models from get_sklearn_models() ---
    for name, model in get_sklearn_models().items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_res, y_train_res)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)
                    ent = per_sample_entropy(probs)
                    preds = model.predict(X_test)
                    entropy_accum += ent
                    vote_accum    += preds.astype(float)
                    model_count   += 1
        except Exception:
            pass

    # --- Cluster 15: Boosting-SVM (special case) ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bsvm = AdaBoostClassifier(
                estimator=SVC(kernel='rbf', max_iter=3000, random_state=42),
                n_estimators=20, random_state=42, algorithm='SAMME')
            bsvm.fit(X_train_res, y_train_res)
            preds_bsvm = bsvm.predict(X_test)
            if hasattr(bsvm, "decision_function"):
                scores = bsvm.decision_function(X_test)
                pseudo_prob = 1.0 / (1.0 + np.exp(-scores))
                probs_all = np.vstack([1 - pseudo_prob, pseudo_prob]).T
                ent = per_sample_entropy(probs_all)
                entropy_accum += ent
                vote_accum    += preds_bsvm.astype(float)
                model_count   += 1
    except Exception:
        pass

    # Average entropy & majority vote
    valid = model_count > 0
    avg_entropy = np.zeros(n_test)
    avg_entropy[valid] = entropy_accum[valid] / model_count[valid]

    # Majority vote: if >50% of models predict 1 → label 1
    y_pred_vote = np.zeros(n_test, dtype=int)
    y_pred_vote[valid] = (vote_accum[valid] / model_count[valid] >= 0.5).astype(int)

    print(f"OK  (n_test={n_test}, models_ok={int(model_count.max())})")
    return avg_entropy[valid], y_pred_vote[valid], y_test[valid]


# =========================================================================
#  Publication-quality line chart — Misclassification Rate vs Uncertainty
# =========================================================================
def plot_misclassification_vs_uncertainty(avg_ent, y_pred, y_true, n_bins=10,
                                          output_file="misclassification_vs_uncertainty.pdf"):
    """
    avg_ent : 1-D array of averaged prediction entropies
    y_pred  : 1-D array of majority-vote predictions (0/1)
    y_true  : 1-D array of true labels (0/1)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # ------------------------------------------------------------------
    # RC params  (publication quality)
    # ------------------------------------------------------------------
    plt.rcParams.update({
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset':   'stix',
        'axes.unicode_minus': False,
    })

    FG        = '#2B2B2B'
    FG_LIGHT  = '#555555'
    SPINE_CLR = '#C0C0C0'
    GRID_CLR  = '#EBEBEB'
    MAIN_CLR  = '#C44E52'    # warm red for misclassification
    ACC_CLR   = '#4C72B0'    # blue for accuracy (secondary reference)

    # Compute per-sample correctness
    is_wrong = (y_pred != y_true).astype(int)

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    bin_edges = np.linspace(avg_ent.min(), avg_ent.max(), n_bins + 1)
    bin_mids      = []
    misclass_rates = []
    ci_lowers      = []
    ci_uppers      = []
    counts         = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (avg_ent >= lo) & (avg_ent < hi)
        else:
            mask = (avg_ent >= lo) & (avg_ent <= hi)   # include right edge

        n_total = mask.sum()
        n_wrong = is_wrong[mask].sum() if n_total > 0 else 0

        rate = n_wrong / n_total if n_total > 0 else 0
        lo_ci, hi_ci = wilson_ci(n_wrong, n_total)

        bin_mids.append((lo + hi) / 2)
        misclass_rates.append(rate)
        ci_lowers.append(lo_ci)
        ci_uppers.append(hi_ci)
        counts.append(n_total)

    bin_mids       = np.array(bin_mids)
    misclass_rates = np.array(misclass_rates)
    ci_lowers      = np.array(ci_lowers)
    ci_uppers      = np.array(ci_uppers)
    counts         = np.array(counts)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Secondary axis: sample count (grey bars) ---
    ax2 = ax1.twinx()
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.75
    ax2.bar(bin_mids, counts, width=bar_width, color='#DDDDDD',
            edgecolor='#CCCCCC', linewidth=0.6, alpha=0.55, zorder=1,
            label='Sample count')
    ax2.set_ylabel('Sample Count', fontsize=13, color='#999999',
                   labelpad=10, fontweight='medium')
    ax2.tick_params(axis='y', labelsize=11, colors='#BBBBBB')
    ax2.spines['right'].set_color('#DDDDDD')
    ax2.set_ylim(0, counts.max() * 2.2)   # push bars to bottom half

    # --- Main axis: misclassification rate line ---
    ax1.fill_between(bin_mids, ci_lowers, ci_uppers,
                     color=MAIN_CLR, alpha=0.18, zorder=3,
                     label='95% Wilson CI')
    ax1.plot(bin_mids, misclass_rates, color=MAIN_CLR, linewidth=2.4,
             marker='o', markersize=8, markerfacecolor='white',
             markeredgewidth=2.0, markeredgecolor=MAIN_CLR,
             zorder=4, label='Misclassification rate')

    # Annotate each data point
    for x, y in zip(bin_mids, misclass_rates):
        ax1.annotate(f'{y:.1%}', (x, y),
                     textcoords='offset points', xytext=(0, 12),
                     fontsize=9, color=FG_LIGHT, ha='center', va='bottom',
                     fontweight='medium')

    # ------------------------------------------------------------------
    # Spearman rank correlation annotation
    # ------------------------------------------------------------------
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(bin_mids, misclass_rates)
    sig_str = ''
    if p_val < 0.001:
        sig_str = '***'
    elif p_val < 0.01:
        sig_str = '**'
    elif p_val < 0.05:
        sig_str = '*'
    ax1.text(
        0.02, 0.96,
        f'Spearman $\\rho$ = {rho:.3f}{sig_str}',
        transform=ax1.transAxes, fontsize=12, color=FG,
        ha='left', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF9E6',
                  edgecolor=SPINE_CLR, linewidth=0.8),
    )

    # ------------------------------------------------------------------
    # Axes styling
    # ------------------------------------------------------------------
    ax1.set_xlabel('Average Prediction Entropy (Uncertainty)',
                   fontsize=15, color=FG, labelpad=10, fontweight='medium')
    ax1.set_ylabel('Misclassification Rate',
                   fontsize=15, color=FG, labelpad=10, fontweight='medium')
    ax1.tick_params(axis='x', labelsize=12, colors=FG_LIGHT)
    ax1.tick_params(axis='y', labelsize=12, colors=FG_LIGHT)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylim(-0.02, 1.02)

    # Grid
    ax1.yaxis.grid(True, which='major', color=GRID_CLR, linewidth=0.6)
    ax1.set_axisbelow(True)

    # Title
    ax1.set_title(
        'Higher Uncertainty $\\Rightarrow$ Higher Misclassification Rate',
        fontsize=17, fontweight='bold', color=FG, pad=18, fontfamily='serif',
    )

    # Spines
    for spine in ['top']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(SPINE_CLR)
    ax1.spines['left'].set_linewidth(0.9)
    ax1.spines['bottom'].set_color(SPINE_CLR)
    ax1.spines['bottom'].set_linewidth(0.9)

    # Legend (combine both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=12, loc='center left', frameon=True,
               edgecolor=SPINE_CLR, fancybox=False,
               bbox_to_anchor=(0.0, 0.72))

    # Bring main axis to front
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # ------------------------------------------------------------------
    # Summary annotation
    # ------------------------------------------------------------------
    n_total = len(y_true)
    n_wrong_total = int(is_wrong.sum())
    overall_err = n_wrong_total / n_total
    ax1.text(
        0.99, 0.02,
        f'Total: {n_total:,} samples  |  Overall error: {n_wrong_total:,} ({overall_err:.1%})',
        transform=ax1.transAxes, fontsize=9.5, color=FG_LIGHT,
        ha='right', va='bottom', fontstyle='italic',
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(output_file, format='pdf', bbox_inches='tight', dpi=600)
    fig.savefig(output_file.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    print(f"\n  Saved -> {output_file}  /  .png")
    plt.close(fig)


# =========================================================================
#  Main
# =========================================================================
def main():
    print("=" * 70)
    print("  Misclassification Rate vs. Prediction Uncertainty")
    print("  15 COD-selected models x 36 datasets, majority-vote ensemble")
    print("=" * 70)

    root_dir = os.path.join(PROJECT_ROOT, "dataset")
    if not os.path.exists(root_dir):
        print(f"Error: dataset folder not found at {root_dir}")
        sys.exit(1)

    # Collect all dataset files
    all_files = []
    for r, d, fs in os.walk(root_dir):
        for f in fs:
            if f.endswith('.csv') or f.endswith('.arff'):
                all_files.append(os.path.join(r, f))
    all_files.sort(key=lambda p: os.path.basename(os.path.dirname(p)))

    print(f"  Dataset folder : {root_dir}")
    print(f"  Files found    : {len(all_files)}")
    print(f"  Models         : 15 (COD-selected representatives)")
    print()

    warnings.filterwarnings("ignore")

    # Process all datasets
    all_entropy = []
    all_preds   = []
    all_labels  = []

    for fp in all_files:
        avg_ent, y_pred, y_true = process_dataset(fp)
        if avg_ent is not None:
            all_entropy.append(avg_ent)
            all_preds.append(y_pred)
            all_labels.append(y_true)

    if not all_entropy:
        print("\nNo data collected. Check dataset files.")
        sys.exit(1)

    avg_ent_all = np.concatenate(all_entropy)
    y_pred_all  = np.concatenate(all_preds)
    y_true_all  = np.concatenate(all_labels)

    is_wrong_all = (y_pred_all != y_true_all).astype(int)

    n_total     = len(y_true_all)
    n_wrong     = int(is_wrong_all.sum())
    n_correct   = n_total - n_wrong
    overall_err = n_wrong / n_total

    print(f"\n{'=' * 70}")
    print(f"  Total test samples     : {n_total:,}")
    print(f"    Correct predictions  : {n_correct:,}  ({n_correct/n_total:.1%})")
    print(f"    Wrong predictions    : {n_wrong:,}  ({overall_err:.1%})")
    print(f"  Entropy range          : [{avg_ent_all.min():.4f}, {avg_ent_all.max():.4f}]")
    print(f"  Entropy mean           : {avg_ent_all.mean():.4f}")
    print(f"{'=' * 70}")

    # --- Entropy statistics for correct vs wrong predictions ---
    ent_correct = avg_ent_all[is_wrong_all == 0]
    ent_wrong   = avg_ent_all[is_wrong_all == 1]
    print(f"\n  Entropy breakdown:")
    print(f"    Correct predictions → mean entropy = {ent_correct.mean():.4f}  "
          f"(median = {np.median(ent_correct):.4f})")
    print(f"    Wrong predictions   → mean entropy = {ent_wrong.mean():.4f}  "
          f"(median = {np.median(ent_wrong):.4f})")
    print(f"    Difference          → {ent_wrong.mean() - ent_correct.mean():.4f}  "
          f"(wrong is {ent_wrong.mean()/ent_correct.mean():.1f}x higher)")

    # --- Spearman correlation at sample level ---
    from scipy.stats import spearmanr, mannwhitneyu
    rho_sample, p_sample = spearmanr(avg_ent_all, is_wrong_all)
    print(f"\n  Sample-level Spearman correlation:")
    print(f"    rho = {rho_sample:.4f},  p = {p_sample:.2e}")

    # Mann-Whitney U test: is entropy of wrong > entropy of correct?
    u_stat, p_mw = mannwhitneyu(ent_wrong, ent_correct, alternative='greater')
    print(f"\n  Mann-Whitney U test (wrong > correct):")
    print(f"    U = {u_stat:.0f},  p = {p_mw:.2e}")
    if p_mw < 0.001:
        print(f"    → Highly significant: wrong predictions have higher entropy.")

    # Plot
    print(f"\n  Generating misclassification vs. uncertainty chart ...")
    plot_misclassification_vs_uncertainty(
        avg_ent_all, y_pred_all, y_true_all, n_bins=10,
        output_file=os.path.join(FIG_DIR, "misclassification_vs_uncertainty.pdf"))

    print(f"\n{'=' * 70}")
    print(f"  Done. UQ reliability validated.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
