"""
============================================================================
  Risk-Coverage Curves  (TOSEM Style)
============================================================================

For each of the 36 SDP datasets, train the 15 COD-selected models (with
SMOTE).  Collect **per-instance** predictions, errors, and UQ metrics.

Risk-Coverage analysis:
  - Sort test samples by uncertainty (low → high)
  - For each coverage level c ∈ [0.5, 1.0], keep only the c-fraction of
    most-certain samples, compute the error rate (risk) among those kept.
  - If UQ is reliable, lower coverage → lower risk (we reject uncertain
    samples and gain accuracy).

Two plots:
  A.  Grouped by model family (4 panels, suitable for paper body)
  B.  All 15 models on one axis (suitable for appendix)

Output:  risk_coverage_grouped.pdf  /  risk_coverage_all15.pdf
         riskcov_instances.csv.gz   (per-instance data)
============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')
sys.path.insert(0, PROJECT_ROOT)

from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              BaggingClassifier)
from imblearn.over_sampling import SMOTE

from benchmark_sdp import load_and_preprocess, get_sklearn_models

warnings.filterwarnings("ignore")

# =========================================================================
#  Config
# =========================================================================
RANDOM_STATE    = 42
UQ_COL          = 'LeastConf'    # primary UQ metric for x-ordering
COVERAGE_MIN    = 0.5
COVERAGE_MAX    = 1.0
COVERAGE_POINTS = 51


# =========================================================================
#  Per-sample UQ computation
# =========================================================================
def per_instance_uq(probs, ensemble_probs=None):
    """Compute per-sample UQ dict from probability array (N, C)."""
    eps = 1e-10
    probs = np.asarray(probs, dtype=float)
    probs_safe = np.clip(probs, eps, 1 - eps)

    entropy    = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    confidence = np.max(probs, axis=1)
    leastconf  = 1.0 - confidence

    if probs.shape[1] > 1:
        s = np.sort(probs, axis=1)
        margin = s[:, -1] - s[:, -2]
    else:
        margin = np.abs(2 * probs.flatten() - 1)

    deepgini = 1.0 - np.sum(probs ** 2, axis=1)

    out = {
        'Entropy':    entropy,
        'Confidence': confidence,
        'LeastConf':  leastconf,
        'Margin':     margin,
        'DeepGini':   deepgini,
        'Variance':   np.full(len(entropy), np.nan),
        'ExpEntropy': np.full(len(entropy), np.nan),
        'BALD':       np.full(len(entropy), np.nan),
    }

    if ensemble_probs is not None:
        ens = np.asarray(ensemble_probs)
        if ens.ndim == 2:                          # (E, N) -> (E, N, 2)
            p = ens[:, :, np.newaxis]
            ens = np.concatenate([1 - p, p], axis=2)

        out['Variance'] = np.var(ens[:, :, 1], axis=0)

        ens_safe = np.clip(ens, eps, 1 - eps)
        member_ent = -np.sum(ens_safe * np.log(ens_safe), axis=2)
        exp_ent = np.mean(member_ent, axis=0)
        out['ExpEntropy'] = exp_ent

        mean_prob = np.mean(ens, axis=0)
        mean_safe = np.clip(mean_prob, eps, 1 - eps)
        pred_ent  = -np.sum(mean_safe * np.log(mean_safe), axis=1)
        out['BALD'] = pred_ent - exp_ent

    return out


# =========================================================================
#  Process one dataset: train 15 models, collect per-instance rows
# =========================================================================
def process_dataset(file_path):
    file_name = os.path.basename(file_path)
    parent    = os.path.basename(os.path.dirname(file_path))
    ds_label  = f"{parent}/{file_name}"
    print(f"  -> {ds_label} ... ", end="", flush=True)

    result = load_and_preprocess(file_path)
    if result[0] is None:
        print("SKIP")
        return []

    X_train, X_test, y_train, y_test = result[0], result[1], result[2], result[3]

    # SMOTE
    try:
        if np.sum(y_train == 1) > 1:
            k = min(np.sum(y_train == 1) - 1, 5)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except Exception:
        X_train_res, y_train_res = X_train, y_train

    rows = []

    # --- 14 sklearn models ---
    for name, model in get_sklearn_models().items():
        try:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)

            if not hasattr(model, 'predict_proba'):
                continue

            y_prob_all = model.predict_proba(X_test)       # (N, 2)
            err = (y_pred != y_test).astype(int)

            # Ensemble probs for RF / Bagging
            ens_probs = None
            if isinstance(model, RandomForestClassifier):
                ens_probs = np.array([t.predict_proba(X_test) for t in model.estimators_])
            elif isinstance(model, BaggingClassifier):
                pp = [e.predict_proba(X_test) for e in model.estimators_
                      if hasattr(e, 'predict_proba')]
                if pp:
                    ens_probs = np.array(pp)

            uq = per_instance_uq(y_prob_all, ens_probs)

            for i in range(len(y_test)):
                rows.append({
                    'Dataset': ds_label, 'Model': name,
                    'y_true': int(y_test[i]), 'y_pred': int(y_pred[i]),
                    'error': int(err[i]),
                    'Entropy':    float(uq['Entropy'][i]),
                    'LeastConf':  float(uq['LeastConf'][i]),
                    'DeepGini':   float(uq['DeepGini'][i]),
                    'Variance':   float(uq['Variance'][i]) if np.isfinite(uq['Variance'][i]) else np.nan,
                    'BALD':       float(uq['BALD'][i]) if np.isfinite(uq['BALD'][i]) else np.nan,
                })
        except Exception:
            pass

    # --- 15th model: Boosting-SVM ---
    try:
        bsvm = AdaBoostClassifier(
            estimator=SVC(kernel='rbf', max_iter=3000, random_state=RANDOM_STATE),
            n_estimators=20, random_state=RANDOM_STATE, algorithm='SAMME')
        bsvm.fit(X_train_res, y_train_res)
        y_pred = bsvm.predict(X_test)
        err = (y_pred != y_test).astype(int)

        if hasattr(bsvm, 'decision_function'):
            scores = bsvm.decision_function(X_test)
            pp = 1.0 / (1.0 + np.exp(-scores))
            probs_all = np.vstack([1 - pp, pp]).T
            uq = per_instance_uq(probs_all, None)

            for i in range(len(y_test)):
                rows.append({
                    'Dataset': ds_label, 'Model': 'Boosting-SVM',
                    'y_true': int(y_test[i]), 'y_pred': int(y_pred[i]),
                    'error': int(err[i]),
                    'Entropy':    float(uq['Entropy'][i]),
                    'LeastConf':  float(uq['LeastConf'][i]),
                    'DeepGini':   float(uq['DeepGini'][i]),
                    'Variance':   np.nan,
                    'BALD':       np.nan,
                })
    except Exception:
        pass

    n_models = len(set(r['Model'] for r in rows))
    print(f"OK  (n_test={len(y_test)}, models={n_models})")
    return rows


# =========================================================================
#  Risk-Coverage curve computation
# =========================================================================
def risk_coverage_curve(uncertainty, error, coverages):
    """Single dataset/model: sort by uncertainty, compute risk at each coverage."""
    u = np.asarray(uncertainty)
    e = np.asarray(error, dtype=float)
    mask = np.isfinite(u) & np.isfinite(e)
    u, e = u[mask], e[mask]
    if len(e) == 0:
        return coverages, np.full(len(coverages), np.nan)

    order = np.argsort(u)          # low uncertainty first
    e_sorted = e[order]
    N = len(e_sorted)

    risks = []
    for c in coverages:
        k = max(1, min(int(np.ceil(c * N)), N))
        risks.append(e_sorted[:k].mean())
    return coverages, np.array(risks)


def macro_avg_curve(df, uq_col, coverages, min_n=10):
    """Macro-average across datasets (more fair than pooling)."""
    curves = []
    for _, g in df.groupby('Dataset'):
        if len(g) < min_n:
            continue
        _, y = risk_coverage_curve(g[uq_col].values, g['error'].values, coverages)
        if not np.all(np.isnan(y)):
            curves.append(y)
    if not curves:
        return coverages, np.full(len(coverages), np.nan)
    return coverages, np.nanmean(np.vstack(curves), axis=0)


# =========================================================================
#  TOSEM-style matplotlib settings
# =========================================================================
def apply_tosem_style():
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size':        9,
        'axes.titlesize':   10,
        'axes.labelsize':   9,
        'xtick.labelsize':  8,
        'ytick.labelsize':  8,
        'legend.fontsize':  7.5,
        'lines.linewidth':  1.8,
        'axes.linewidth':   0.8,
        'pdf.fonttype':     42,
        'ps.fonttype':      42,
        'axes.grid':        True,
        'grid.alpha':       0.25,
        'grid.linewidth':   0.6,
        'grid.linestyle':   '-',
        'mathtext.fontset': 'stix',
    })


# =========================================================================
#  Plot A — Grouped by model family (4 panels)
# =========================================================================
def plot_grouped(df, uq_col, filename=None):
    if filename is None:
        filename = os.path.join(FIG_DIR, 'risk_coverage_grouped.pdf')
    print(f"\n>>> Plot A (Grouped): {filename}")
    apply_tosem_style()

    coverages = np.linspace(COVERAGE_MIN, COVERAGE_MAX, COVERAGE_POINTS)
    baseline  = df.groupby('Dataset')['error'].mean().mean()

    groups = {
        'Linear / Probabilistic': ['LR', 'NB', 'Boosting-NB'],
        'Neighbors / Trees':      ['KNN', 'CART', 'C4.5 (J48)', 'Greedy-RL'],
        'Boosting Methods':       ['Boosting-DT', 'Boosted-RS', 'Boosting-MLP', 'Boosting-SVM'],
        'Ensembles':              ['RF', 'Voting', 'Bagged AdaBoost', 'Bagging-DT'],
    }

    cmap = plt.get_cmap('tab10')

    # Use explicit subplots_adjust instead of constrained_layout
    # to fully control spacing and avoid text overlap
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.18,
                        hspace=0.30, wspace=0.14)
    axes = axes.ravel()
    panels = ['(a)', '(b)', '(c)', '(d)']

    for ax, pan, (gname, models) in zip(axes, panels, groups.items()):
        # Baseline
        ax.plot(coverages, np.full_like(coverages, baseline),
                ls='--', lw=1.3, color='0.55', label='Random')

        for j, m in enumerate(models):
            sub = df[df['Model'] == m]
            if len(sub) == 0:
                continue
            x, y = macro_avg_curve(sub, uq_col, coverages)
            ax.plot(x, y, color=cmap(j % 10), lw=1.8, label=m)

        ax.set_title(f'{pan} {gname}', fontsize=10, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.linspace(COVERAGE_MIN, COVERAGE_MAX, 6))

        # Per-panel legend (inside each panel, avoids bottom overlap)
        ax.legend(loc='upper left', fontsize=7, frameon=True,
                  edgecolor='#D0D0D0', fancybox=False,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    # Shared axis labels
    fig.text(0.53, 0.06, 'Coverage (fraction of samples kept)',
             ha='center', fontsize=11, fontweight='medium')
    fig.text(0.02, 0.54, 'Risk (error rate on kept samples)',
             ha='center', va='center', rotation=90, fontsize=11, fontweight='medium')

    fig.suptitle(f'Risk-Coverage Curves by Model Family (UQ = {uq_col})',
                 fontsize=13, fontweight='bold', y=0.96)

    fig.savefig(filename, bbox_inches='tight', dpi=600)
    fig.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filename}  /  .png")


# =========================================================================
#  Plot B — All 15 models on one axis
# =========================================================================
ALL_15 = [
    'RF', 'Voting', 'Boosting-DT', 'Bagged AdaBoost', 'Greedy-RL',
    'Bagging-DT', 'Boosted-RS', 'LR', 'KNN', 'CART',
    'C4.5 (J48)', 'Boosting-MLP', 'Boosting-SVM', 'Boosting-NB', 'NB',
]


def plot_all15(df, uq_col, filename=None):
    if filename is None:
        filename = os.path.join(FIG_DIR, 'risk_coverage_all15.pdf')
    print(f"\n>>> Plot B (All 15): {filename}")
    apply_tosem_style()

    coverages = np.linspace(COVERAGE_MIN, COVERAGE_MAX, COVERAGE_POINTS)
    baseline  = df.groupby('Dataset')['error'].mean().mean()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(coverages, np.full_like(coverages, baseline),
            ls='--', lw=1.3, color='0.55', label='Random baseline')

    cmap = plt.get_cmap('tab20')
    ls_pool = ['-', '--', '-.', ':']

    for idx, m in enumerate(ALL_15):
        sub = df[df['Model'] == m]
        if len(sub) == 0:
            continue
        x, y = macro_avg_curve(sub, uq_col, coverages)
        ax.plot(x, y, lw=1.5,
                color=cmap(idx % 20),
                ls=ls_pool[idx % len(ls_pool)],
                label=m)

    ax.set_xlabel('Coverage (fraction of samples kept)')
    ax.set_ylabel('Risk (error rate on kept samples)')
    ax.set_title(f'Risk-Coverage Curves (All 15 Models, UQ = {uq_col})',
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=600)
    fig.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filename}  /  .png")


# =========================================================================
#  Main
# =========================================================================
def main():
    np.random.seed(RANDOM_STATE)

    out_gz = os.path.join(DATA_DIR, 'riskcov_instances.csv.gz')

    # Fast path: if per-instance data already exists, skip training
    if os.path.exists(out_gz):
        print('=' * 65)
        print('  Risk-Coverage Curves  (loading cached data)')
        print('=' * 65)
        df = pd.read_csv(out_gz)
        print(f'  Loaded: {out_gz}  ({len(df):,} rows)')
    else:
        root_dir = os.path.join(PROJECT_ROOT, 'dataset')
        if not os.path.exists(root_dir):
            print(f"Error: dataset folder not found at {root_dir}")
            sys.exit(1)

        all_files = []
        for r, _, fs in os.walk(root_dir):
            for f in fs:
                if f.endswith('.csv') or f.endswith('.arff'):
                    all_files.append(os.path.join(r, f))
        all_files.sort()

        print('=' * 65)
        print('  Risk-Coverage Curves')
        print(f'  15 COD-selected models x {len(all_files)} datasets')
        print(f'  UQ metric for ordering: {UQ_COL}')
        print('=' * 65)

        all_rows = []
        for fp in all_files:
            rows = process_dataset(fp)
            all_rows.extend(rows)

        if not all_rows:
            print('\nNo data collected.')
            sys.exit(1)

        df = pd.DataFrame(all_rows)
        df.to_csv(out_gz, index=False, compression='gzip')
        print(f'\n  Saved: {out_gz}  ({len(df):,} rows)')

    n_models = df['Model'].nunique()
    n_ds     = df['Dataset'].nunique()
    n_total  = len(df)
    n_err    = int(df['error'].sum())

    print(f'\n  Models   : {n_models}')
    print(f'  Datasets : {n_ds}')
    print(f'  Instances: {n_total:,}')
    print(f'  Errors   : {n_err:,}  ({n_err/n_total:.1%})')

    present = set(df['Model'].unique())
    missing = [m for m in ALL_15 if m not in present]
    if missing:
        print(f'  Warning: missing models: {missing}')

    # Plot
    plot_grouped(df, uq_col=UQ_COL)
    plot_all15(df, uq_col=UQ_COL)

    print(f"\n{'=' * 65}")
    print('  Done.')
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
