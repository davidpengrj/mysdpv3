"""
============================================================================
  Unsupervised Meta-learning Algorithm Selection via COD + Hierarchical
  Clustering — Reproducing Wan et al. (TOSEM) "Data Complexity: A New
  Perspective" methodology
============================================================================

Method
------
1. Train every candidate classifier on the SAME data using 5-fold
   stratified cross-validation (`cross_val_predict`) so that each
   classifier produces an out-of-fold prediction for every sample.
2. Compute pairwise Classifier Output Difference (COD) — the fraction
   of samples on which two classifiers disagree (= Hamming distance of
   their prediction vectors).
3. Feed the condensed COD distance matrix into Agglomerative / Hierarchical
   Clustering (average linkage).
4. Cut the dendrogram at a user-specified threshold (default 0.13) to
   form clusters, then pick one representative per cluster.

References
----------
- Wan et al., "Data Complexity: A New Perspective for Analyzing the
  Difficulty of Defect Prediction Tasks", TOSEM, 2023.
- Peterson & Martinez, "Estimating the potential for combining learning
  models", ICML Workshop, 2005  (origin of COD).
============================================================================
"""

import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import OrderedDict, defaultdict

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- classifiers ---
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
)

# --- rule-based classifiers (optional: imodels) ---
try:
    from imodels import BoostedRulesClassifier, GreedyRuleListClassifier
    HAS_IMODELS = True
except ImportError:
    HAS_IMODELS = False

# --- clustering & plotting ---
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


# =========================================================================
#  1. Define the full candidate algorithm pool  (32 classifiers)
# =========================================================================
def get_candidate_classifiers():
    """
    Return an OrderedDict  name -> sklearn estimator (or Pipeline).
    Models that benefit from feature scaling are wrapped in a Pipeline
    with StandardScaler.

    Pool covers: statistical, tree, ensemble (bagging/boosting/voting),
    neural-network, rule-based, and distance-based paradigms.
    """
    clfs = OrderedDict()

    # =================================================================
    # A. Base learners — statistical / linear
    # =================================================================
    clfs['NB'] = GaussianNB()

    clfs['LR'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000,
            solver='lbfgs', random_state=42))
    ])

    clfs['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RidgeClassifier(alpha=1.0, random_state=42))
    ])

    clfs['LDA'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())
    ])

    # =================================================================
    # B. Distance-based
    # =================================================================
    clfs['KNN'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])

    # =================================================================
    # C. SVM (max_iter capped for speed)
    # =================================================================
    clfs['linSVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', max_iter=5000, random_state=42))
    ])

    clfs['rbfSVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', max_iter=5000, random_state=42))
    ])

    # =================================================================
    # D. Neural-network
    # =================================================================
    clfs['MLP1'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, random_state=42))
    ])

    clfs['MLP2'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])

    clfs['MLP WeightDecay'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.1,
            max_iter=500, random_state=42))
    ])

    # PyTorch MC Dropout surrogate (approximate via stochastic MLP)
    clfs['MC Dropout'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.01,
            max_iter=500, random_state=42))
    ])

    # =================================================================
    # E. Tree-based
    # =================================================================
    clfs['CART'] = DecisionTreeClassifier(
        criterion='gini', max_depth=10, random_state=42)

    clfs['C4.5 (J48)'] = DecisionTreeClassifier(
        criterion='entropy', max_depth=10, random_state=42)

    # =================================================================
    # F. Ensemble — homogeneous
    # =================================================================
    clfs['RF'] = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)

    clfs['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)

    clfs['GBDT'] = GradientBoostingClassifier(
        n_estimators=100, random_state=42)

    # =================================================================
    # G. Bagging variants (heterogeneous base learners)
    # =================================================================
    clfs['Bagging-SVM'] = BaggingClassifier(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', max_iter=3000, random_state=42))]),
        n_estimators=10, random_state=42, n_jobs=-1)

    clfs['Bagging-DT'] = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10, random_state=42, n_jobs=-1)

    # =================================================================
    # H. Boosting variants (heterogeneous base learners)
    # =================================================================
    clfs['Boosting-DT'] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=50, random_state=42, algorithm='SAMME')

    clfs['Boosting-NB'] = AdaBoostClassifier(
        estimator=GaussianNB(),
        n_estimators=50, random_state=42, algorithm='SAMME')

    # NOTE: AdaBoost requires base estimators that accept sample_weight.
    # Pipeline objects do NOT forward sample_weight, so we use raw
    # estimators here. Data is pre-scaled in main() before cross_val_predict
    # for these three classifiers (see NEEDS_PRESCALE flag below).
    clfs['Boosting-LR'] = AdaBoostClassifier(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_estimators=50, random_state=42, algorithm='SAMME')

    clfs['Boosting-SVM'] = AdaBoostClassifier(
        estimator=SVC(kernel='rbf', max_iter=3000, random_state=42),
        n_estimators=20, random_state=42, algorithm='SAMME')

    # sklearn MLPClassifier does NOT support sample_weight, so we use
    # a Perceptron (linear single-layer network) which does support it.
    from sklearn.linear_model import Perceptron
    clfs['Boosting-MLP'] = AdaBoostClassifier(
        estimator=Perceptron(max_iter=200, random_state=42),
        n_estimators=20, random_state=42, algorithm='SAMME')

    # =================================================================
    # I. Hybrid ensemble
    # =================================================================
    clfs['Bagged AdaBoost'] = BaggingClassifier(
        estimator=AdaBoostClassifier(n_estimators=10, random_state=42),
        n_estimators=10, random_state=42, n_jobs=-1)

    # =================================================================
    # J. Voting ensemble (soft voting over diverse base learners)
    # =================================================================
    clfs['Voting'] = VotingClassifier(
        estimators=[
            ('nb', GaussianNB()),
            ('lr', Pipeline([('s', StandardScaler()),
                             ('c', LogisticRegression(max_iter=1000,
                                                      random_state=42))])),
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42,
                                          n_jobs=-1)),
        ],
        voting='soft', n_jobs=-1)

    # =================================================================
    # K. Rule-based (require `imodels`; skip gracefully if missing)
    # =================================================================
    if HAS_IMODELS:
        clfs['Boosted-RS'] = BoostedRulesClassifier(n_estimators=50)
        clfs['Greedy-RL'] = GreedyRuleListClassifier(max_depth=6)
    else:
        # Fallback: shallow tree approximations
        clfs['Boosted-RS'] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50, random_state=42, algorithm='SAMME')
        clfs['Greedy-RL'] = DecisionTreeClassifier(
            criterion='entropy', max_depth=6, random_state=42)

    return clfs


# =========================================================================
#  2. Compute pairwise COD matrix
# =========================================================================
def compute_cod_matrix(predictions: dict):
    """
    Parameters
    ----------
    predictions : dict   name -> np.ndarray of shape (n_samples,)
        Out-of-fold predicted labels (0/1) for every classifier.

    Returns
    -------
    names : list[str]
    cod_matrix : np.ndarray of shape (n_classifiers, n_classifiers)
        Symmetric matrix where entry (i, j) = Hamming distance between
        classifier i and classifier j predictions.
    """
    names = list(predictions.keys())
    n = len(names)
    pred_matrix = np.array([predictions[nm] for nm in names])  # (n_clf, n_samples)

    cod = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # COD = fraction of disagreements = Hamming distance
            d = np.mean(pred_matrix[i] != pred_matrix[j])
            cod[i, j] = d
            cod[j, i] = d

    return names, cod


# =========================================================================
#  3. Hierarchical clustering + selection
# =========================================================================
# Priority table: when we must pick ONE representative from a cluster,
# prefer faster / simpler / more commonly used classifiers.
# Lower number = higher priority.
REPRESENTATIVE_PRIORITY = {
    # --- Ensemble (strong, fast, common) ---
    'RF':               1,
    'ExtraTrees':       2,
    'GBDT':             3,
    'Boosting-DT':      4,
    'Bagged AdaBoost':  5,
    'Voting':           6,
    # --- Base statistical ---
    'NB':               7,
    'LR':               8,
    'Ridge':            9,
    'LDA':             10,
    # --- Tree ---
    'CART':            11,
    'C4.5 (J48)':      12,
    # --- Distance ---
    'KNN':             13,
    # --- SVM ---
    'linSVM':          14,
    'rbfSVM':          15,
    # --- Neural-net ---
    'MLP1':            16,
    'MLP2':            17,
    'MLP WeightDecay': 18,
    # --- Bagging variants ---
    'Bagging-DT':      19,
    'Bagging-SVM':     20,
    # --- Boosting variants ---
    'Boosting-NB':     21,
    'Boosting-LR':     22,
    'Boosting-SVM':    23,
    'Boosting-MLP':    24,
    # --- Rule-based ---
    'Boosted-RS':      25,
    'Greedy-RL':       26,
}


def cluster_and_select(names, cod_matrix, threshold=0.13,
                       linkage_method='average'):
    """
    Perform hierarchical clustering on the COD distance matrix and
    select one representative per cluster.

    Returns
    -------
    linkage_matrix : ndarray   (for plotting the dendrogram)
    cluster_labels : ndarray   cluster id per classifier
    selected : list[str]       one representative name per cluster
    cluster_map : dict         cluster_id -> list of classifier names
    """
    # Convert full square matrix to condensed form for scipy
    condensed = squareform(cod_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)

    # Cut at threshold
    labels = fcluster(Z, t=threshold, criterion='distance')

    # Group classifiers by cluster
    cluster_map = defaultdict(list)
    for name, cid in zip(names, labels):
        cluster_map[cid].append(name)

    # Pick representative from each cluster (lowest priority number)
    selected = []
    for cid in sorted(cluster_map.keys()):
        members = cluster_map[cid]
        members_sorted = sorted(
            members, key=lambda m: REPRESENTATIVE_PRIORITY.get(m, 999))
        selected.append(members_sorted[0])

    return Z, labels, selected, dict(cluster_map)


# =========================================================================
#  4. Display-name mapping  (abbreviation -> full name for figures)
# =========================================================================
DISPLAY_NAMES = {
    # --- Statistical / Linear ---
    'NB':               'Naive Bayes',
    'LR':               'Logistic Regression',
    'Ridge':            'Ridge Classifier',
    'LDA':              'Linear Discriminant Analysis',
    # --- Distance-based ---
    'KNN':              'K-Nearest Neighbors',
    # --- SVM ---
    'linSVM':           'Linear SVM',
    'rbfSVM':           'RBF-Kernel SVM',
    # --- Neural-network ---
    'MLP1':             'MLP (1-Hidden Layer)',
    'MLP2':             'MLP (2-Hidden Layers)',
    'MLP WeightDecay':  'MLP (Weight Decay)',
    # --- Tree ---
    'CART':             'Decision Tree (CART)',
    'C4.5 (J48)':       'Decision Tree (C4.5)',
    # --- Ensemble homogeneous ---
    'RF':               'Random Forest',
    'ExtraTrees':       'Extra-Trees',
    'GBDT':             'Gradient Boosting Decision Tree',
    # --- Bagging variants ---
    'Bagging-SVM':      'Bagging-SVM',
    'Bagging-DT':       'Bagging-Decision Tree',
    # --- Boosting variants ---
    'Boosting-DT':      'AdaBoost-Decision Tree',
    'Boosting-NB':      'AdaBoost-Naive Bayes',
    'Boosting-LR':      'AdaBoost-Logistic Regression',
    'Boosting-SVM':     'AdaBoost-SVM',
    'Boosting-MLP':     'AdaBoost-Perceptron',
    # --- Hybrid / Voting ---
    'Bagged AdaBoost':  'Bagged AdaBoost',
    'Voting':           'Voting Ensemble',
    # --- Rule-based ---
    'Boosted-RS':       'Boosted Rule Stumps',
    'Greedy-RL':        'Greedy Rule List',
}


def _to_display(names):
    """Map internal short names to full display names."""
    return [DISPLAY_NAMES.get(n, n) for n in names]


# =========================================================================
#  5. Publication-quality Dendrogram  (horizontal / top-venue style)
# =========================================================================
def plot_dendrogram(Z, names, threshold=0.13, output_file='dendrogram_cod.pdf'):
    """
    Draw a HORIZONTAL dendrogram — algorithm names are printed left-side
    as normal horizontal text, so they are always easy to read regardless
    of PDF zoom level.  Uses a compact figure size (~10 × 12 in) that
    maps 1:1 to an A4/letter page, keeping fonts large on screen.
    """
    from scipy.cluster.hierarchy import set_link_color_palette

    # ------------------------------------------------------------------
    # RC params — Type-42 fonts for camera-ready PDF
    # ------------------------------------------------------------------
    plt.rcParams.update({
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset':   'stix',
        'axes.unicode_minus': False,
    })

    # ------------------------------------------------------------------
    # Colour palette  (colour-blind safe, Tableau-10 inspired)
    # ------------------------------------------------------------------
    PALETTE = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
    ]
    set_link_color_palette(PALETTE)

    FG        = '#2B2B2B'
    FG_LIGHT  = '#555555'
    SPINE_CLR = '#C0C0C0'
    GRID_CLR  = '#EBEBEB'

    # ------------------------------------------------------------------
    # Figure — compact size so PDF viewer shows it ~1:1 on screen
    # Height scales with number of classifiers so labels never overlap
    # ------------------------------------------------------------------
    display_names = _to_display(names)
    n_classifiers = len(display_names)
    fig_h = max(10, n_classifiers * 0.48)
    fig_w = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------------
    # Draw HORIZONTAL dendrogram  (orientation='left')
    # Algorithm names appear on the Y-axis as normal horizontal text
    # ------------------------------------------------------------------
    dn = dendrogram(
        Z,
        labels=display_names,
        ax=ax,
        orientation='left',
        leaf_font_size=14,
        color_threshold=threshold,
        above_threshold_color='#AAAAAA',
    )

    # Thicken links
    for coll in ax.collections:
        coll.set_linewidth(2.0)
    for line in ax.lines:
        line.set_linewidth(2.0)

    set_link_color_palette(None)

    # ------------------------------------------------------------------
    # Threshold line  (vertical, since dendrogram is horizontal)
    # ------------------------------------------------------------------
    ax.axvline(x=threshold, color='#D62728', linestyle='--', linewidth=1.6,
               alpha=0.85, zorder=5)

    ylim = ax.get_ylim()
    ax.text(
        threshold + 0.003, ylim[1] * 0.98,
        f'threshold = {threshold}',
        fontsize=13, color='#D62728', fontstyle='italic',
        ha='left', va='top', rotation=0,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#D62728', alpha=0.90, linewidth=0.8),
    )

    # ------------------------------------------------------------------
    # X-axis  (distance axis — now horizontal)
    # ------------------------------------------------------------------
    ax.set_xlabel('Classifier Output Difference (COD)',
                  fontsize=16, color=FG, labelpad=12, fontweight='medium')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis='x', which='major', labelsize=13, colors=FG_LIGHT,
                   length=5, width=0.8)
    ax.tick_params(axis='x', which='minor', length=3, width=0.6,
                   colors=SPINE_CLR)

    # Subtle vertical grid
    ax.xaxis.grid(True, which='major', color=GRID_CLR, linewidth=0.6,
                  linestyle='-')
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Y-axis  (algorithm names — horizontal text, big & bold)
    # ------------------------------------------------------------------
    ax.tick_params(axis='y', labelsize=14, pad=6, length=0, colors=FG)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(14)
        lbl.set_fontweight('bold')
        lbl.set_color(FG)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        'Classifier Output Difference — Hierarchical Clustering',
        fontsize=18, fontweight='bold', color=FG, pad=18,
        fontfamily='serif',
    )

    # ------------------------------------------------------------------
    # Spines
    # ------------------------------------------------------------------
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(SPINE_CLR)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_linewidth(0.9)

    # ------------------------------------------------------------------
    # Cluster count annotation
    # ------------------------------------------------------------------
    n_clusters = len(set(dn['color_list'])) - (1 if '#AAAAAA' in dn['color_list'] else 0)
    ax.text(
        0.99, 0.01,
        f'{n_clusters} clusters  |  {n_classifiers} classifiers',
        transform=ax.transAxes, fontsize=11, color=FG_LIGHT,
        ha='right', va='bottom', fontstyle='italic',
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(output_file, format='pdf', bbox_inches='tight', dpi=600)
    fig.savefig(output_file.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    print(f"  Saved dendrogram -> {output_file}  /  .png")
    plt.close(fig)


# =========================================================================
#  6. Main
# =========================================================================
def main():
    print("=" * 70)
    print("  Algorithm Selection via COD + Hierarchical Clustering")
    print("  (Wan et al., TOSEM — 'Data Complexity: A New Perspective')")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 5-a.  Generate synthetic SDP-like dataset
    # ------------------------------------------------------------------
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=12,
        n_redundant=4, n_clusters_per_class=2,
        weights=[0.7, 0.3],          # imbalanced, typical for SDP
        flip_y=0.05, random_state=42
    )
    print(f"\n  Dataset : synthetic  (n={len(y)}, "
          f"features={X.shape[1]}, defect_rate={y.mean():.1%})")

    # ------------------------------------------------------------------
    # 5-b.  Collect 5-fold CV predictions for every classifier
    # ------------------------------------------------------------------
    clfs = get_candidate_classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pre-scaled copy for AdaBoost variants whose base learners need
    # scaling but can't use Pipeline (sample_weight forwarding issue).
    NEEDS_PRESCALE = {'Boosting-LR', 'Boosting-SVM', 'Boosting-MLP'}
    scaler_global = StandardScaler()
    X_scaled = scaler_global.fit_transform(X)

    predictions = OrderedDict()
    print(f"  Classifiers : {len(clfs)}")
    print(f"  CV strategy : 5-fold stratified\n")

    for idx, (name, clf) in enumerate(clfs.items(), 1):
        print(f"    [{idx:2d}/{len(clfs)}]  {name:<22s} ... ",
              end='', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X_use = X_scaled if name in NEEDS_PRESCALE else X
                y_pred = cross_val_predict(clf, X_use, y, cv=cv, n_jobs=1)
                predictions[name] = y_pred
                acc = np.mean(y_pred == y)
                print(f"done  (CV acc = {acc:.3f})")
            except Exception as e:
                print(f"FAILED ({e})")

    # ------------------------------------------------------------------
    # 5-c.  Compute COD distance matrix
    # ------------------------------------------------------------------
    names, cod_matrix = compute_cod_matrix(predictions)
    n_clf = len(names)

    print(f"\n  COD matrix ({n_clf} x {n_clf}) computed.")
    print(f"  Mean pairwise COD = {cod_matrix[np.triu_indices(n_clf, k=1)].mean():.4f}")
    print(f"  Max  pairwise COD = {cod_matrix[np.triu_indices(n_clf, k=1)].max():.4f}")
    print(f"  Min  pairwise COD = {cod_matrix[np.triu_indices(n_clf, k=1)].min():.4f}")

    # ------------------------------------------------------------------
    # 5-d.  Hierarchical clustering
    # ------------------------------------------------------------------
    THRESHOLD = 0.085
    Z, labels, selected, cluster_map = cluster_and_select(
        names, cod_matrix, threshold=THRESHOLD, linkage_method='average')

    print(f"\n{'=' * 70}")
    print(f"  Clustering Results  (threshold = {THRESHOLD})")
    print(f"{'=' * 70}")
    print(f"  Number of clusters : {len(cluster_map)}\n")

    for cid in sorted(cluster_map.keys()):
        members = cluster_map[cid]
        rep = [m for m in selected if m in members][0]
        print(f"  Cluster {cid}:")
        for m in members:
            tag = "  <-- representative" if m == rep else ""
            disp = DISPLAY_NAMES.get(m, m)
            print(f"      - {disp}{tag}")
        print()

    print(f"  Selected representative algorithms ({len(selected)}):")
    for i, s in enumerate(selected, 1):
        disp = DISPLAY_NAMES.get(s, s)
        print(f"    {i}. {disp}")

    # ------------------------------------------------------------------
    # 5-e.  Plot dendrogram
    # ------------------------------------------------------------------
    print()
    plot_dendrogram(Z, names, threshold=THRESHOLD,
                    output_file=os.path.join('figures', 'dendrogram_cod.pdf'))

    # ------------------------------------------------------------------
    # 5-f.  Save full COD matrix to CSV
    # ------------------------------------------------------------------
    import pandas as pd
    cod_df = pd.DataFrame(cod_matrix, index=names, columns=names)
    cod_csv = os.path.join('data', 'cod_distance_matrix.csv')
    cod_df.to_csv(cod_csv)
    print(f"\n  COD distance matrix saved -> {cod_csv}")

    # Print a compact summary (top-5 most similar & most different pairs)
    pairs = []
    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            pairs.append((names[i], names[j], cod_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])

    print(f"\n  Top-5 most SIMILAR pairs (lowest COD):")
    for a, b, d in pairs[:5]:
        print(f"    {a:<22s} <-> {b:<22s}  COD = {d:.4f}")

    print(f"\n  Top-5 most DIFFERENT pairs (highest COD):")
    for a, b, d in pairs[-5:]:
        print(f"    {a:<22s} <-> {b:<22s}  COD = {d:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
