import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import warnings

# === Sklearn 模型库导入 (COD-selected 15 representatives) ===
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)

# =========================================================================
#  ARFF 文件解析器 (不依赖 scipy / liac-arff)
# =========================================================================
def parse_arff(file_path):
    """手动解析 ARFF 文件，返回 pandas DataFrame"""
    attributes = []
    attr_types = []
    data_lines = []
    in_data = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith('%') or raw_line.startswith('###'):
                continue
            if raw_line.lower().startswith('@data'):
                in_data = True
                continue
            if not in_data:
                # 处理一行中可能有多个 @attribute 粘连的情况
                # 例如: "@attribute A numeric@attribute B numeric"
                import re
                sub_lines = re.split(r'(?i)(?=@attribute\s)', raw_line)
                for seg in sub_lines:
                    seg = seg.strip()
                    if not seg:
                        continue
                    if seg.lower().startswith('@attribute'):
                        parts = seg.split(None, 2)  # @attribute name type
                        attr_name = parts[1] if len(parts) > 1 else 'unknown'
                        attr_type = parts[2] if len(parts) > 2 else 'numeric'
                        attributes.append(attr_name)
                        attr_types.append(attr_type.strip())
            else:
                # 数据行
                if raw_line:
                    data_lines.append(raw_line.split(','))

    if not data_lines:
        return None

    # 处理列数不一致：以 attributes 数量为准，截断或补齐数据行
    n_attrs = len(attributes)
    cleaned_lines = []
    for row in data_lines:
        if len(row) == n_attrs:
            cleaned_lines.append(row)
        elif len(row) > n_attrs:
            cleaned_lines.append(row[:n_attrs])
        else:
            cleaned_lines.append(row + [''] * (n_attrs - len(row)))

    df = pd.DataFrame(cleaned_lines, columns=attributes)

    # 将 numeric 列转为数值
    for i, (col, atype) in enumerate(zip(attributes, attr_types)):
        if 'numeric' in atype.lower() or 'real' in atype.lower() or 'integer' in atype.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# =========================================================================
#  1. 数据加载与预处理
# =========================================================================
def load_and_preprocess(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.arff':
            df = parse_arff(file_path)
            if df is None:
                return None, None, None, None, None, None
        else:
            return None, None, None, None, None, None

        # --- 1.1 识别 Label 列 ---
        possible_labels = ['defects', 'bug', 'class', 'label', 'problems', 'defective',
                           'is_defective', 'isdefective']
        label_col = None
        for target in possible_labels:
            for raw_col in df.columns:
                if raw_col.lower() == target:
                    label_col = raw_col
                    break
            if label_col:
                break
        if label_col is None:
            label_col = df.columns[-1]

        # --- 1.2 识别 LOC 列 (用于计算 Cost Effectiveness) ---
        possible_loc = ['loc', 'countlinecode', 'loc_total', 'countline',
                        'loc_code', 'lines', 'code_size', 'loc_blank',
                        'ck_oo_numberoflinesofcode', 'numberoflinesofcode']
        loc_col = None
        for target in possible_loc:
            for raw_col in df.columns:
                if raw_col.lower() == target:
                    loc_col = raw_col
                    break
            if loc_col:
                break

        if loc_col:
            loc_values = pd.to_numeric(df[loc_col], errors='coerce').fillna(1).values
        else:
            loc_values = np.ones(len(df))

        # --- 1.3 处理 Label ---
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].map(
                lambda x: 1 if str(x).strip().lower() in ['true', 'yes', 'buggy', '1', 'y'] else 0
            )
        else:
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0)
            df[label_col] = df[label_col].apply(lambda x: 1 if x > 0 else 0)

        y = df[label_col].values

        # --- 1.4 处理 Features ---
        X_df = df.drop(columns=[label_col])
        # 移除 LOC 列（如果在 Feature 中且与 Label 不同）避免信息泄漏
        X_df = X_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').fillna(0)
        X = X_df.values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # --- 1.5 切分数据 (同时切分 X, y, loc) ---
        try:
            return train_test_split(X, y, loc_values, test_size=0.2, random_state=42, stratify=y)
        except:
            return train_test_split(X, y, loc_values, test_size=0.2, random_state=42)
    except Exception as e:
        return None, None, None, None, None, None


# =========================================================================
#  全面不确定性计算函数
# =========================================================================
def calculate_comprehensive_uq(probs, ensemble_probs=None):
    metrics = {}
    epsilon = 1e-10

    # === 1. 基础分布指标 ===
    probs_safe = np.clip(probs, epsilon, 1. - epsilon)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    metrics['Entropy'] = np.mean(entropy)

    max_prob = np.max(probs, axis=1)
    metrics['Confidence'] = np.mean(max_prob)
    metrics['LeastConf'] = np.mean(1.0 - max_prob)

    if probs.shape[1] > 1:
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        p = probs.flatten()
        margin = np.abs(2 * p - 1)
    metrics['Margin'] = np.mean(margin)

    deep_gini = 1.0 - np.sum(probs ** 2, axis=1)
    metrics['DeepGini'] = np.mean(deep_gini)

    # === 2. 集成指标 ===
    if ensemble_probs is not None:
        if ensemble_probs.ndim == 2:
            p_pos = ensemble_probs[:, :, np.newaxis]
            p_neg = 1.0 - p_pos
            ensemble_probs = np.concatenate([p_neg, p_pos], axis=2)

        variance = np.var(ensemble_probs[:, :, 1], axis=0)
        metrics['Variance'] = np.mean(variance)

        ens_probs_safe = np.clip(ensemble_probs, epsilon, 1. - epsilon)
        member_entropies = -np.sum(ens_probs_safe * np.log(ens_probs_safe), axis=2)
        expected_entropy = np.mean(member_entropies, axis=0)
        metrics['ExpEntropy'] = np.mean(expected_entropy)

        mean_ens_prob = np.mean(ensemble_probs, axis=0)
        mean_ens_prob_safe = np.clip(mean_ens_prob, epsilon, 1. - epsilon)
        pred_entropy_ens = -np.sum(mean_ens_prob_safe * np.log(mean_ens_prob_safe), axis=1)
        bald = pred_entropy_ens - expected_entropy
        metrics['BALD'] = np.mean(bald)
    else:
        metrics['Variance'] = np.nan
        metrics['ExpEntropy'] = np.nan
        metrics['BALD'] = np.nan

    return metrics


# =========================================================================
#  ECE 计算函数
# =========================================================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prob_in_bin = y_prob[in_bin]
        if len(prob_in_bin) > 0:
            avg_confidence = np.mean(prob_in_bin)
            avg_accuracy = np.mean(y_true[in_bin])
            bin_weight = len(prob_in_bin) / n_total
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
    return ece


# =========================================================================
#  SDP 专用指标: Cost Effectiveness @ 20%
# =========================================================================
def calculate_ce_20(y_true, y_prob, loc):
    df = pd.DataFrame({'y': y_true, 'prob': y_prob, 'loc': loc})
    df = df.sort_values(by='prob', ascending=False)

    total_loc = df['loc'].sum()
    total_bugs = df['y'].sum()

    if total_bugs == 0:
        return 0.0
    if total_loc == 0:
        return 0.0

    df['cum_loc'] = df['loc'].cumsum()
    df['cum_bugs'] = df['y'].cumsum()

    cutoff_loc = 0.2 * total_loc
    mask = df['cum_loc'] <= cutoff_loc

    if mask.sum() == 0:
        bugs_found = df.iloc[0]['y']
    else:
        bugs_found = df[mask].iloc[-1]['cum_bugs']

    return bugs_found / total_bugs



# =========================================================================
#  Sklearn 模型库 — COD-selected 15 representative classifiers
# =========================================================================
def get_sklearn_models():
    """
    15 models selected via Classifier Output Difference (COD) +
    hierarchical clustering (threshold=0.085, average linkage).
    Each model represents a distinct prediction-behaviour cluster.
    """
    models = {}

    # Cluster 1: Bagged AdaBoost  (hybrid bagging+boosting)
    models['Bagged AdaBoost'] = BaggingClassifier(
        estimator=AdaBoostClassifier(n_estimators=10, random_state=42),
        n_estimators=10, random_state=42, n_jobs=1)

    # Cluster 2: Boosted-RS  (boosted decision stumps ≈ rule sets)
    models['Boosted-RS'] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50, random_state=42, algorithm='SAMME')

    # Cluster 3: LR  (represents LR/Ridge/LDA/linSVM/Bagging-LR/Boosting-LR)
    models['LR'] = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000, solver='lbfgs', random_state=42)

    # Cluster 4: Boosting-MLP  (AdaBoost + Perceptron)
    models['Boosting-MLP'] = AdaBoostClassifier(
        estimator=Perceptron(max_iter=200, random_state=42),
        n_estimators=20, random_state=42, algorithm='SAMME')

    # Cluster 5: NB  (represents NB/Bagging-NB)
    models['NB'] = GaussianNB()

    # Cluster 6: Boosting-NB  (AdaBoost + Naive Bayes)
    models['Boosting-NB'] = AdaBoostClassifier(
        estimator=GaussianNB(),
        n_estimators=50, random_state=42, algorithm='SAMME')

    # Cluster 7: KNN  (represents KNN/Kernel-KNN/rbfSVM/MLP1/MLP2/...)
    models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=1)

    # Cluster 8: RF  (represents RF/ExtraTrees/GBDT/LogitBoost)
    models['RF'] = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1)

    # Cluster 9: Voting  (soft voting over diverse base learners)
    models['Voting'] = VotingClassifier(
        estimators=[
            ('nb', GaussianNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42,
                                          n_jobs=1)),
        ],
        voting='soft', n_jobs=1)

    # Cluster 10: Boosting-DT  (classic AdaBoost + shallow tree)
    models['Boosting-DT'] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=50, random_state=42, algorithm='SAMME')

    # Cluster 11: Bagging-DT
    models['Bagging-DT'] = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10, random_state=42, n_jobs=1)

    # Cluster 12: C4.5 (J48)  (entropy-based decision tree)
    models['C4.5 (J48)'] = DecisionTreeClassifier(
        criterion='entropy', max_depth=10, random_state=42)

    # Cluster 13: Greedy-RL  (shallow entropy tree ≈ greedy rule list)
    models['Greedy-RL'] = DecisionTreeClassifier(
        criterion='entropy', max_depth=6, random_state=42)

    # Cluster 14: CART  (gini-based decision tree)
    models['CART'] = DecisionTreeClassifier(
        criterion='gini', max_depth=10, random_state=42)

    return models


# =========================================================================
#  单个数据集处理函数
# =========================================================================
def process_single_dataset(file_path):
    file_name = os.path.basename(file_path)
    # 加上所属子文件夹名作为分组标识，例如 "PROMISE/ant-1.7.csv"
    parent_folder = os.path.basename(os.path.dirname(file_path))
    dataset_label = f"{parent_folder}/{file_name}"
    print(f"  -> Processing: {dataset_label}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, X_test, y_train, y_test, loc_train, loc_test = load_and_preprocess(file_path)

    if X_train is None:
        print(f"     [SKIP] {dataset_label} (insufficient data or single class)")
        return []

    try:
        if np.sum(y_train == 1) > 1:
            k = min(np.sum(y_train == 1) - 1, 5)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except:
        X_train_res, y_train_res = X_train, y_train

    dataset_results = []

    # --- Sklearn 模型 ---
    for name, model in get_sklearn_models().items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)

                auc = 0.5
                ece = 0.0
                ce_20 = 0.0
                uq_metrics = {k: np.nan for k in
                              ['Entropy', 'Confidence', 'LeastConf', 'Margin', 'DeepGini',
                               'Variance', 'ExpEntropy', 'BALD']}

                if hasattr(model, "predict_proba"):
                    y_prob_all = model.predict_proba(X_test)
                    y_prob = y_prob_all[:, 1]

                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_prob)

                    ece = calculate_ece(y_test, y_prob)
                    ce_20 = calculate_ce_20(y_test, y_prob, loc_test)

                    # 提取集成概率 (用于计算 Variance / BALD 等集成 UQ 指标)
                    ensemble_probs_arr = None
                    base_model = model
                    # VotingClassifier 包装：提取内部 estimators
                    if isinstance(base_model, VotingClassifier):
                        preds = []
                        for _, est in base_model.named_estimators_.items():
                            if hasattr(est, "predict_proba"):
                                try:
                                    preds.append(est.predict_proba(X_test))
                                except:
                                    pass
                        if len(preds) >= 2:
                            ensemble_probs_arr = np.array(preds)
                    elif isinstance(base_model, RandomForestClassifier):
                        ensemble_probs_arr = np.array(
                            [tree.predict_proba(X_test) for tree in base_model.estimators_])
                    elif isinstance(base_model, BaggingClassifier):
                        preds = []
                        for estimator in base_model.estimators_:
                            if hasattr(estimator, "predict_proba"):
                                try:
                                    preds.append(estimator.predict_proba(X_test))
                                except:
                                    pass
                        if preds:
                            ensemble_probs_arr = np.array(preds)

                    uq_metrics = calculate_comprehensive_uq(y_prob_all, ensemble_probs_arr)

                mcc = matthews_corrcoef(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                precision = precision_score(y_test, y_pred, zero_division=0)

                res_dict = {
                    'Dataset': dataset_label,
                    'Model': name,
                    'AUC': auc,
                    'F1': f1,
                    'MCC': mcc,
                    'Recall(PD)': recall_val,
                    'FPR(PF)': fpr,
                    'Precision': precision,
                    'CE@20%': ce_20,
                    'ECE': ece
                }
                res_dict.update(uq_metrics)
                dataset_results.append(res_dict)

        except Exception as e:
            pass

    # --- Cluster 15: Boosting-SVM (AdaBoost + RBF SVM, SAMME) ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bsvm = AdaBoostClassifier(
                estimator=SVC(kernel='rbf', max_iter=3000, random_state=42),
                n_estimators=20, random_state=42, algorithm='SAMME')
            bsvm.fit(X_train_res, y_train_res)
            y_pred = bsvm.predict(X_test)

            # SAMME without predict_proba: use decision_function as score
            auc = 0.5
            ece = 0.0
            ce_20 = 0.0
            uq_metrics = {k: np.nan for k in
                          ['Entropy', 'Confidence', 'LeastConf', 'Margin', 'DeepGini',
                           'Variance', 'ExpEntropy', 'BALD']}

            if hasattr(bsvm, "decision_function"):
                scores = bsvm.decision_function(X_test)
                # Convert decision scores to pseudo-probabilities via sigmoid
                pseudo_prob = 1.0 / (1.0 + np.exp(-scores))
                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, pseudo_prob)
                ece = calculate_ece(y_test, pseudo_prob)
                ce_20 = calculate_ce_20(y_test, pseudo_prob, loc_test)
                probs_all = np.vstack([1 - pseudo_prob, pseudo_prob]).T
                uq_metrics = calculate_comprehensive_uq(probs_all, None)

            mcc = matthews_corrcoef(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            prec = precision_score(y_test, y_pred, zero_division=0)

            res_dict = {
                'Dataset': dataset_label,
                'Model': 'Boosting-SVM',
                'AUC': auc, 'F1': f1, 'MCC': mcc,
                'Recall(PD)': recall_val, 'FPR(PF)': fpr_val,
                'Precision': prec, 'CE@20%': ce_20, 'ECE': ece
            }
            res_dict.update(uq_metrics)
            dataset_results.append(res_dict)
    except Exception as e:
        pass

    return dataset_results


# =========================================================================
#  主程序入口
# =========================================================================
if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    if not os.path.exists(root_dir):
        print(f"Error: dataset folder not found at {root_dir}")
        exit(1)

    # 收集所有 .csv 和 .arff 文件
    all_files = []
    for r, d, fs in os.walk(root_dir):
        for f in fs:
            if f.endswith('.csv') or f.endswith('.arff'):
                all_files.append(os.path.join(r, f))

    print(f"=" * 70)
    print(f"  SDP Benchmark (Full Metrics + Uncertainty Quantification)")
    print(f"=" * 70)
    print(f"  Dataset folder : {root_dir}")
    print(f"  Files found    : {len(all_files)}")

    # 按子文件夹分类打印
    folder_counts = {}
    for fp in all_files:
        folder = os.path.basename(os.path.dirname(fp))
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    for folder, count in sorted(folder_counts.items()):
        print(f"    - {folder}: {count} files")
    print(f"=" * 70)
    print()

    warnings.filterwarnings("ignore")

    # 并行处理 (n_jobs 按 CPU 核心数调整，避免过多)
    n_jobs = min(os.cpu_count() or 4, 16)
    results_lists = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_dataset)(f) for f in all_files
    )

    final_results = [item for sublist in results_lists for item in sublist]

    if final_results:
        output_file = os.path.join("data", "benchmark_results_IVDP_FullUQ.csv")
        df = pd.DataFrame(final_results)

        cols = ['Dataset', 'Model', 'AUC', 'F1', 'MCC', 'Recall(PD)', 'FPR(PF)', 'Precision',
                'CE@20%', 'ECE',
                'Entropy', 'Confidence', 'LeastConf', 'Margin', 'DeepGini',
                'Variance', 'ExpEntropy', 'BALD']

        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        df = df[cols]
        df.to_csv(output_file, index=False)

        print(f"\n{'=' * 70}")
        print(f"  DONE!")
        print(f"  Total results : {len(final_results)} rows")
        print(f"  Datasets used : {df['Dataset'].nunique()}")
        print(f"  Models tested : {df['Model'].nunique()}")
        print(f"  Output file   : {output_file}")
        print(f"{'=' * 70}")
    else:
        print("\nNo results produced. Check dataset files.")
