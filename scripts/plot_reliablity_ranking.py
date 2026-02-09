import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.ticker as ticker
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')

# --- 1. 设置出版级绘图风格 (Top-Tier Conference Style) ---
sns.set_theme(style="ticks", context="talk")

# 字体配置
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置：深灰文本，视觉更柔和
text_color = '#333333'
plt.rcParams['text.color'] = text_color
plt.rcParams['axes.labelcolor'] = text_color
plt.rcParams['xtick.color'] = text_color
plt.rcParams['ytick.color'] = text_color

# 配色方案 (Tableau 风格，高级感)
color_reliable = "#4e79a7"  # 蓝色
color_unreliable = "#e15759" # 红色

def plot_independent_rankings(csv_path, output_base_name):
    # 读取数据
    try:
        df = pd.read_csv(csv_path)
        if 'FPR' not in df.columns:
            potential_fpr = [col for col in df.columns if 'FPR' in col or 'False Alarm' in col]
            if potential_fpr:
                print(f"警告: 未找到精确的 'FPR' 列。将使用 '{potential_fpr[0]}' 代替。")
                df.rename(columns={potential_fpr[0]: 'FPR'}, inplace=True)
            else:
                 print("警告: 数据中未找到 'FPR' 相关列，该指标将被跳过。")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    metrics = ['AUC', 'F1', 'MCC', 'Recall(PD)', 'Precision', 'FPR']
    uncertainty_col = 'Entropy'
    valid_metrics = [m for m in metrics if m in df.columns]

    if not valid_metrics:
        print("未在数据中找到任何指定的指标列。")
        return

    for metric in metrics:
        if metric not in valid_metrics: continue
        print(f"正在处理指标: {metric}...")

        # 数据计算
        sub_df = df.dropna(subset=[metric, uncertainty_col])
        model_corrs = []
        for model in sub_df['Model'].unique():
            m_df = sub_df[sub_df['Model'] == model]
            if len(m_df) < 5: continue
            r, _ = spearmanr(m_df[uncertainty_col], m_df[metric])
            if np.isnan(r): r = 0
            model_corrs.append({'Model': model, 'Correlation': r})

        if not model_corrs: continue

        corr_df = pd.DataFrame(model_corrs).sort_values('Correlation')

        # --- 配色逻辑 ---
        is_error_metric = (metric == 'FPR')
        if is_error_metric:
            corr_df['color'] = corr_df['Correlation'].apply(
                lambda x: color_reliable if x > 0 else color_unreliable)
            display_title = "Uncertainty vs FPR"
        else:
            corr_df['color'] = corr_df['Correlation'].apply(
                lambda x: color_reliable if x < 0 else color_unreliable)
            display_title = f"Uncertainty vs {metric}"

        # --- 绘图 (适配 15 个模型) ---
        n_models = len(corr_df)
        fig_h = max(6, n_models * 0.5)
        fig, ax = plt.subplots(figsize=(11, fig_h))

        sns.barplot(
            x='Correlation',
            y='Model',
            hue='Model',
            data=corr_df,
            palette=dict(zip(corr_df['Model'], corr_df['color'])),
            ax=ax,
            edgecolor=None,
            linewidth=0,
            zorder=3,
            alpha=0.9,
            legend=False,
        )

        # --- 美学细节 ---

        # 1. 极简边框：去左、上、右
        sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
        
        # 2. 网格线：极淡，置于底层
        ax.grid(True, axis='x', color='#EAEAF2', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
        
        # 3. 零刻度线
        ax.axvline(0, color='#888888', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

        # 4. 标题：简洁大方
        ax.set_title(display_title, fontsize=24, fontweight='bold', pad=30, color='#222222', loc='center')

        # 5. X轴调整
        ax.set_xlabel('Spearman Correlation', fontsize=16, labelpad=15, fontweight='medium')
        ax.tick_params(axis='x', labelsize=14, color='#CCCCCC') 
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # 6. Y轴调整 (字体随模型数量自适应)
        ax.set_ylabel('')
        y_fontsize = 20 if n_models <= 15 else (16 if n_models <= 20 else 13)
        ax.tick_params(axis='y', labelsize=y_fontsize, length=0, left=False, pad=10)
        
        plt.tight_layout()

        # 保存
        base, ext = os.path.splitext(output_base_name)
        metric_clean = metric.replace('(', '').replace(')', '')
        current_output_file = f"{base}_{metric_clean}{ext}"

        plt.savefig(current_output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"  Saved: {current_output_file}")
        plt.close(fig)

# --- 执行 ---
file_path = os.path.join(DATA_DIR, 'benchmark_results_IVDP_FullUQ.csv')
output_base = os.path.join(FIG_DIR, 'reliability_ranking.pdf')

if __name__ == "__main__":
    plot_independent_rankings(file_path, output_base)