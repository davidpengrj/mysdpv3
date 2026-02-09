import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'figures')

# === 1. 绘图风格设置 ===
# 保持使用 'poster' 风格，这样字体和线条默认都会比较粗大，适合论文/PPT
sns.set(style="whitegrid", context="poster") 

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 适配中文
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['pdf.fonttype'] = 42 # 确保PDF文本可编辑

def plot_metric_effectiveness_ranking(file_path, target_perf_metric):
    """
    生成指标有效性排名箱线图 (去除了左下角/右上角的文字注释)
    """
    print(f"\n>>> 正在处理性能指标: {target_perf_metric} ...")

    # --- 数据加载 ---
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # --- 定义 UQ 指标 ---
    uq_metrics_candidates = [
        'Entropy', 'Confidence', 'LeastConf', 'Margin', 
        'DeepGini', 'Variance', 'ExpEntropy', 'BALD'
    ]
    uq_metrics = [m for m in uq_metrics_candidates if m in df.columns]
    
    if target_perf_metric not in df.columns:
        print(f"⚠️ 跳过: CSV 中未找到列 '{target_perf_metric}'")
        return

    # --- 计算 Spearman 相关性 ---
    correlation_data = []
    models = df['Model'].unique()
    
    for uq in uq_metrics:
        for model in models:
            sub_df = df[df['Model'] == model][[uq, target_perf_metric]].dropna()
            if len(sub_df) > 5:
                corr, _ = spearmanr(sub_df[uq], sub_df[target_perf_metric])
                correlation_data.append({
                    'UQ Metric': uq,
                    'Correlation': corr,
                    'Model': model
                })
    
    res_df = pd.DataFrame(correlation_data)
    if res_df.empty:
        print(f"⚠️ 无法计算 {target_perf_metric} 的相关性。")
        return

    # --- 绘图逻辑 ---
    # 画布保持较大尺寸 (16x10)，保证清晰度
    plt.figure(figsize=(16, 10))
    
    # 排序：按中位数从小到大排序
    order = res_df.groupby('UQ Metric')['Correlation'].median().sort_values().index
    
    # 绘制箱线图 - 线宽设为 2.5，非常清晰
    sns.boxplot(x='UQ Metric', y='Correlation', data=res_df, order=order, 
                palette="RdBu_r", showfliers=False, width=0.6, linewidth=2.5)
    
    # 添加散点 - 颜色加深，大小适中
    sns.stripplot(x='UQ Metric', y='Correlation', data=res_df, order=order,
                  color='#333333', alpha=0.6, jitter=True, size=8)
    
    # 0刻度参考线
    plt.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    # --- 标题和标签 (已移除 plt.text 注释) ---
    plt.title(f'Metric Effectiveness: Spearman Correlation with {target_perf_metric}', 
              fontsize=24, fontweight='bold', pad=25)
    
    plt.ylabel(f'Spearman Correlation (vs {target_perf_metric})', fontsize=20, labelpad=15)
    plt.xlabel('Uncertainty Quantification Metrics', fontsize=20, labelpad=15)
    
    # 坐标轴刻度字体
    plt.xticks(rotation=30, fontsize=16)
    plt.yticks(fontsize=16)

    # --- 保存 ---
    safe_name = target_perf_metric.replace('(', '').replace(')', '').replace('/', '_')
    output_file = os.path.join(FIG_DIR, f"Experiment1_Ranking_{safe_name}.pdf")
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✅ 图表已保存: {output_file}")
    plt.close()

if __name__ == "__main__":
    file_name = os.path.join(DATA_DIR, "benchmark_results_IVDP_FullUQ.csv")
    
    target_metrics = ['MCC', 'F1', 'AUC', 'Recall(PD)', 'Precision', 'FPR(PF)']
    
    print(f"🚀 开始批量生成 6 张图表...")
    for metric in target_metrics:
        plot_metric_effectiveness_ranking(file_name, target_perf_metric=metric)
    print("\n🎉 所有图表生成完毕！")