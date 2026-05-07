
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置区域 ---

# 定义输入文件和输出图表的路径
RESULTS_FILE = os.path.join('Qwen3-1.7B_results', 'llm_judge_results.json')
OUTPUT_CHART = 'radar_chart.pdf'

# 定义维度和方法的顺序，以确保图表和表格的一致性
DIMENSIONS = ['plausibility', 'structure_clarity', 'innovation_potential']
# 确保 'cgmcts' 在列表前面，以便在图表中突出显示
METHOD_ORDER = ['cgmcts', 'cot', 'react', 'tot', 'simple']

# --- 核心功能函数 ---

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """加载JSON结果文件，并将其解析为扁平化的Pandas DataFrame。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：结果文件未找到于 '{filepath}'")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 '{filepath}'")
        return pd.DataFrame()

    parsed_records = []
    for item in data:
        theme = item.get('theme')
        judgment = item.get('judgment', {})
        evaluations = judgment.get('evaluations', {})
        final_decision = judgment.get('final_decision', {})
        winner_letter = final_decision.get('best_proposal')

        # 确定获胜者的方法名称
        winner_method = evaluations.get(winner_letter, {}).get('method')

        for _, eval_data in evaluations.items():
            method = eval_data.get('method')
            if not method: continue
            
            record = {
                'theme': theme,
                'method': method,
                'is_winner': 1 if method == winner_method else 0
            }
            for dim in DIMENSIONS:
                record[dim] = eval_data.get(dim)
            
            parsed_records.append(record)
            
    return pd.DataFrame(parsed_records)

def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """根据DataFrame计算指标并生成最终的汇总表格。"""
    # 按方法分组
    grouped = df.groupby('method')

    # 计算平均值和标准差
    means = grouped[DIMENSIONS].mean()
    stds = grouped[DIMENSIONS].std()

    # 创建格式化的 Avg. ± Std. 列
    summary_df = pd.DataFrame()
    for dim in DIMENSIONS:
        summary_df[f'{dim.replace(" ", " ").title()} (Avg. ± Std.)'] = \
            means[dim].map('{:.2f}'.format) + ' ± ' + stds[dim].map('{:.2f}'.format)

    # 计算综合得分
    df['overall_score'] = df[DIMENSIONS].mean(axis=1)
    overall_scores = df.groupby('method')['overall_score'].mean()
    summary_df['Overall Score (Avg.)'] = overall_scores.map('{:.2f}'.format)

    # 计算获胜率
    total_themes = df['theme'].nunique()
    win_counts = grouped['is_winner'].sum()
    win_rates = (win_counts / total_themes) * 100
    summary_df['Win Rate (%)'] = win_rates.map('{:.1f}%'.format)

    # 调整列和行的顺序
    summary_df = summary_df.reindex(METHOD_ORDER)
    return summary_df

def plot_radar_chart(df: pd.DataFrame, filepath: str):
    """生成并保存雷达图。"""
    # 计算每个方法在各维度上的平均分
    avg_scores = df.groupby('method')[DIMENSIONS].mean().reindex(METHOD_ORDER)

    labels = [label.replace('_', ' ').title() for label in DIMENSIONS]
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合曲线

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 设置雷达图背景颜色
    ax.set_facecolor('#f0f0f0') # 浅灰色背景

    # 定义颜色循环，确保 'cgmcts' 有一个突出颜色
    # Using a custom list of distinct colors for better visual separation
    custom_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00'] # Red, Blue, Green, Purple, Orange

    for i, (method, row) in enumerate(avg_scores.iterrows()):
        stats = row.tolist()
        stats += stats[:1] # 闭合曲线

        # 为 cgmcts 使用更突出和更粗的线条
        is_ours = (method == 'cgmcts')
        linewidth = 3 if is_ours else 1.5
        alpha_fill = 0.06 if is_ours else 0.03 # 更低的填充透明度，避免覆盖
        color = custom_colors[i % len(custom_colors)] # Assign colors from custom_colors

        ax.plot(angles, stats, linewidth=linewidth, linestyle='solid', label=method, color=color)
        ax.fill(angles, stats, alpha=alpha_fill, color=color)

    # 设置图表样式
    ax.set_yticklabels([]) # 隐藏半径轴的刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_rlabel_position(30)

    # 调整y轴刻度从4开始
    yticks_labels = [4, 5, 6, 7, 8, 9, 10]
    plt.yticks(yticks_labels, [str(y) for y in yticks_labels], color="grey", size=10)
    plt.ylim(4, 10) # Y轴范围从4到10

    ax.set_title('Multi-Dimensional Performance Comparison', size=16, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fancybox=True, shadow=True)

    # 保存为高质量的PDF文件
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"\n雷达图已保存至: {filepath}")


# --- 主执行函数 ---

def main():
    """主函数，协调数据加载、分析和可视化。"""
    # 1. 加载和解析数据
    df = load_and_parse_data(RESULTS_FILE)
    if df.empty:
        return

    # 2. 创建并打印汇总表格
    summary_table = create_summary_table(df)
    print("--- 实验结果核心表格 ---")
    print(summary_table.to_string())
    print("---------------------------")

    # 3. 绘制并保存雷达图
    plot_radar_chart(df, OUTPUT_CHART)

if __name__ == "__main__":
    main()
