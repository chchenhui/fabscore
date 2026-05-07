import pandas as pd
import os

def calculate_retrofit_savings_with_features(baseline_filepath, upgrade_filepath, output_filepath):
    """
    加载'改造前'和'改造后'的数据集，计算节能率，并保留指定的建筑特征。

    参数:
        baseline_filepath (str): 指向 baseline...csv 文件的路径。
        upgrade_filepath (str): 指向 upgrade...csv 文件的路径。
        output_filepath (str): 保存最终结果的路径。
    """
    print("--- 任务开始: 正在计算建筑能耗前后对比并保留关键特征 ---")
    
    # --- 步骤 1: 加载数据 ---
    try:
        print(f"   -> 正在加载'改造前'数据: {os.path.basename(baseline_filepath)}")
        df_baseline = pd.read_csv(baseline_filepath)
        
        print(f"   -> 正在加载'改造后'数据: {os.path.basename(upgrade_filepath)}")
        df_upgrade = pd.read_csv(upgrade_filepath)
        
        print("✅ 数据加载成功！")
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件 {e.filename}。请确保文件路径和名称完全正确。")
        return

    # --- 步骤 2: 准备用于分析的关键列 ---
    
    # **第一步：定义列名映射规则**
    feature_rename_map = {
        'in.ashrae_iecc_climate_zone_2004': 'climate',
        'in.geometry_building_type_recs': 'btype',
        'in.vintage': 'period',
        'in.sqft': 'area_sqft', # 稍后转换为平方米
        'in.hvac_heating_type_and_fuel': 'system'
    }
    energy_col = 'out.site_energy.total.energy_consumption.kwh'
    
    # **第二步：从 Baseline 文件中提取特征和改造前能耗**
    required_baseline_cols = ['bldg_id', energy_col] + list(feature_rename_map.keys())
    if not all(col in df_baseline.columns for col in required_baseline_cols):
        missing = [col for col in required_baseline_cols if col not in df_baseline.columns]
        print(f"❌ 错误: 'baseline'文件中缺少必要的列: {missing}。")
        return
        
    df_baseline_features = df_baseline[required_baseline_cols].copy()
    df_baseline_features.rename(columns=feature_rename_map, inplace=True)
    df_baseline_features.rename(columns={energy_col: 'consumption_before'}, inplace=True)
    df_baseline_features['area_m2'] = df_baseline_features['area_sqft'] * 0.092903
    
    # **第三步：从 Upgrade 文件中提取改造后能耗**
    if 'bldg_id' not in df_upgrade.columns or energy_col not in df_upgrade.columns:
        print(f"❌ 错误: 'upgrade'文件中缺少 'bldg_id' 或 '{energy_col}' 列。")
        return
    df_upgrade_slim = df_upgrade[['bldg_id', energy_col]].rename(columns={energy_col: 'consumption_after'})

    # --- 步骤 3: 关联'改造前'和'改造后'的数据 ---
    print("\n--- 步骤 2: 正在通过建筑ID关联数据 ---")
    # 先合并特征和改造前能耗，再合并改造后能耗
    df_merged = pd.merge(df_baseline_features, df_upgrade_slim, on='bldg_id', how='inner')
    
    if df_merged.empty:
        print("⚠️ 警告: 未能在两个文件中找到任何共有的建筑ID。请确保文件匹配。")
        return
        
    print(f"✅ 成功关联了 {len(df_merged)} 栋建筑。")

    # --- 步骤 4: 计算节能数据 ---
    print("\n--- 步骤 3: 正在计算节能数据 ---")
    df_merged['energy_saved_kwh'] = df_merged['consumption_before'] - df_merged['consumption_after']
    df_merged['savings_percentage'] = df_merged.apply(
        lambda row: (row['energy_saved_kwh'] / row['consumption_before']) * 100 if row['consumption_before'] > 0 else 0,
        axis=1
    )
    print("✅ 节能率计算完成。")
    
    # --- 步骤 5: 整理并保存最终结果 ---
    # 定义最终输出的列和顺序
    final_columns = [
        'bldg_id', 'climate', 'btype', 'period', 'area_m2', 'system',
        'consumption_before', 'consumption_after', 'energy_saved_kwh', 'savings_percentage'
    ]
    df_final = df_merged[final_columns]
    
    df_final.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"\n--- 任务完成！ ---")
    print(f"✅ 分析结果已成功保存至: {output_filepath}")
    
    # 显示结果预览
    print("\n--- 结果预览 (前5行) ---")
    print(df_final.head())


# --- 如何使用这个脚本 ---
if __name__ == "__main__":
    # 1. 确保您的 'baseline' 文件位于此路径
    baseline_file = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\ResStock\baseline_metadata_and_annual_results.csv'
    
    # 2. 确保您的 'upgrade01' 文件位于此路径
    upgrade_file = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\ResStock\upgrade01_metadata_and_annual_results.csv'
    
    # 3. 定义您希望保存最终结果的路径和文件名
    output_file = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\retrofit_savings_comparison_with_features.csv'
    
    # 4. 运行主函数
    if not os.path.exists(baseline_file) or not os.path.exists(upgrade_file):
        print(f"✋ 请注意: 请确保 baseline 和 upgrade 文件都已下载并放置在正确的路径下。")
    else:
        calculate_retrofit_savings_with_features(baseline_file, upgrade_file, output_file)