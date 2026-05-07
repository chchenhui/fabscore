import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def parse_single_excel_file(filepath):
    """
    智能解析单个iNSPiRe Excel文件，精确提取并关联'改造前' (Reference) 和
    '改造后' (Target) 的数据，并计算节能相关指标。

    参数:
        filepath (str): 指向 .xlsx 文件的完整路径。

    返回:
        pandas.DataFrame: 一个包含所有建筑参数以及以下关键列的DataFrame:
                          - consumption_before (改造前能耗)
                          - consumption_after (改造后能耗)
                          - savings_percentage (节能百分比)
                          如果文件处理失败或不包含所需数据，则返回 None。
    """
    filename = os.path.basename(filepath)
    print(f"--- 正在处理文件: {filename} ---")
    
    try:
        # 1. 精确提取 "改造前" (Reference) 数据
        # 根据您提供的文件，我们现在知道表头在第二行 (header=1)
        df_ref = pd.read_excel(filepath, sheet_name='Reference buildings simulations', header=0).dropna(how='all')
        
        required_ref_cols = ['Climate', 'Type of building', 'Age of construction', 'Consumption (kWh/m2y)']
        if not all(col in df_ref.columns for col in required_ref_cols):
            print(f"  [警告] 在 {filename} 的 'Reference' 工作表中缺少必要的列。")
            return None
            
        df_ref = df_ref[required_ref_cols].copy()
        df_ref.rename(columns={'Consumption (kWh/m2y)': 'consumption_before'}, inplace=True)

        # 2. 精确提取 "改造后" (Target) 数据
        # 根据您提供的文件，这个表的复杂表头意味着实际数据表头在第四行 (header=3)
        df_target = pd.read_excel(filepath, sheet_name='Target buildings simulations', header=1).dropna(how='all')
        
        consumption_after_col = 'Building average yearly'
        if consumption_after_col not in df_target.columns:
            print(f"  [警告] 在 {filename} 的 'Target' 工作表中未找到能耗列: '{consumption_after_col}'。")
            return None
            
        df_target.rename(columns={consumption_after_col: 'consumption_after', 'Building type': 'Type of building'}, inplace=True)

        # 3. 关联 "改造前" 和 "改造后" 的数据
        merge_keys = ['Climate', 'Type of building', 'Age of construction']
        if not all(key in df_target.columns and key in df_ref.columns for key in merge_keys):
            print(f"  [警告] 在 {filename} 的两个工作表中缺少用于关联的共同键。")
            return None
            
        df_merged = pd.merge(df_target, df_ref, on=merge_keys, how='inner')
        
        if df_merged.empty:
            print(f"  [警告] 在 {filename} 中未能找到任何匹配的'改造前'和'改造后'记录。")
            return None
        
        # 4. 计算节能率
        df_merged['consumption_before'] = pd.to_numeric(df_merged['consumption_before'], errors='coerce')
        df_merged.dropna(subset=['consumption_before'], inplace=True)
        df_merged = df_merged[df_merged['consumption_before'] > 0]
        
        df_merged['savings_percentage'] = (df_merged['consumption_before'] - df_merged['consumption_after']) / df_merged['consumption_before']
        
        # 清理节能率异常的数据
        df_merged = df_merged[(df_merged['savings_percentage'] >= 0) & (df_merged['savings_percentage'] <= 1)]

        print(f"  -> 成功为 {filename} 关联并计算了 {len(df_merged)} 条改造记录。")
        return df_merged

    except Exception as e:
        print(f"  [错误] 处理文件 {filename} 时发生意外错误: {e}。")
        return None

def process_all_files_in_directory(root_dir, output_csv_path):
    """
    扫描指定目录下的所有 .xlsx 文件，调用解析函数，并将所有结果合并保存。
    """
    print(f"\n--- 步骤 1: 开始扫描文件夹中的所有Excel文件 ---")
    print(f"   源文件夹: {root_dir}")

    # 收集所有从单个文件中成功解析出的数据框
    all_dfs = [parse_single_excel_file(os.path.join(root_dir, f)) 
               for f in os.listdir(root_dir) if f.endswith('.xlsx') and not f.startswith('~')]

    # 过滤掉解析失败的结果 (None)
    valid_dfs = [df for df in all_dfs if df is not None and not df.empty]
    
    if not valid_dfs:
        print("\n❌ 错误: 未能从任何文件中提取有效的改造前后数据。请检查文件内容和结构。")
        return

    # 将所有有效的数据框合并成一个最终的数据集
    final_dataset = pd.concat(valid_dfs, ignore_index=True)
    
    # 保存最终的数据集
    final_dataset.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print("\n--- 步骤 2: 数据整合完成！ ---")
    print(f"✅ 成功处理并合并了 {len(valid_dfs)} 个Excel文件。")
    print(f"✅ 生成的最终数据集包含 {len(final_dataset)} 行。")
    print(f"✅ 新的训练集已保存至: {output_csv_path}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义包含所有xlsx文件的根目录
    source_directory = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\iNSPiRe FP7 Retrofit Database'
    
    # 2. 定义您希望保存最终整合后CSV文件的路径和名称
    output_filepath = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\consolidated_retrofit_data.csv'
    
    # 3. 运行主函数
    process_all_files_in_directory(source_directory, output_filepath)
    
    print("\n🎉 脚本执行完毕！")