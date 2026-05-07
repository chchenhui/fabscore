import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def parse_retrofit_excel_definitive(filepath):
    """
    最终版的解析器，使用高精度方法来关联'改造前'和'改造后'的数据。
    """
    filename = os.path.basename(filepath)
    print(f"--- 正在处理: {filename} ---")
    
    try:
        # 1. 精确提取 "改造前" (Reference) 数据
        # 根据您提供的文件结构，我们现在知道表头在第二行 (header=1)
        df_ref = pd.read_excel(filepath, sheet_name='Reference buildings simulations', header=0).dropna(how='all')
        
        required_ref_cols = ['Climate', 'Type of building', 'Age of construction', 'Consumption (kWh/m2y)']
        if not all(col in df_ref.columns for col in required_ref_cols):
            print(f"  [警告] 在 {filename} 的 'Reference' 工作表中缺少必要的列。")
            return None
            
        df_ref = df_ref[required_ref_cols].copy()
        df_ref.rename(columns={'Consumption (kWh/m2y)': 'consumption_before'}, inplace=True)

        # 2. 精确提取 "改造后" (Target) 数据
        # 根据您提供的文件结构，这个表的复杂表头意味着实际数据表头在第四行 (header=3)
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
        
        # 4. 计算节能率 (我们的目标变量)
        df_merged['consumption_before'] = pd.to_numeric(df_merged['consumption_before'], errors='coerce')
        df_merged.dropna(subset=['consumption_before'], inplace=True)
        df_merged = df_merged[df_merged['consumption_before'] > 0]
        
        df_merged['energy_savings_percentage'] = (df_merged['consumption_before'] - df_merged['consumption_after']) / df_merged['consumption_before']
        
        print(f"  -> 成功关联并计算了 {len(df_merged)} 条改造记录。")
        return df_merged

    except Exception as e:
        print(f"  [错误] 处理文件 {filename} 时发生意外错误: {e}。")
        return None

# --- 主程序 ---
# 1. 创建整合的数据集
root_dir = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\iNSPiRe FP7 Retrofit Database'
output_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\retrofit_savings_dataset_final.csv'

print("--- 步骤 1: 正在从所有Excel文件中创建改造节能数据集 (最终版) ---")
all_dfs = [parse_retrofit_excel_definitive(os.path.join(root_dir, f)) 
           for f in os.listdir(root_dir) if f.endswith('.xlsx') and not f.startswith('~')]

valid_dfs = [df for df in all_dfs if df is not None and not df.empty]
if not valid_dfs:
    print("\n❌ 错误: 未能从任何文件中提取有效的改造前后数据。请检查您的Excel文件是否都包含 'Reference buildings simulations' 和 'Target buildings simulations' 这两个工作表，并且格式正确。")
else:
    final_dataset = pd.concat(valid_dfs, ignore_index=True)
    # 对最终数据集进行清洗，确保节能率在0%到100%之间
    final_dataset = final_dataset[(final_dataset['energy_savings_percentage'] >= 0) & (final_dataset['energy_savings_percentage'] <= 1)]
    final_dataset.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ 改造节能数据集创建完成！共 {len(final_dataset)} 条有效记录。")
    print(f"   数据已保存至: {output_path}")

    # 2. 进行建模
    print("\n\n--- 步骤 2: 开始训练模型以预测节能率 ---")
    df_model = final_dataset.copy()
    target_col = 'energy_savings_percentage'
    
    # 建模前移除中间计算列
    cols_to_drop_modeling = [col for col in ['consumption_before', 'consumption_after', target_col] if col in df_model.columns]
    X = df_model.drop(columns=cols_to_drop_modeling)
    y = df_model[target_col]
    
    # 自动移除那些只有一个唯一值的列（因为它们没有预测能力）
    single_value_cols = [col for col in X.columns if X[col].nunique() <= 1]
    X.drop(columns=single_value_cols, inplace=True)
    print(f"   移除了 {len(single_value_cols)} 个无信息量的特征列。")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ 数据已拆分为 {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本。")
    
    # 自动识别数值型和分类型特征
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # 为不同类型的特征创建不同的预处理流程
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # 将两种流程整合到一个预处理器中
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')
    
    # 构建最终的XGBoost模型管道
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])
    
    print("\n--- 步骤 3: 正在训练XGBoost模型 ---")
    model_pipeline.fit(X_train, y_train)
    print("✅ 模型训练完成！")
    
    print("\n--- 步骤 4: 评估模型性能 ---")
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"📈 最终模型性能评估:")
    print(f"   - R² 分数 (R-squared): {r2:.4f}")
    print(f"   - 平均绝对误差 (MAE): {mae*100:.2f} 个百分点") # 乘以100，结果更直观
    
    print("\n--- 步骤 5: 分析特征重要性 ---")
    try:
        # 从管道中提取处理后的特征名和模型的重要性分数
        cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
        all_feature_names = numeric_features + cat_feature_names
        
        importances = model_pipeline.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)

        print("对'节能率'预测最重要的15个特征:")
        print(feature_importance_df)
    except Exception as e:
        print(f"无法提取特征重要性: {e}")

    print("\n🎉 恭喜您！已完成从原始数据到最终节能率预测模型的全部流程！")