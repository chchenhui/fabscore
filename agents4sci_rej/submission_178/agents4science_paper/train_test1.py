import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- 步骤 1: 加载并抽样数据集 ---
print("--- 步骤 1: 正在加载并抽样数据集 ---")
data_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\consolidated_retrofit_data.csv'

try:
    df_full = pd.read_csv(data_path)
    df_sample = df_full.sample(frac=0.2, random_state=42)
    print(f"✅ 数据加载并抽样完成，共 {len(df_sample)} 条记录用于分析。")
except FileNotFoundError:
    print(f"❌ 错误: 找不到数据集 '{data_path}'。")
    exit()

# --- 步骤 2: 使用“特征白名单”精确定义特征和目标 ---
print("\n--- 步骤 2: 正在使用“特征白名单”精确定义特征和目标 ---")
target_col = 'consumption_after'

# **--- 关键修正：定义一个“白名单”，只包含真正的输入参数 ---**
# 这个列表是基于您提供的完整列名精心挑选的，只包括建筑物理特性和改造措施
allowlist_features = [
    'Climate', 'Building type', 'Age of construction', 'Retrofit energy level heating demand',
    'Heating set temperature', 'Cooling set temperature', 'Building Orientation',
    'Solar thermal coll. area', 'Solar thermal coll. Inclination', 'Solar PV pan. area',
    'Solar PV pan. Inclination', 'H&C device', 'Distribution sys.',
    'Distribution sys. Heating temp. supply', 'Distribution sys. Cooling temp. supply',
    'Living area', 'Ground/Cellar area', 'Façade area', 'Perimeter area', 'Roof area',
    'Building inner volume', 'Windows area', 'Ground/Cellar insulation thickness',
    'Façade insulation thickness', 'Perimeter insulation thickness', 'roof insulation thickness',
    'Windows galzing type', 'MECH VENT EFFICIENCY', 'AIR CHANGE RATE', 'VENTILATION RATE',
    'STORAGE VOLUME', 'Solar pipes diameter', 'Solar pipes thickness', 'Solar pipes length',
    'solar pipes insulation thickness', 'Quantity of glycol', 'solar expansion vessels',
    'Thermal energy source', 'HEATING CAPACITY', 'COOLING CAPACITY', 'Split Units total capacity',
    'BUFFER VOLUME', 'Pipes diameter', 'Pipes thickness', 'Pipes length',
    'Pipes insulation thickness', 'Expansion vessels', 'Total volume flow pumps', 'n° pumps',
    'Distribution sys. Capacity', 'Distribution sys. Area - GF', 'Distribution sys. Area - F1'
]

# 确保只选择数据集中实际存在的“白名单”特征
X = df_sample[[col for col in allowlist_features if col in df_sample.columns]]
y = df_sample[target_col]

print(f"   已定义预测目标: '{target_col}'")
print(f"   最终使用的特征数量: {len(X.columns)} (已通过白名单筛选)")

# --- 步骤 3, 4, 5, 6, 7 (与之前脚本类似，但现在基于无泄漏数据) ---
# 3. 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n--- 步骤 3: 数据已拆分为 {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本 ---")

# 4. 构建预处理流程
print("\n--- 步骤 4: 正在构建预处理流程 ---")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='passthrough')
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])

# 5. 训练“诚实”的模型
print("\n--- 步骤 5: 正在训练最终的“诚实”模型 ---")
model_pipeline.fit(X_train, y_train)
print("✅ 模型训练完成！")

# 6. 评估模型的真实性能
print("\n--- 步骤 6: 正在评估模型的真实性能 ---")
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"📈 最终模型性能评估:")
print(f"   - 真实 R² 分数 (R-squared): {r2:.4f}")
print(f"   - 平均绝对误差 (MAE): {mae:.2f} kWh/m2y")

# 7. 分析真实的特征重要性
print("\n--- 步骤 7: 正在分析真实的特征重要性 ---")
try:
    cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)
    print("对'改造后能耗'预测最重要的15个特征 (真实洞察):")
    print(feature_importance_df)
except Exception as e:
    print(f"无法提取特征重要性: {e}")

print("\n🎉 恭喜您！这代表了基于当前数据的最可靠、最真实的模型分析结果！")