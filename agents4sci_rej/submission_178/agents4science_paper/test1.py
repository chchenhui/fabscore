import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 步骤 1: 加载模拟 (Sim) 和真实 (Real) 数据集 ---
print("--- 步骤 1: 正在加载模拟 (Sim) 和真实 (Real) 数据集 ---")
try:
    # 加载我们整合好的欧洲模拟数据 (Sim)，这将是我们的训练集
    df_sim_train = pd.read_csv(r'C:\Users\liang\Desktop\Agent4Science2025\dataset\full_retrofit_analysis.csv')
    
    # 加载包含真实世界前后对比的测试数据 (Real)
    df_real_test = pd.read_csv(r'C:\Users\liang\Desktop\Agent4Science2025\dataset\retrofit_savings_comparison_with_features_converted.csv')
    
    print(f"✅ 模拟训练集加载成功！共 {len(df_sim_train)} 条记录。")
    print(f"✅ 真实测试集加载成功！共 {len(df_real_test)} 条记录。")

except FileNotFoundError as e:
    print(f"❌ 错误: 找不到文件 {e.filename}。请确保文件路径正确。")
    exit()

# --- 步骤 2: 特征对齐与数据准备 ---
print("\n--- 步骤 2: 正在进行特征对齐与数据准备 ---")

# **第一步：定义列名映射规则**
# 我们将模拟数据中的列名，重命名以匹配真实数据中的标准名称
sim_rename_map = {
    'Climate': 'climate',
    'Type of building': 'btype',
    'Age of construction': 'period',
    'Living area': 'area_m2',
    'H&C device': 'system'
}
df_sim_train.rename(columns=sim_rename_map, inplace=True)

# **第二步：定义我们将使用的共有特征**
common_features = ['climate', 'btype', 'period', 'area_m2', 'system']
print(f"   将使用以下共有特征进行建模: {common_features}")

# **第三步：准备最终的 X_train, y_train (模拟) 和 X_test, y_test (真实)**
if not all(f in df_sim_train.columns for f in common_features):
    missing = [f for f in common_features if f not in df_sim_train.columns]
    print(f"❌ 错误: 模拟数据中缺少必要的共有特征: {missing}。")
    exit()
if not all(f in df_real_test.columns for f in common_features):
    missing = [f for f in common_features if f not in df_real_test.columns]
    print(f"❌ 错误: 真实数据中缺少必要的共有特征: {missing}。")
    exit()

X_train = df_sim_train[common_features]
y_train = df_sim_train['savings_percentage'] # 目标是模拟的改造后能耗

X_test = df_real_test[common_features]
y_test = df_real_test['savings_ratio'] # 目标是真实的改造后能耗

# 清理数据，确保没有空值
y_train.dropna(inplace=True); X_train = X_train.loc[y_train.index]
y_test.dropna(inplace=True); X_test = X_test.loc[y_test.index]

print("✅ 特征对齐完成。")

# --- 步骤 3: 构建预处理与混合模型 ---
print("\n--- 步骤 3: 正在构建预处理与混合模型 ---")
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='passthrough')

# **定义混合模型**
# 基础模型层
estimators = [
    ('ridge', Ridge()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('xgb', xgb.XGBRegressor(random_state=42))
]
# 元模型层 + 构建 Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge() # 使用一个简单的线性模型作为元模型
)

# 构建最终的完整管道
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('stacking_regressor', stacking_regressor)])

# --- 步骤 4: 在模拟数据上训练，在真实数据上测试 ---
print("\n--- 步骤 4: 在模拟数据上训练混合模型，在真实数据上测试 ---")
model_pipeline.fit(X_train, y_train)
print("✅ 混合模型训练完成！")
y_pred = model_pipeline.predict(X_test)

# --- 步骤 5: 评估混合模型的Sim-to-Real泛化性能 ---
print("\n--- 步骤 5: 评估混合模型的Sim-to-Real泛化性能 ---")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📈 混合模型在真实世界数据集上的性能评估:")
print(f"   - R² 分数 (R-squared): {r2:.4f}")
print(f"   - 平均绝对误差 (MAE): {mae:.2f} kWh/m2y")

print("\n--- 分析与解读 ---")
print("这个分数衡量了您的混合模型在真实世界中的泛化能力。")
if r2 < 0:
    print("分析结果: R²分数为负，这再次证实了Sim-to-Real的巨大挑战。即使是更复杂的混合模型，也难以跨越由于数据分布差异（域偏移）造成的鸿沟。这表明问题的核心在于数据本身，而非模型不够复杂。")
else:
    print("分析结果: 混合模型表现出了一定的正向预测能力。您可以将此R²分数与之前单一XGBoost模型的分数进行比较，看是否有提升。通常，在域偏移问题中，混合模型的提升有限。")

print("\n🎉 Sim-to-Real 混合模型分析流程执行完毕！")