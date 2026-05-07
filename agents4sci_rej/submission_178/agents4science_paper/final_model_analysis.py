import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. 加载我们精心整理的数据集 ---
data_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\train_final_consolidated_v2.csv'

try:
    df = pd.read_csv(data_path)
    print("✅ 步骤 1: 成功加载整合后的数据集！")
    print(f"   数据集包含 {len(df)} 行, {len(df.columns)} 列。")
except FileNotFoundError:
    print(f"❌ 错误: 未找到 '{data_path}'。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"❌ 错误: 读取文件时出错: {e}")
    exit()

# --- 2. 选择建模数据并定义目标 ---
print("\n--- 步骤 2: 准备建模数据 ---")

# 我们选择'Building stock statistics'作为预测的来源
df_model = df[df['data_source_table'] == 'Building stock statistics'].copy()
print(f"   已筛选出 'Building stock statistics' 数据，共 {len(df_model)} 行用于建模。")

# 定义目标变量
target_col = 'Average consumption (kWh/m2y)'

# 定义需要移除的特征：ID类、源表名、以及所有其他可能泄漏的目标变量
cols_to_drop = [
    'source_file', 'data_source_table', 'Average demand (kWh/m2y)',
    'N. available data on demand', 'N. suitable data on demand', 'St.Dev on suitable data on demand',
    'N. available data on consumption', 'N. suitable data on consumption', 'St.Dev on suitable data on consumption'
]
# 确保目标列也在移除列表中，以防其被用作特征
if target_col not in cols_to_drop:
    cols_to_drop.append(target_col)
    
features = [col for col in df_model.columns if col not in cols_to_drop]
X = df_model[features]
y = df_model[target_col]

# 移除目标值为空的行
y.dropna(inplace=True)
X = X.loc[y.index]

print(f"   预测目标: '{target_col}'")
print(f"   使用的特征数量: {len(X.columns)}")

# 将数据拆分为训练集 (80%) 和测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ 数据已拆分为 {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本。")

# --- 3. 构建预处理与建模的完整流程 ---
print("\n--- 步骤 3: 构建预处理与建模流程 ---")

# 识别不同类型的特征
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 将预处理器和XGBoost模型串联
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', xgb.XGBRegressor(objective='reg:squarederror',
                                                                n_estimators=100,
                                                                learning_rate=0.1,
                                                                max_depth=5,
                                                                random_state=42))])
# --- 4. 训练最终模型 ---
print("\n--- 步骤 4: 训练XGBoost高性能模型 ---")
model_pipeline.fit(X_train, y_train)
print("✅ 模型训练完成！")

# --- 5. 评估并解读最终结果 ---
print("\n--- 步骤 5: 评估模型在测试集上的最终性能 ---")
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📈 最终模型性能评估:")
print(f"   - R² 分数 (R-squared): {r2:.4f}")
print(f"   - 平均绝对误差 (MAE): {mae:.2f} kWh/m2y")

if r2 < 0:
    print("\n   分析: 模型性能较差，建议检查特征或尝试不同的模型参数。")
elif 0 <= r2 < 0.6:
    print("\n   分析: 模型有一定的预测能力，但仍有较大提升空间。")
elif 0.6 <= r2 < 0.8:
    print("\n   分析: 模型表现良好，能够解释大部分数据变异性。")
else:
    print("\n   分析: 模型表现非常出色，具有很高的预测价值！")
    
# --- 6. 分析特征重要性 ---
print("\n--- 步骤 6: 分析特征重要性 ---")
# 提取特征名称
feature_names = numeric_features + \
                model_pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                .named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
# 提取重要性分数
importances = model_pipeline.named_steps['regressor'].feature_importances_
# 创建DataFrame并排序
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

print("对预测最重要的10个特征:")
print(feature_importance_df)

print("\n🎉 恭喜您！已完成从原始数据到最终模型分析的全部流程！")