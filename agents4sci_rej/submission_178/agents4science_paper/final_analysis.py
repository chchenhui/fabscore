import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # 导入Imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. 数据加载 ---
data_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset'
train_file = os.path.join(data_path, 'train_clean.csv')

try:
    df = pd.read_csv(train_file)
    print("✅ 步骤 1: 数据加载成功！")
    print(f"原始数据集包含 {len(df)} 行。")
except FileNotFoundError:
    print(f"❌ 错误: 在路径 '{train_file}' 未找到 'train_clean.csv' 文件。请再次核对路径。")
    exit()

# --- 2. 数据聚合与清洗 ---
print("\n--- 步骤 2: 正在聚合数据以预测每栋建筑的总能耗 ---")

def aggregate_building_data(df):
    leaky_features = [
        'Average demand (kWh/m2y)', 'N. available data on demand', 'N. suitable data on demand',
        'St.Dev on suitable data on demand', 'N. available data on consumption',
        'N. suitable data on consumption', 'St.Dev on suitable data on consumption',
        'Type of energy use'
    ]
    df_cleaned = df.drop(columns=leaky_features, errors='ignore')
    agg_rules = {}
    for col in df_cleaned.columns:
        if col not in ['source_file', 'Average consumption (kWh/m2y)']:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                agg_rules[col] = 'mean'
            else:
                agg_rules[col] = 'first'
    agg_rules['Average consumption (kWh/m2y)'] = 'sum'
    df_agg = df_cleaned.groupby('source_file', as_index=False).agg(agg_rules)
    return df_agg

df_agg = aggregate_building_data(df)
print(f"✅ 数据聚合完成。数据集现在代表 {len(df_agg)} 栋独立建筑。")

# --- 3. 准备建模数据 ---
print("\n--- 步骤 3: 准备特征 (X) 和目标 (y) ---")

df_agg.dropna(subset=['btype'], inplace=True)
print(f"移除关键特征'btype'的空值后，剩余 {len(df_agg)} 栋建筑。")

target_col = 'Average consumption (kWh/m2y)'
features = [col for col in df_agg.columns if col not in [target_col, 'source_file']]
X = df_agg[features]
y = df_agg[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ 数据已拆分为 {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本。")

# --- 4. 构建预处理流程 ---
print("\n--- 步骤 4: 构建特征预处理流程 (已加入缺失值填充) ---")

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# --- FIX: 在Pipeline中为数值和分类特征都加入SimpleImputer ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # 用中位数填充
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # 用'missing'填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 5. 选择并训练模型 ---
print("\n--- 步骤 5: 使用岭回归 (Ridge Regression) 训练模型 ---")

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', Ridge(alpha=1.0))])
model_pipeline.fit(X_train, y_train)
print("✅ 模型训练完成！")

# --- 6. 评估模型性能 ---
print("\n--- 步骤 6: 评估模型在测试集上的真实性能 ---")

y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📈 最终模型性能评估:")
print(f"   - R² 分数 (R-squared): {r2:.4f}")
print(f"   - 平均绝对误差 (MAE): {mae:.2f} kWh/m2y")

if r2 < 0:
    print("\n   分析: R²分数为负，说明模型性能较差。这很可能是由于聚合后的有效数据量过少（仅22个样本），模型无法从中学习到普适的规律。")
elif 0 <= r2 < 0.5:
    print("\n   分析: R²分数较低，模型只解释了一小部分数据变异性，预测能力有限。")
else:
    print("\n   分析: 模型表现合理，能够解释一部分数据变异性。")

print("\n🎉 端到端分析流程执行完毕！")