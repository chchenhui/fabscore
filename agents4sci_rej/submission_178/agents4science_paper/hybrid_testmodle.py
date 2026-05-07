# -*- coding: utf-8 -*-
"""
Hybrid Retrofits Model: Physics Proxy (UA×HDD + Vent Loss) + Residual Learning (XGBoost)
Author: Chris & Assistant
Python: 3.13+
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# -----------------------------
# 0) 数据路径与读取
# -----------------------------
DATA_PATH = r"C:\Users\liang\Desktop\Agent4Science2025\dataset\consolidated_retrofit_data.csv"
SAMPLE_FRAC = 0.20
RANDOM_SEED = 42

print("=== Hybrid 最小可运行脚本 ===")
print(f"[加载数据] {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"❌ 找不到数据集：{DATA_PATH}")

df_full = pd.read_csv(DATA_PATH)
df = df_full.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"✅ 数据加载并抽样完成：{len(df)} 行\n")

# -----------------------------
# 1) 工具函数
# -----------------------------
def coerce_numeric_series(s, default=0.0, length=None):
    """将含%/单位/逗号等的字符串稳健转为 float（取首个数字）。"""
    if s is None:
        if length is None:
            length = len(df)
        return pd.Series([default]*length, dtype=float)
    ss = s.astype(str).str.strip().str.lower()
    is_percent = ss.str.endswith('%')
    ss_clean = ss.str.replace('%', '', regex=False)
    ss_clean = ss_clean.apply(lambda x: re.sub(r'[^0-9\-,\.eE+]', ' ', x)).str.strip()
    ss_clean = ss_clean.str.replace(',', '.', regex=False)
    def first_number(x):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x)
        return m.group(0) if m else np.nan
    nums = ss_clean.apply(first_number)
    vals = pd.to_numeric(nums, errors='coerce')
    vals = np.where(is_percent, vals/100.0, vals)
    vals = pd.Series(vals, dtype=float).fillna(default)
    return vals

def coerce_efficiency_series(s, default_if_yes=0.6, default_if_no=0.0, fallback_default=0.0):
    """'YES/NO/True/False/百分比/数值' → [0,1] 效率。"""
    length = len(df)
    if s is None:
        return pd.Series([fallback_default]*length, dtype=float)
    ss = s.astype(str).str.strip().str.lower()
    yes_tokens = {'yes','y','true','t','on','present'}
    no_tokens  = {'no','n','false','f','off','absent'}
    is_yes = ss.isin(yes_tokens)
    is_no  = ss.isin(no_tokens)
    numeric_part = coerce_numeric_series(s, default=np.nan)
    eff = pd.Series(np.where(is_yes, default_if_yes,
                     np.where(is_no, default_if_no, numeric_part)), dtype=float)
    eff = eff.fillna(fallback_default).clip(0.0, 1.0)
    return eff

def get(col, default=0.0):
    """安全取列；不存在则返回默认值序列。"""
    if col in df.columns:
        return df[col]
    return pd.Series([default]*len(df))

# -----------------------------
# 2) y_phys_proxy 计算（UA×HDD + 通风热损）→ kWh/m²·y
# -----------------------------
print("[Step] 构建物理基线 y_phys_proxy ...")

# 2.1 常量与映射
CLIMATE_HDD = {
    'Nordic': 5000,
    'Central': 3000,
    'Continental': 3500,
    'Oceanic': 2200,
    'Mediterranean': 1500,
    'Southern dry': 800,
    'Alpine': 4000,
}
DEFAULT_HDD = 2500.0

EPS_THK = 1.0   # 防除0
K_FACADE = 0.8
K_ROOF   = 1.0
K_GROUND = 0.6

GLAZING_U = {'single': 5.0, 'double': 2.8, 'triple': 1.5}
DEFAULT_WINDOW_U = 3.0

RHO_AIR = 1.2      # kg/m³
CP_AIR  = 0.2778   # Wh/(kg·K)
HOURS_PER_DAY = 24
SYSTEM_EFF = 1.0   # 如需考虑设备效率，可调整

# 2.2 取原始列（稳健数值化）
facade_area  = coerce_numeric_series(get('Façade area', 0.0)).clip(lower=0)
roof_area    = coerce_numeric_series(get('Roof area', 0.0)).clip(lower=0)
ground_area  = coerce_numeric_series(get('Ground/Cellar area', 0.0)).clip(lower=0)
window_area  = coerce_numeric_series(get('Windows area', 0.0)).clip(lower=0)
living_area  = coerce_numeric_series(get('Living area', 1.0)).replace(0, 1.0)
volume       = coerce_numeric_series(get('Building inner volume', 0.0)).clip(lower=0)

t_f = coerce_numeric_series(get('Façade insulation thickness', 0.0)).clip(lower=0)
t_r = coerce_numeric_series(get('roof insulation thickness', 0.0)).clip(lower=0)
t_g = coerce_numeric_series(get('Ground/Cellar insulation thickness', 0.0)).clip(lower=0)

glazing_type = get('Windows galzing type', "").astype(str).str.lower().str.strip()

climate_series = get('Climate', 'Unknown').astype(str).str.strip()
HDD = climate_series.map(CLIMATE_HDD).fillna(DEFAULT_HDD)

ACH = coerce_numeric_series(get('AIR CHANGE RATE', 0.5), default=0.5).clip(lower=0)
MECH_ETA = coerce_efficiency_series(get('MECH VENT EFFICIENCY', 0.0),
                                    default_if_yes=0.6, default_if_no=0.0, fallback_default=0.0)

# 2.3 UA_proxy
UA_facade = K_FACADE * facade_area / (t_f + EPS_THK)
UA_roof   = K_ROOF   * roof_area   / (t_r + EPS_THK)
UA_ground = K_GROUND * ground_area / (t_g + EPS_THK)

U_win = glazing_type.map(GLAZING_U).fillna(DEFAULT_WINDOW_U)
UA_window = U_win * window_area

UA_proxy = UA_facade + UA_roof + UA_ground + UA_window   # 量级近似 ~ W/K

# 2.4 年传导 & 年通风热损
cond_Wh  = UA_proxy * (HDD * HOURS_PER_DAY)
cond_kWh = cond_Wh / 1000.0

vent_WattHour_per_K = ACH * volume * RHO_AIR * CP_AIR
vent_loss = vent_WattHour_per_K * (HDD * HOURS_PER_DAY) * (1 - MECH_ETA)
vent_kWh  = vent_loss / 1000.0

# 2.5 转为 kWh/m²·y
y_phys_proxy = (cond_kWh + vent_kWh) / living_area * SYSTEM_EFF
y_phys_proxy = pd.Series(y_phys_proxy).replace([np.inf, -np.inf], np.nan)
y_phys_proxy = y_phys_proxy.fillna(y_phys_proxy.median())
df['y_phys_proxy'] = y_phys_proxy

print("✅ y_phys_proxy 构建完成\n")

# -----------------------------
# 3) 残差目标与特征准备
# -----------------------------
target_col = 'consumption_after'
if target_col not in df.columns:
    raise SystemExit(f"❌ 目标列 {target_col} 不在数据中！")

y_true = df[target_col].astype(float)
y_resid = y_true - y_phys_proxy

# 与你一致的“白名单”特征
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
use_cols = [c for c in allowlist_features if c in df.columns]
X = df[use_cols].copy()

# 将类别列转为字符串，避免 OneHot 报错
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]
for col in categorical_features:
    X[col] = X[col].astype(str)

print(f"[特征] 使用列数：{len(X.columns)}，数值列 {len(numeric_features)}，类别列 {len(categorical_features)}")
print(f"[目标] 使用残差 y_resid（consumption_after - y_phys_proxy）\n")

# -----------------------------
# 4) 训练/测试划分
# -----------------------------
X_train, X_test, y_resid_train, y_resid_test, y_phys_train, y_phys_test, y_true_train, y_true_test = train_test_split(
    X, y_resid, y_phys_proxy, y_true, test_size=0.2, random_state=RANDOM_SEED
)
print(f"[拆分] 训练集 {len(X_train)} ，测试集 {len(X_test)}\n")

# -----------------------------
# 5) 预处理 + 残差模型
# -----------------------------
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
    ],
    remainder='drop'
)

resid_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_SEED,
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', resid_model)
])

print("[训练] Stage-2 残差模型 ...")
model_pipeline.fit(X_train, y_resid_train)
print("✅ 残差模型训练完成\n")

# -----------------------------
# 6) 组合预测与评估
# -----------------------------
resid_hat = model_pipeline.predict(X_test)
y_hat = y_phys_test.values + resid_hat

r2 = r2_score(y_true_test, y_hat)
mae = mean_absolute_error(y_true_test, y_hat)

print("=== 最终混合模型（物理 + 残差）评估 ===")
print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.3f} kWh/m²·y\n")

# -----------------------------
# 7) 特征重要性（前15）
# -----------------------------
try:
    cat_feature_names = model_pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat'] \
        .named_steps['onehot'] \
        .get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    fi = (pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
          .sort_values('Importance', ascending=False)
          .head(15))
    print("前15个残差模型特征重要性：")
    print(fi.to_string(index=False))
except Exception as e:
    print(f"⚠️ 无法提取特征重要性：{e}")

print("\n🎯 使用提示：")
print("1) y_phys_proxy 为工程量级近似；若有真实 HDD/更准确 U 值或RC模型，可直接替换常数提升可信度。")
print("2) 若气候标签不同，请在 CLIMATE_HDD 中增补映射。")
print("3) 建议后续增加 Sim→Real 测试（真实监测数据集做外部测试），检验跨域泛化。")
