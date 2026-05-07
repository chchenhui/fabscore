import pandas as pd
import re

# === 改成你的 Target 表路径与工作表名 ===
XLSX_PATH = r"C:\Users\liang\Desktop\Agent4Science2025\dataset\iNSPiRe FP7 Retrofit Database\SFH_I_AWHP_CEI.xlsx"
SHEET_NAME = "Target buildings simulations"  # 或者实际 sheet 名称

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, header=0)

# 1) 锁定“Total Heating + Cooling + DHW + Ventilation performance”组内的 FEE/FET
block_key = "Total Heating + Cooling + DHW + Ventilation performance"
fee_key = "Final Energy electric demand - FEE"
fet_key = "Final Energy thermal demand - FET"

def find_col(key1, key2):
    pat1 = re.compile(re.escape(key1), re.I)
    pat2 = re.compile(re.escape(key2), re.I)
    for c in df.columns:
        if pat1.search(str(c)) and pat2.search(str(c)):
            return c
    return None

col_fee = find_col(block_key, fee_key)
col_fet = find_col(block_key, fet_key)

if not col_fee or not col_fet:
    raise RuntimeError("找不到 FEE/FET 列，请检查 sheet 名或表头文本是否一致。")

# 2) 取 Living area 并转 m²（若单位是 sqft）
area_col_candidates = ["Living area", "living area", "LIVING AREA"]
col_area = next((c for c in area_col_candidates if c in df.columns), None)
if not col_area:
    raise RuntimeError("找不到 Living area 列，请核对列名。")

area = pd.to_numeric(df[col_area], errors="coerce")
# 简单判断是否为 ft²：若最大值远大于几千（例如 50,000），很可能是 ft²
area_m2 = area * 0.092903 if area.max(skipna=True) and area.max() > 5000 else area

# 3) 计算改造后总能耗与强度
fee = pd.to_numeric(df[col_fee], errors="coerce")
fet = pd.to_numeric(df[col_fet], errors="coerce")
df["_retrofit_total_final_kwh_y"] = fee.fillna(0) + fet.fillna(0)
df["_retrofit_eui_kwh_m2y"] = df["_retrofit_total_final_kwh_y"] / area_m2

# 4) 展示前几行结果
cols_show = [col_area, col_fee, col_fet, "_retrofit_total_final_kwh_y", "_retrofit_eui_kwh_m2y"]
print(df[cols_show].head(10))
