import pandas as pd

# 读取两个 CSV
baseline = pd.read_csv("baseline_metadata_and_annual_results.csv")
upgrade = pd.read_csv("upgrade01_metadata_and_annual_results.csv")

# 只保留需要的关键列
baseline_sel = baseline[["bldg_id", "out.site_energy.total.energy_consumption.kwh"]].rename(
    columns={"out.site_energy.total.energy_consumption.kwh": "baseline_kwh"}
)

upgrade_sel = upgrade[[
    "bldg_id",
    "out.site_energy.total.energy_consumption.kwh",
    "out.site_energy.total.energy_consumption.kwh.savings"
]].rename(columns={
    "out.site_energy.total.energy_consumption.kwh": "upgrade_kwh",
    "out.site_energy.total.energy_consumption.kwh.savings": "kwh_sav"
})

# 按 bldg_id 合并
df = pd.merge(baseline_sel, upgrade_sel, on="bldg_id", how="inner")

# 计算百分比节能率
df["pct_sav"] = df["kwh_sav"] / df["baseline_kwh"] * 100

# 输出前几行检查
print(df.head())

# 保存结果
df.to_csv("retrofit_savings.csv", index=False)
print("结果已保存到 retrofit_savings.csv")
