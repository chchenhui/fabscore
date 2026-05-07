# pick_top20.py
import pandas as pd
import os

# ---------- 1. 读取人类数据 ----------
human_file = '前测-人类回答_正确率.xlsx'
if not os.path.exists(human_file):
    raise FileNotFoundError(human_file)

# 重命名列，只保留图片名和正确率
human_df = pd.read_excel(human_file)
human_df = human_df.rename(columns={
    human_df.columns[0]: 'picture',
    human_df.columns[-1]: 'human_acc'
})[['picture', 'human_acc']]
human_df['picture'] = human_df['picture'].astype(str).str.strip()

# ---------- 2. 读取 AI 数据 ----------
ai_file = 'result.csv'
if not os.path.exists(ai_file):
    raise FileNotFoundError(ai_file)

ai_df = pd.read_csv(ai_file)[['pictures', 'correct_ratio']]
ai_df = ai_df.rename(columns={
    'pictures': 'picture',
    'correct_ratio': 'ai_acc'
})
ai_df['picture'] = ai_df['picture'].astype(str).str.strip()

# ---------- 3. 根据图片名匹配 ----------
merged = pd.merge(human_df, ai_df, on='picture', how='inner')
if merged.empty:
    raise ValueError('两张表没有匹配到任何同名图片，请检查图片名是否一致！')

# ---------- 4. 计算差异并排序 ----------
merged['diff'] = merged['human_acc'] - merged['ai_acc']
top20 = merged.sort_values('diff', ascending=False).head(20)

# ---------- 5. 输出 ----------
top20.to_excel('top20_diff.xlsx', index=False)
top20.to_csv('top20_diff.csv', index=False, encoding='utf-8-sig')

print('已生成 top20_diff.xlsx 与 top20_diff.csv')
print(top20)