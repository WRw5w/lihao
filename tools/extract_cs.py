import pandas as pd

file_path = 'D:/03_Downloads/2026年本科生学科竞赛立项项目名单.xlsx'
df = pd.read_excel(file_path, header=1) # The real header is at row index 1

# Filter for rows where either college or competition name contains "计算机"
mask1 = df['学院(中心）'].astype(str).str.contains('计算机', na=False)
mask2 = df['竞赛项目名称'].astype(str).str.contains('计算机', na=False)

filtered_df = df[mask1 | mask2].copy()

# Fill NaN in specific columns if needed or just convert to string
filtered_df = filtered_df.fillna('')

# Convert to markdown
md_content = "# 2026年计算机相关竞赛名单\n\n"
md_content += filtered_df.to_markdown(index=False)

output_path = 'D:/02_Projects/ML/jinyinsai/初赛数据集/计算机竞赛.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Extracted {len(filtered_df)} competitions to {output_path}")
