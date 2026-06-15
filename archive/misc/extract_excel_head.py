import pandas as pd
import json

df = pd.read_excel('D:/03_Downloads/2026年本科生学科竞赛立项项目名单.xlsx', header=None)
data = df.head(10).to_dict(orient='records')
with open('D:/02_Projects/ML/jinyinsai/初赛数据集/excel_head.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
