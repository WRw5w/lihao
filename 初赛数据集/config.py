import os
from pathlib import Path

# 设置为数据集所在的根目录
# 默认配置为当前 config.py 文件所在的目录
BASE_DATA_DIR = Path(__file__).parent.resolve()

# 具体的训练集和测试集路径
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"

# 输出文件的默认路径
DEFAULT_OUTPUT_CSV = BASE_DATA_DIR / "pred_results.csv"
DEFAULT_OUTPUT_ZIP = BASE_DATA_DIR / "pred_results.zip"
DEFAULT_WORK_DIR = BASE_DATA_DIR / "outputs"
