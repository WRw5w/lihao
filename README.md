# 面向噪声标签数据的细粒度图像识别鲁棒微调

全球校园人工智能算法精英大赛 · 算法挑战赛（初赛）。骨干网络为 **CLIP ViT-B/32**（OpenAI 官方权重，timm `vit_base_patch32_clip_224.openai`）。

## 方法概要

**kNN 邻域一致性去噪（按类保留 75% + 高一致保底）→ 教师头共识伪标签回收 → LoRA 微调**（r=16 注入 attention 层，骨干冻结）。模型选型使用按一致性分带的本地验证协议（中带 0.3–0.6 为主指标）。详见 `docs/技术报告_算法使用说明.md` 与 `docs/EXPERIMENTS.md`。

第二版（严格验证协议，详见 `docs/OPTIMIZATION.md`）：

- **先分层划分训练/验证集**，kNN、可靠样本筛选、教师训练、伪标签全部只用训练分区；验证样本只查询训练图库（旧版在划分前用全量标签筛选，验证指标有标签泄漏，仅作历史参考）。
- 保留样本按 kNN 一致性 × 教师置信度 × 预测 margin **连续加权**；丢弃样本仅当教师预测与 kNN 多数投票一致且置信度/margin 双达标时才回收（共识伪标签）。
- 紧凑整数目标替代 N×C 稠密软标签矩阵（省显存）；梯度裁剪、早停、完整断点续训；`--predict` 必须显式指定 checkpoint（best/last/full）。
- 可调 LoRA 深度（`--lora-blocks 4/6/12`）；默认裁剪范围 0.8–1.0（更适合细粒度）。

## 目录结构

```
config.py            # 路径与常量集中配置（换机器/换阶段只改这里）
robustft/            # 核心库：models / data / denoise / engine / features / submission
main.py              # 特征提取（生成 outputs/cache）+ 旧版置信度基线
exp_head.py          # 头部级消融 + 最终头部训练与提交生成
exp_round2.py        # 第二轮调优（分带验证）
finetune_lora.py     # LoRA 微调训练 / 推理
check_submission.py  # 提交文件格式校验
tests/               # 单元测试 + A_ce 数值回归门
data/train data/test # 数据集（git 忽略）
outputs/             # 特征缓存、checkpoint、实验记录（重资产，git 忽略 *.pt）
submissions/         # pred_results*.csv/zip
docs/                # 赛题文档、技术报告、实验记录、重构方案
tools/  archive/     # 无关脚本 / 历史归档（含旧版代码 archive/legacy）
```

## 环境

```bash
# Python 3.11+，CUDA GPU（实测 RTX 5060 Laptop 8GB 足够）
pip install -r requirements-cuda.txt   # torch/torchvision（CUDA 索引，先装）
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 复现全流程

```bash
python main.py                              # 1. 全量特征提取（~1.5h，产生 outputs/cache）
python exp_head.py                          # 2.（可选）第一轮消融
python exp_round2.py                        # 3.（可选）第二轮调优（含等步数对照组）
python finetune_lora.py --epochs 15         # 4. LoRA 微调（严格 90/10 验证 -> best.pt/last.pt）
python finetune_lora.py --epochs 15 --resume outputs/lora/last.pt   #    （可选）断点续训
python finetune_lora.py --full --epochs 15  # 5. 选定配方后全量重训 -> full.pt
python finetune_lora.py --predict --checkpoint full   # 6. 推理必须显式指定 checkpoint
python check_submission.py --csv submissions/pred_results_lora.csv --zip submissions/pred_results_lora.zip
```

全流程固定随机种子（seed=42），噪声筛选完全自动化，无人工清洗环节。

## 验证与回归

```bash
python tests/test_regression.py             # 单元测试 + A_ce 数值回归（需缓存与 GPU）
python tests/test_regression.py --skip-gpu  # 仅单元测试
python -m unittest discover -s tests -v     # robust_utils 单测 + LoRA 管线契约测试
python finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache  # 训练管线冒烟（不碰正式 checkpoint）
```

注意：测试/调试 `main.py` 或 `finetune_lora.py` 时务必指定隔离的 `--work-dir`（如 `outputs_tmp`），避免覆盖 `outputs/cache`（1.5h GPU 重算）与 `outputs/lora/best.pt`（1h 训练）。

## 本地验证成绩（10% 分层留出，噪声标签）

| 方案 | 噪声全集 | 中带（主指标） |
|---|---|---|
| 置信度两阶段基线 | 0.5840 | — |
| kNN75 + 软标签自训练（头部级） | 0.6450 | 0.8182 |
| LoRA 微调（旧协议，**有标签泄漏**，仅作历史参考） | 0.7353 | 0.8887 |

> 注意：旧版 LoRA 流程在划分验证集之前用全量标签做 kNN 筛选和教师训练，上表 LoRA 行的验证指标偏乐观，不能与新版严格协议的数字直接比较。第二版严格协议成绩见 `outputs/lora/history.json`。
