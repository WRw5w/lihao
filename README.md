# 面向噪声标签数据的细粒度图像识别鲁棒微调

全球校园人工智能算法精英大赛 · 算法挑战赛（初赛）。**骨干网络强制为 CLIP ViT-B/32**（OpenAI 官方权重 timm `vit_base_patch32_clip_224.openai`，全程冻结），在含噪标签的细粒度训练数据上做参数高效、噪声鲁棒的微调。

- **线上指标**：干净、类别均衡测试集上的**整体 Top-1 Accuracy**（赛题第六节）。
- **合规红线**：单模型单推理流程（翻转/多尺度 TTA 合规；多模型集成/融合/投票**禁止**）；不宜无约束全参数微调；测试集仅用于推理；去噪须自动化、全流程可复现。

## 当前最优配方

- **结构**：attn+MLP LoRA（rank 32 / alpha 64）注入冻结 CLIP ViT-B/32 @ **448px** + EMA(0.999) + RandAugment。
- **数据**：kNN 一致性去噪保留 **keep-ratio 0.90** + 放宽共识伪标签（thresh 0.6 / margin 0.05）。
- **推理**：多尺度 TTA（448/512/576 + 翻转）+ **balanced 校正**（利用"测试集类别均衡"这一已知事实把预测分布拉回均匀）→ 记为 `tta_balanced`，是目前线上最大提分点。
- **训练轮数**：全量重训取 val 峰值轮（该配方约 **4 轮**）；轮次不是杠杆（见下"教训"）。

> ⚠️ **核心教训**：本地 noisy-val 指标（mid_03_06 / noisy_all）**不可靠预测榜分**——长训练会因"记忆脏验证标签"而虚高，与干净榜分方向相反。**线上榜分是唯一可信信号。** 完整版本演进、每步增益与榜单实测见 **`docs/训练版本演进记录.md`**（活文档）。

## 算法流水线

```
kNN 邻域一致性去噪（按类 top-keep + 高一致保底）
  → 教师头共识伪标签回收（教师预测 ∩ kNN 多数投票，且置信度/margin 双达标）
  → LoRA 微调（严格 90/10 验证：先划分，再做一切标签驱动统计）
  → 多尺度 TTA + balanced 推理 → 提交
```

**严格验证协议**：先分层划分训练/验证集（seed 42），kNN/可靠样本筛选/教师训练/伪标签**全部只用训练分区**，验证样本只查询训练图库（杜绝标签泄漏）；保留样本按 kNN 一致性 × 教师置信度 × 预测 margin **连续加权**。

## 目录结构

```
config.py              # 路径与常量集中配置（换机器/换赛段只改这里）
robustft/              # 核心库：models / data / denoise / engine / features / submission / robust_utils
main.py                # 特征提取（→ outputs/cache）+ 旧置信度基线（保留可跑）
exp_head.py            # 头部级消融 + 最终头部训练（thin entry over robustft）
exp_round2.py          # 第二轮调优（分带验证）
finetune_lora.py       # LoRA 训练/推理主脚本（--full / --predict / --robust-loss gce / --snapshot-after …）
check_submission.py    # 提交文件格式校验
tools/                 # 推理杠杆与工具
  ├─ tta_predict.py        # 多尺度+翻转 TTA（同时产出 balanced 版）
  ├─ balanced_predict.py   # logit 校正到均匀先验（balanced 推理，强度可调）
  ├─ swa_soup.py           # 同架构 checkpoint 权重平均（SWA / model soup，合规单模型）
  └─ collect_pipeline_results.py   # 汇总 exp_pipelines 候选结果
tests/                 # 单测 + A_ce 数值回归门
data/train  data/test  # 数据集（git 忽略）
outputs/               # 特征缓存 + 生产 checkpoint（重资产，*.pt git 忽略）
outputs_full_*/        # 各全量重训模型（…dr_rank32_keep90 = 当前冠军 / …gce / swa_run60 / …）
exp_pipelines/         # 实验：bake-off(B/C/D/E) + 自动优化候选(auto_*) + 60轮诊断(run60)
submissions/           # pred_results*.csv/zip（含 tta_balanced 等上榜候选）
docs/                  # 活文档：训练版本演进记录 / 技术报告 / 赛题正文(含评分标准)
docs/archive/          # 历史文档：重构方案 / EXPERIMENTS / OPTIMIZATION / superpowers(计划+规格)
archive/               # 旧版代码(legacy) + 原始数据zip备份 + 无关文件(misc)
```

## 环境

```bash
# Python 3.11+，CUDA GPU（实测 RTX 5060 Laptop 8GB 足够；系统 RAM 16GB 偏紧）
pip install -r requirements-cuda.txt   # torch/torchvision（CUDA 索引，先装）
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 复现全流程（当前最优配方）

```bash
# 1. 全量特征提取（~1.5h GPU，产生 outputs/cache；只需一次）
python main.py

# 2. 验证候选（90/10 严格协议，定轮数）—— 冠军配方
python finetune_lora.py --epochs 8 --img-size 448 --batch-size 64 \
  --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --num-workers 2 --no-pin --work-dir outputs_full_c448_dr_rank32_keep90

# 3. 全量重训（满数据，--epochs 取上一步 val 峰值轮，约 4）—— 产出 full.pt
python finetune_lora.py --full --epochs 4 --img-size 448 --batch-size 64 \
  --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --num-workers 2 --no-pin --work-dir outputs_full_c448_dr_rank32_keep90

# 4. 推理：多尺度 TTA + balanced → 提交文件
python tools/tta_predict.py --work-dir outputs_full_c448_dr_rank32_keep90 \
  --out-prefix submissions/pred_results_final --scales 448,512,576 --num-workers 2 --no-pin

# 5. 校验（zip 内须恰好是 pred_results.csv）
python check_submission.py --csv submissions/pred_results_final_tta_balanced.csv \
  --zip submissions/pred_results_final_tta_balanced.zip
```

全流程固定随机种子（seed=42），噪声筛选完全自动化、无人工清洗环节（满足复赛/半决赛"可复现"交付要求）。

旧实验入口仍可跑：`python exp_head.py`（头部级消融）、`python exp_round2.py`（第二轮分带调优）。

## 验证与回归

```bash
python tests/test_regression.py             # 单元测试 + A_ce 数值回归（需缓存与 GPU）
python tests/test_regression.py --skip-gpu  # 仅单元测试
python -m unittest discover -s tests -v     # robust_utils 单测 + LoRA 管线契约测试
python finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache  # 训练管线冒烟（不碰正式 checkpoint）
```

> ⚠️ 调试 `main.py` / `finetune_lora.py` 务必用隔离 `--work-dir`（如 `outputs_tmp`），避免覆盖 `outputs/cache`（1.5h GPU 重算）与生产 checkpoint。低 RAM 下 DataLoader 锁页内存会僵死，全程加 `--no-pin`。

## 文档导航

| 文档 | 内容 |
|---|---|
| `docs/训练版本演进记录.md` | **主记录（活）**：版本演进、每步增益、推理杠杆、榜单实测、关键教训 |
| `docs/技术报告_算法使用说明.md` | 技术报告 / 算法使用说明 |
| `docs/面向噪声…-1 (2).md` / `.pdf` | 赛题正文（含评分标准与提交格式） |
| `docs/archive/` | 历史：重构方案、EXPERIMENTS、OPTIMIZATION、superpowers(计划+规格) |
