# 实验记录

骨干：CLIP ViT-B/32（timm `vit_base_patch32_clip_224.openai`，OpenAI 官方权重）
特征：冻结骨干 + 水平翻转 TTA 平均，L2 归一化，768 维（pre-projection），缓存于 `outputs/cache/`
验证协议：按类分层留出 10%（10285 样本，标签含噪），seed=42

## 噪声扫描（kNN 邻域一致性，k=16，余弦相似度加权）

| 一致性阈值 | 低于阈值的训练样本占比 |
|---|---|
| < 0.1 | 21.4% |
| < 0.3 | 42.0% |
| < 0.5 | 59.1% |

结论：噪声水平较高；但低一致性 ≠ 必然错标（细粒度难样本也会低），激进过滤有害（见第二轮）。

## 验证指标说明

- **noisy-val**：留出集全量准确率（标签含噪，会奖励拟合噪声，偏差大）
- **分带指标**：按验证样本自身的 kNN 一致性分带
  - low（<0.3，4308 样本）：标签多为错标，此带准确率高≈拟合噪声，不应追求
  - **mid（0.3–0.6，2673 样本）：标签基本正确但样本难，区分度最高，主要决策依据**
  - high（≥0.6，3304 样本）：接近饱和（98%+）

## 第一轮消融（epochs=15）

| 方法 | noisy-val | 伪干净-val |
|---|---|---|
| A 纯 CE+LS（全量） | 0.6264 | 0.9870 |
| B 两阶段置信度选样（原基线，keep 0.8） | 0.5840 | 0.9749 |
| C kNN 一致性过滤（keep 0.75） | 0.5881 | 0.9809 |
| D GMM 损失建模加权 | 0.5934 | 0.9791 |
| **E kNN 过滤 + 伪标签自训练** | **0.6279** | **0.9900** |
| F = E + mixup + EMA | 0.5910 | 0.9827 |

- E 双指标第一；F 中 EMA（decay 0.999）在 ~180 步内未收敛，弃用。

## 第二轮调优（epochs=20，分带指标）

| 配置 | noisy | low | mid | high |
|---|---|---|---|---|
| ref_ce | 0.6409 | 0.2711 | 0.8058 | 0.9897 |
| knn50_soft_r1 | 0.5928 | 0.2006 | 0.7415 | 0.9840 |
| knn60_soft_r1 | 0.6144 | 0.2268 | 0.7759 | 0.9891 |
| **knn75_soft_r1** | **0.6450** | 0.2725 | **0.8182** | **0.9906** |
| fused60_soft_r3 | 0.6440 | 0.2830 | 0.8032 | 0.9861 |

- **最终方案：kNN 一致性按类保留 75% + 一致性≥0.7 保底 + 一轮软标签自训练（伪标签阈值 0.7）**
- keep 0.5/0.6 删掉过多"难而正确"样本，全面劣化
- fused60 多轮自训练 low 带升高（0.283），有拟合噪声迹象，不取

## 最终提交（头部级，2026-06-12）

- 全量 103218 样本，kNN 过滤保留 77494（75.08%），自训练阶段使用 77576（伪标签 82）
- 产出 `pred_results.csv` / `pred_results.zip`（24967 条预测），格式校验 PASS
- 模型存于 `outputs/artifacts/final_head_knn_soft.pt`

## LoRA 骨干微调（`finetune_lora.py`，2026-06-12）

- LoRA r=16, α=32 注入全部 attention qkv/proj（1.27M 可训练参数，骨干冻结）；头部用冻结特征教师热启动
- 训练数据沿用同一 kNN 选样 + 软标签配方（69844 训练图像）；OneCycle，AMP，batch 192，10 epochs（~5 min/epoch，RTX 5060 Laptop）
- 验证（10285 留出样本，与头部实验同一划分）：

| epoch | noisy | low | mid | high |
|---|---|---|---|---|
| 1 | 0.6718 | 0.3304 | 0.8152 | 0.9816 |
| 5 | 0.7242 | 0.4131 | 0.8807 | 0.9866 |
| **10（best）** | **0.7353** | 0.4311 | **0.8887** | **0.9918** |

- 对比头部级最优（knn75_soft：noisy 0.6450 / mid 0.8182）：**全面大幅领先**（噪声全集 +9.0pt，中带 +7.1pt）
- 第 10 epoch 仍是 best，曲线接近收敛但尚未过拟合迹象；后续可尝试更多 epoch / 全量数据重训
- checkpoint：`outputs/lora/best.pt`、训练曲线 `outputs/lora/history.json`

## 复现命令

```bash
python main.py                          # 特征提取 + 原基线（产生特征缓存）
python exp_head.py                      # 第一轮消融
python exp_round2.py                    # 第二轮调优
python exp_head.py --final --method knn_soft --epochs 20 --keep-ratio 0.75   # 最终头部 + 提交
python check_submission.py              # 提交格式校验
python finetune_lora.py                 # LoRA 微调（90/10 验证）
python finetune_lora.py --predict       # 用 best.pt 生成 pred_results_lora.csv
```
