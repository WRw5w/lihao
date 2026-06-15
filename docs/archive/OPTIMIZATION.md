# 综合优化说明

## 赛题合规边界

- 仅使用 OpenAI 官方 CLIP ViT-B/32 预训练权重。
- 每阶段仅使用该阶段官方训练数据。
- 测试集只用于最终推理。
- 最终提交使用单个 LoRA 模型和单一推理流程，不进行模型集成。

## 严格验证协议

旧版 LoRA 流程在划分验证集之前使用全量标签完成 kNN 筛选和教师头训练，
因此旧版 LoRA 验证指标存在标签泄漏，只能作为历史参考。

新版流程：

1. 首先按类别分层划分训练集和验证集。
2. kNN、可靠样本筛选、教师训练和伪标签生成仅使用训练分区。
3. 验证样本只能查询训练图库，不参与教师训练或模型初始化。
4. 使用中一致性分带作为主要 checkpoint 选择指标。

## 鲁棒训练升级

- 保留每类 kNN 一致性靠前的样本，并保留高一致性样本。
- 对保留样本按 kNN 一致性、教师原标签置信度和预测 margin 连续加权。
- 仅回收教师预测与 kNN 多数预测一致，且置信度和 margin 均达标的样本。
- LoRA 可选择最后 4、6 或 12 个 Transformer block。
- 默认裁剪范围由 `0.65-1.0` 调整为更适合细粒度识别的 `0.8-1.0`。
- 使用紧凑整数目标替代 `N×C` 稠密软标签矩阵，大幅降低复赛长尾场景显存占用。
- 加入梯度裁剪、早停、完整断点续训状态和明确的 checkpoint 策略。

## 公平消融

`exp_round2.py` 新增等训练步数控制组：

- `knnXX_teacher`：筛选子集训练一次。
- `knnXX_continue_control`：不加伪标签，继续训练同样 epoch。
- `knnXX_soft_r1`：加入伪标签，继续训练同样 epoch。

只有 `continue_control` 与 `soft_r1` 的差异才可归因于伪标签。

## 推荐运行顺序

```bash
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main.py
python finetune_lora.py --epochs 15
python finetune_lora.py --full --epochs 15
python finetune_lora.py --predict --checkpoint full --output-csv pred_results.csv
python check_submission.py
```
