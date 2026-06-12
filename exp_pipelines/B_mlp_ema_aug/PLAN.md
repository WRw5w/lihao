# 方案 B：容量 + 正则（attn+MLP LoRA、EMA、RandAugment）

**假设**：现行配方只给 attention 注入 LoRA（1.27M 可训练参数），容量是瓶颈；
扩到 MLP fc1/fc2 后容量约 2.2 倍（2.74M），同时用 EMA 权重平均和 RandAugment
强增强抑制噪声标签过拟合。

**与基准 A 的差异**：`--lora-target attn_mlp --ema-decay 0.999 --randaug`，其余不变。

**命令**：
```bash
python finetune_lora.py --epochs 12 --num-workers 2 \
  --lora-target attn_mlp --ema-decay 0.999 --randaug \
  --work-dir exp_pipelines/B_mlp_ema_aug --cache-dir outputs/cache
```

**判读**：mid 应稳定超过 A 的 12 轮参考值 0.8762；若 low 带明显上升说明
容量被噪声利用，EMA/增强不足以补偿。
