# 方案 D：数据召回（少丢样本，靠连续权重压噪声）

**假设**：现行 keep 0.75 + 共识伪标签只回收 44 个样本，等于直接扔掉约 23k
张图，其中包含大量"难而正确"的样本（细粒度任务里最有价值的部分）。放宽
保留率与回收阈值、让连续可靠性权重（kNN 一致性 × 教师置信度 × margin）
来调节每个样本的贡献，可能比硬丢弃更优。

**与基准 A 的差异**：方案 B 全部改动 + `--keep-ratio 0.85
--pseudo-thresh 0.6 --pseudo-margin 0.05`。

**命令**：
```bash
python finetune_lora.py --epochs 12 --num-workers 2 \
  --lora-target attn_mlp --ema-decay 0.999 --randaug \
  --keep-ratio 0.85 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --work-dir exp_pipelines/D_recall_more --cache-dir outputs/cache
```

**判读**：与 B 直接对比（唯一差异是数据策略）。若 mid 升且 low 不升，
说明召回有效；若 low 带明显上升，则是把噪声学进去了，维持保守过滤。
