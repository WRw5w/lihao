# 方案 E：浅层适配（只动后 6 个 Transformer 块）

**假设**：噪声标签场景下，全深度适配让模型有过多自由度去拟合错误标签；
只适配最后 6 块（高层语义层），冻结的低层保留 CLIP 预训练的通用视觉
特征，相当于结构性正则。队友 LY 实现了 `--lora-blocks` 正是为此对照。

**与基准 A 的差异**：方案 B 全部改动 + `--lora-blocks 6`
（可训练参数约为 B 的一半，1.5M 左右）。

**命令**：
```bash
python finetune_lora.py --epochs 12 --num-workers 2 \
  --lora-target attn_mlp --ema-decay 0.999 --randaug --lora-blocks 6 \
  --work-dir exp_pipelines/E_shallow6 --cache-dir outputs/cache
```

**判读**：与 B 直接对比（唯一差异是适配深度）。若 E≈B 而 low 带更低，
说明浅适配抗噪更好且省算力；若 E 明显低于 B，说明容量确实是瓶颈，
反向支持 B/C 路线。
