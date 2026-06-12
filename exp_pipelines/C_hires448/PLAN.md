# 方案 C：高分辨率 448（细粒度最大杠杆）

**假设**：ViT-B/32 在 224 输入下只有 7×7=49 个 patch token，空间分辨率对
细粒度识别（区分相近子类的局部纹理差异）严重不足。448 输入 → 14×14=196
token，等效"视力"×4。位置编码由 timm 对官方权重做双三次插值，骨干权重
不变，合规。

**与基准 A 的差异**：方案 B 全部改动 + `--img-size 448`；单轮耗时约 3 倍，
故只训 8 轮；batch 96（显存不够自动降 64）。

**命令**：
```bash
python finetune_lora.py --epochs 8 --num-workers 2 --batch-size 96 \
  --lora-target attn_mlp --ema-decay 0.999 --randaug --img-size 448 \
  --work-dir exp_pipelines/C_hires448 --cache-dir outputs/cache
```

**判读**：若 8 轮内 mid 就追平/超过 A 的 12 轮参考值 0.8762，说明分辨率
是主要瓶颈，胜出后全量阶段可用"224 全量训练 → 448 短调"两段式省时。
注意：kNN 筛选/教师仍基于 224 特征缓存（只影响样本选择，不影响合规）。
