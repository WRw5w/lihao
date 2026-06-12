# 初赛 baseline

这个方案用的是 `timm/vit_base_patch32_clip_224.openai`，对应赛题要求的 CLIP ViT-B/32 OpenAI 预训练权重。

流程很直接：

1. 从 `train/` 和 `test/` 读取图片。
2. 用冻结的 CLIP 图像骨干抽取特征。
3. 先训练一个带噪声容忍的 cosine 分类头。
4. 按置信度筛掉低可信样本，再二阶段重训。
5. 生成 `pred_results.csv`，并打包成 `pred_results.zip`。

## 运行

```bash
python main.py
```

默认会在当前目录下生成：

- `pred_results.csv`
- `pred_results.zip`
- `outputs/`

## 调参

常用参数：

- `--stage1-epochs`
- `--stage2-epochs`
- `--keep-ratio`
- `--head-batch-size`
- `--extract-batch-size`

调试时可以用：

```bash
python main.py --max-train-samples 2000 --max-test-samples 2000
```

