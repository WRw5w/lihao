# 在 SLURM 集群上跑（给帮忙的朋友看）

这个文件夹是**完全自包含**的:代码快照 + 224 特征缓存 + **数据集(已在 `./data/`)** + 脚本。
数据已打包进来,`env.sh` 默认 `DATA_DIR=./data`,所以数据**零配置**。只需改集群专属配置。

## 你需要改的 2 处(都在 slurm 脚本里)

1. **环境**:取消注释并改成你们集群拿到 `python+torch+timm+CUDA` 的方式
   (`module load ...` / `conda activate ...` / 或 `export PY=/venv/bin/python`)。依赖见 `requirements.txt`。
2. **SBATCH 资源**:`--partition`(必改)、必要时 `--account`/`--qos`、`--time`、`--mem`。

> 数据已内置,无需设 `DATA_DIR`。若你要把数据挪到别处,在 slurm 脚本里取消注释 `export DATA_DIR=...` 即可。
> 数据须与原始赛题**字节一致**(否则训练会主动断言报错;那时删掉 `cache/` 让它用内置数据重建)。

## 跑的步骤

```bash
# 0) 登录节点(有外网):先把 CLIP 权重下载进包内缓存
bash 00_prefetch_weights.sh

# 1) 选一种提交方式:
# (A) 简单:4 个核心臂在 1 块 GPU 上串行
sbatch submit.slurm

# (B) 高效:作业数组,每个臂各占 1 块 GPU 并行
sbatch --array=0-3 submit_array.slurm        # 核心 4 臂
# sbatch --array=0-5 submit_array.slurm       # 全部 6 臂
# sbatch --array=0-3%2 submit_array.slurm     # 最多同时占 2 块卡

# 2) 看状态 / 日志
squeue -u $USER
tail -f runs/slurm-*.out
```

## 产物

```
runs/<arm>/lora/ep01..ep12.pt + full.pt     每臂的 checkpoint(full.pt=SWA后)
runs/submissions/pred_<arm>_tta_balanced.zip  ← 只交这个(均衡校正版)
```
跑完把 `runs/submissions/*_tta_balanced.zip` 回传即可。

## 资源参考(单臂)

- 训练 = 全量 ~10 万图 × 12 epoch + 同轨迹 SWA + 多尺度 TTA。
- 显存:核心臂 batch 已按 ~8GB 设,卡大就在 `arms.tsv` 调高 `bs`(416 比 448 省显存)。
  `r608_aligned`(19×19,361 token)最重,建议 ≥16GB。
- 实验逻辑、对照臂含义、判据见同目录 `README.md`。
