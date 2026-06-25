# 位置编码无损重采样 · 对照实验包（remote-ready）

CLIP ViT-B/32 噪声标签微调项目的一组**对照实验**,验证一个新训练方案:
**13×13 网格 + `align_corners=True` 无损位置编码**。整包自包含,拷到远程 GPU 服务器即可跑。

---

## 1. 假设(为什么这么做)

一张图里有**两层独立的插值**:

| 层 | 是什么 | 现状 |
|---|---|---|
| ① 图像插值 | 测试图(短边中位 375px)缩放到模型输入尺寸 | 448 比 416 放大更狠,416 更温和 |
| ② **位置编码插值** | CLIP 预训练的 **7×7=49** 个位置锚点被插值到新网格 | 项目一直忽略,448→14×14 用 timm 默认 bicubic+antialias 把锚点**糊掉** |

**新方案攻第②层**:因为 **13 = 2×7−1**,把 7×7 用 `bilinear + align_corners=True` 升到 **13×13**(`img_size=416`)时,原始 49 个锚点会**精确落在** 13×13 的偶数节点上,中间节点是相邻锚点的纯线性均值 —— **预训练位置编码零损失保留**。同理 **19×19@608**(=3×7−1)也无损。

已在本机实测(`code/robustft/models.py` 的 `_aligned_resample_abs_pos_embed`):

| 416 重采样方式 | 偶数节点 vs 原始 49 锚点 最大误差 |
|---|---|
| timm 默认(bicubic+antialias,align_corners=False) | **0.0573**（锚点被糊） |
| **aligned**(bilinear + align_corners=True) | **0.0000**（无损） |

416 同时把两层插值都做到最温和:图像层 416<448,位置编码层无损。这是项目反复撞分辨率墙时**唯一没试过、机制上站得住**的方向。

---

## 2. 对照臂（baseline 矩阵)

固定**制胜配方**(cleanlab 去噪 + mixup0.2 + EMA + RandAug + rank32 attn_mlp,全量 12ep + 同轨迹 SWA ep04–12),**只动 `img_size` × 重采样方式**这一个轴。定义见 [`arms.tsv`](arms.tsv)。

| arm | img | 网格 | 重采样 | 作用 | tier |
|---|---|---|---|---|---|
| `b448_default` | 448 | 14×14 | timm 默认 | **服务器上重建现冠军**(对照锚点;历史 clmixsoup3≈78.6) | core |
| `r416_default` | 416 | 13×13 | timm 默认 | 控制"纯换分辨率"的影响 | core |
| **`r416_aligned`** ⭐ | 416 | 13×13 | **aligned** | **新方案(锚点无损)** | core |
| `b224_native` | 224 | 7×7 | 无(原生) | 零位置编码插值 + 最小图像放大的极端对照 | core |
| `b448_aligned` | 448 | 14×14 | aligned | 解耦"align_corners 方法"与"2×−1 干净网格"(14 是偶数,锚点不精确保留) | stretch |
| `r608_aligned` | 608 | 19×19 | aligned | 更多 token + 仍无损锚点,测"位置编码无损后多分辨率是否还有用" | stretch |

**三方读法**
- `r416_aligned − r416_default` = align_corners 无损位置编码的**纯效应**(同分辨率,只差插值)。
- `r416_aligned − b448_default` = "要不要从 448 切到 13×13 无损"的**实战结论**。
- `r416_default ≈ b448_default` 而 `r416_aligned` 明显更高 → 坐实是**位置编码**这一层在起作用,不是分辨率。
- `b448_aligned` vs `r416_aligned`:若后者远高,说明赢在 **2×−1 干净网格**,而不只是换了插值方法。

> ⚠️ **唯一真相是榜分**(本项目 val 在 0.92–0.93 段已多次证伪)。这些臂各产出一个 `*_tta_balanced.zip`,**交上去比榜分**。

---

## 3. 前置条件

1. **数据**:本(发往服务器的)副本已把数据**打包在 `./data/`**(`train/<类别编号>/*.jpg` 共 500 类 + `test/*.jpg` 平铺),`env.sh` 默认 `DATA_DIR=./data`,**零配置**。
   > 若从代码仓库取用而非本副本,则需自备数据放到 `$DATA_DIR/{train,test}`。数据须与原赛题**字节一致**(否则 `finetune_lora.py` 会断言 `ImageFolder order mismatch with feature cache` 报错——这是保护,不是 bug)。
2. **Python 环境**:torch + torchvision(按服务器 CUDA 选 wheel)+ `pip install -r requirements.txt`。见 [`requirements.txt`](requirements.txt)。
3. **GPU**:core 四臂 batch 已按 ~8GB 设(416 比 448 更省显存);显存大就调高 `bs`(改 `arms.tsv`)。`r608_aligned` 较重,建议 ≥16GB。
4. **CLIP 权重**:首次跑 timm 会从 HuggingFace 下载 `vit_base_patch32_clip_224.openai`(需联网)。已离线缓存则在 `env.sh` 取消注释 `HF_HUB_OFFLINE=1`。
5. **特征缓存**:已**打包在 `./cache/`**(224 冻结特征,分辨率无关,所有臂共用)。数据一致就直接用;缺失则 `01_build_cache.sh` 自动重建。

---

## 4. 怎么跑

```bash
# 0) 改 env.sh 顶部:PY / DATA_DIR(/OUT_ROOT 可选)
nano env.sh

# 1) 冒烟自检(几十秒):下载/加载 CLIP,六臂建模+前向,打印锚点保真度
bash 00_smoke.sh
#    期望:r416_aligned / r608_aligned / b224_native 显示 anchor_err=0.0000,全部 finite=True

# 2) 跑核心四臂(串行;每臂 = 训练12ep -> SWA -> TTA+balanced)
bash run_all.sh core
#    或单臂:bash run_arm.sh r416_aligned 416 aligned 32 416,448,480
#    或全部六臂:bash run_all.sh all

# 3) 交榜:只交 *_tta_balanced.zip
ls runs/submissions/*_tta_balanced.zip
```

后台长跑:`nohup bash run_all.sh core > runs/all.log 2>&1 &`,再 `tail -f runs/all.log`。

产物在 `OUT_ROOT`(默认 `./runs/`):
```
runs/<arm>/lora/ep01.pt..ep12.pt + full.pt(SWA后)   runs/<arm>/train.log
runs/submissions/pred_<arm>_tta.csv/.zip             # 无均衡(别交)
runs/submissions/pred_<arm>_tta_balanced.csv/.zip    # ✅ 交这个
```

---

## 5. 结果判据

- 现冠军级 baseline 是 `b448_default`(同配方 + SWA,历史 clmixsoup3 ≈ **78.6**)。
- **`r416_aligned` ≥ b448_default** → 新方案有效,可继续(加宽 SWA / 微调 TTA scales)。
- **`r416_aligned ≈ r416_default ≈ b448_default`** → 位置编码插值不是瓶颈,**归档为有价值的负结果**(补上 416 当年留的空白)。
- 预期:本项目反复结论是瓶颈 = 冻结 B/32 + 不可约噪声,分辨率非主杠杆,**故预期增益偏小(±0.x pt)**。但这是机制上无损、成本低于现 baseline 的干净探针,值得一跑。

---

## 6. 合规 & 铁律(沿用主项目)

- ✅ 骨干仍是赛规强制的 CLIP ViT-B/32,**预训练权重不变**(只重采样位置编码,合规)。
- ✅ 单模型单推理;**SWA/soup 只在同一次训练的相邻 epoch 间做**,绝不跨 seed/跨臂(跨盆地平均会暴跌)。
- ✅ TTA(多尺度+翻转)是单模型测试期增强,不是多模型集成。
- ✅ **只交 `*_tta_balanced.zip`**(均衡校正是 +2.9pt 的免费杠杆)。
- ✅ **只认榜分**。

---

## 7. 改了什么代码(相对主仓库)

`code/` 是运行期快照。相对主仓库只加了一个开关 `--pos-resample {timm,aligned}`(默认 `timm`,对历史跑零影响):
- `code/robustft/models.py`:新增 `_aligned_resample_abs_pos_embed` + `patched_pos_resample` 上下文管理器;`build_lora_model(..., pos_resample=...)` 在 `timm.create_model` 外包一层临时替换 timm 的 `resample_abs_pos_embed`。
- `code/finetune_lora.py`:新增 `--pos-resample` 参数并透传;经 `vars(args)` 存进 checkpoint。
- `code/tools/tta_predict.py`:从 ckpt 读回 `pos_resample`。

> 关键:骨干冻结,`pos_embed` 不参与训练,但 `state_dict()` 会把它存进 checkpoint,推理端 `load_state_dict` 用存好的对齐值**覆盖**临时重采样值 —— 对齐后的位置编码被**焙进 ckpt**,推理/TTA/SWA 自动一致,无需在推理端重复指定。

## 8. 文件清单

```
env.sh            可配置项 + 冻结配方(先改这里)
arms.tsv          对照臂定义(name img pos bs scales tier)
00_smoke.sh       跑前自检(建模+前向+锚点保真)
01_build_cache.sh 特征缓存(缺失才建;已打包)
run_arm.sh        单臂:训练 -> SWA -> TTA+balanced
run_all.sh        批量:[core|stretch|all]
requirements.txt  依赖
cache/            打包的 224 特征缓存(train/test_features.pt)
code/             运行期代码快照(finetune_lora / robustft / tools / main / config)
```
