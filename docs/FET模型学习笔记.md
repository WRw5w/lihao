# FET 模型学习笔记

> 一本边学边记的笔记：把 FET（Feature Enhancement / 局部特征增强）这套新架构吃透。
> 对象：`robustft/fet_modules.py` + `robustft/fet_model.py` + `finetune_fet.py`。
> 出处：FET-FGVC（Pattern Recognition 2024，NICE-FUTURE/FET-FGVC），队友移植 + 适配 CLIP ViT-B/32。
> 背景目标：学长去年同思路在细粒度任务做到 90+；我们想把它从汤的 77.7 往上拱。
> 写于 2026-06-17（FET 首训进行中），随训练/榜分持续补充。

---

## 0. 一句话理解

> **CLIP 只用一个 CLS 全局向量做分类，对"两个长得几乎一样的类"不够细；FET 额外开一条"局部分支"，把图像切成若干判别性区域（parts）、用注意力强化、再用小 Transformer 建模 part 之间的关系，最后和全局特征融合——给模型装上一双"盯细节"的眼睛。**

细粒度识别（fine-grained）的难点：类间差异极小（同属不同种的鸟、同款不同年份的车），**区分靠局部判别区域**（鸟喙形状、车灯纹理）。全局池化容易把这些细节糊掉。FET 的核心就是显式地"找局部、强化局部、关系建模"。

---

## 1. 总体结构（数据怎么流）

```
输入图 (B,3,448,448)
   │
   ▼  CLIP ViT-B/32 (冻结) + LoRA
   ├─────────────► CLS token  cls_feat (B,768)        ← 全局分支
   └─────────────► patch tokens (B,196,768)           ← 局部分支的原料
                        │ (.detach())
                        ▼  LocalBranch
                   ┌─ 切成 8 个通道组(每组96维)
                   ├─ 每组过 CBAM(通道注意力+空间注意力) → (B,16,14,14)
                   ├─ 拼成 8 个 part token → proj → (B,8,768)
                   ├─ part-Transformer ×2 (建模 part 间关系)
                   └─ 池化 + shortcut → local_feat (B,768)
                        │
   fused = cls_feat + 0.5 × local_feat   ← 全局+局部融合
        │ L2 normalize
        ▼
   CosineClassifier → logits (B,500)
```

训练时**额外**走一条 PFI（见 §5），推理时只走上面主路（`forward(images)` 在 `not training` 时提前 return logits）。

**关键点**：`patch_feats.detach()` —— 局部分支吃的是**detach 后**的 patch 特征，梯度不回传到骨干/LoRA。也就是说 LoRA 只通过 cls_feat 这条路学习，局部分支是"在冻结特征上额外加工"。这是稳定性设计，但也意味着局部分支不能反过来调整骨干去突出局部（一个潜在的改进点）。

---

## 2. CBAM：通道注意力 + 空间注意力

CBAM = Convolutional Block Attention Module。两步"问哪里重要"：

### 2.1 通道注意力 `_ChannelGate`
- 对特征图同时做**全局平均池化**和**全局最大池化** → 两个 (B,C) 向量
- 各过同一个共享 MLP（C→C/16→C），相加 → sigmoid → 每个**通道**一个 0~1 权重
- `x * scale`：放大有信息的通道、压制噪声通道
- 直觉：avg 看"整体有没有"，max 看"最强响应在不在"，互补。

### 2.2 空间注意力 `_SpatialGate`
- 沿通道维做 **max + mean** 压缩 → (B,2,H,W)
- 7×7 卷积 → 1 张空间热力图，**乘上 decision_mask**（外部传入的"保留哪些位置"先验）→ sigmoid
- `x * scale`：放大判别性**位置**、压制背景
- 返回 `parts_mask`（这张空间注意力图本身，可用于可视化"模型在看哪")

### 2.3 拼接
`CBAM.forward`：先通道、再空间，最后 `fc`(1×1 卷积) 把通道压到 `part_channels=16`。每个 part 输出 (B,16,14,14)。

---

## 3. LocalBranch：把"通道组"当 parts

这是最 tricky、也最需要清醒看待的设计。

- `embed_dim=768`, `num_parts=8` → 每个 part = **96 个通道**（768/8）。
- **注意**：这里的"part"不是图像上语义区域，而是**特征通道的分组**。把 768 维 patch 特征切成 8 组，假设不同通道组响应不同的判别模式（一种弱假设，靠数据学）。
- 每组 (B,96,14,14) 过各自的 CBAM → (B,16,14,14)。
- 8 组拼回 → 把每个 part 摊平成一个 token：(B, 8, 16×196=3136) → `proj_in` Linear(3136→768) → **8 个 part token (B,8,768)**。
- `decision_mask`：在我们的 CLIP 适配里是**全 1**（CLIP ViT-B/32 没有动态 token 剪枝，所以不给空间先验，让 CBAM 的空间门自己学）。先经 **15×15 高斯核平滑**（让 mask 连续、不突兀）。

### 3.1 part-Transformer（替代原论文的 GCN）
- 原 FET 用图卷积(GCN)建模 part 之间的关系，依赖 `torch_geometric`。
- 移植版**换成 1~2 层标准 Transformer encoder**（MHA+MLP，over 8 个 part token），**去掉了重依赖**，效果近似——8 个 token 的自注意力本就能学 part 间关系。
- 输出 `mean(part tokens) + shortcut(patch 特征均值)` → local_feat (B,768)。

> 学习点：GCN→Transformer 是常见的"去依赖"等价替换。当 node 数很少(8)且无明确图结构时，full self-attention 是更省事的关系建模器。

---

## 4. 融合与分类头
- `fused = cls_feat + local_scale × local_feat`，`local_scale=0.5`（局部只占一半权重，主仍是全局）。
- L2 normalize → **CosineClassifier**（余弦分类头，和我们 LoRA 线同款，对噪声/长尾更稳，因为它归一化掉特征模长）。
- `local_scale` 是个关键旋钮：太大→被噪声局部带偏，太小→等于没加。0.5 是起点，值得扫。

---

## 5. PFI：Pair Feature Interaction（成对特征交互）

训练时**专属**的辅助任务，思想接近"度量学习/对比"：

1. 在一个 batch 内，用 cls_feat 的成对距离，给每个样本找：
   - **同类最近**（intra：最像的同类样本）
   - **异类最近**（inter：最像的不同类样本——最容易混的"对手"）
2. 交换/混合它们的注意力特征（`_pfi_attention_feature`：用 |特征|.mean.softmax 当空间权重加权 patch token 再池化），构造 4 组特征：self / intra / inter-self / inter-other。
3. 这 4 组各自过分类头 → **辅助 CE 损失**，权重 `pfi-weight=0.5`。

**作用**：逼模型"即便看最像的同类/最难的异类，也要分对"——拉近类内、推远类间，增强判别边界。对**噪声标签**尤其可能有用：难负样本挖掘能稳住决策面。

**前提**：一个 batch 必须有**同类样本对** → 需要 **P×K 平衡采样器**（`BalancedBatchSampler`，`--pfi-classes 4 × --pfi-images 8`，每 batch 4 类×每类 8 图）。随机采样在 500 类下几乎没同类对，PFI 就退化了。

> ⚠️ 清醒认知：移植版的 `_pfi_attention_feature` 注释自承是 **simplified approximation**——用的是特征幅值当空间权重，**不是真正的 ViT 注意力权重**。所以这版 PFI 是"近似版"，未必等于原论文拿到 90+ 的那个 PFI。这是后续要验证/还原的点。

---

## 6. 怎么接我们的噪声鲁棒管线（复用，不重造）

`finetune_fet.py` 的 `prepare_targets` 和 LoRA 线**逐行一致**，直接继承我们所有去噪资产：
- kNN 一致性过滤（keep_ratio 0.90 + high-agreement-floor）
- 教师头 + kNN 共识**伪标签回收**
- 多信号**可靠性加权**（agree×confidence×margin）
- **软伪标签融合**（pseudo_soft_alpha）
- 严格 split-first（seed42，统计只用训练分区）
- **共享特征缓存** `outputs/cache/train_features.pt`，不重提取

→ FET 不是从零开新管线，而是**在我们打磨好的去噪地基上，换了个更强的分类架构**。这也让 FET 的结果能和 LoRA 线公平对比（同去噪、同 split、同推理）。

---

## 7. 合规性自查（赛规）

| 赛规要求 | FET 是否满足 | 说明 |
|---|---|---|
| 必用 CLIP ViT-B/32 骨干 | ✅ | `timm vit_base_patch32_clip_224.openai` |
| 不宜无约束全参微调 | ✅ | 骨干 `requires_grad_(False)` 全冻结，只训 LoRA+LocalBranch+head |
| 单模型单推理 | ✅ | 一个模型一次 forward；`checkpoint["single_model"]=True`；PFI 仅训练期 |
| 禁集成/融合/投票 | ✅ | LocalBranch 是**模型内部**特征融合，不是多模型 |
| 测试集仅推理 | ✅ | predict 路径不碰训练 |

trainable = **17.58M**（LoRA 2.7M + LocalBranch/part-Transformer ~12M + head + pfi）。虽比纯 LoRA 大，但骨干 88M 全冻结 → 属 **PEFT + 任务头**，合规。**风险**：额外容量可能更易过拟合噪声标签，靠我们的去噪 + EMA + 权重衰减压住。

---

## 8. 工程实况（8GB / RTX 5060）

- **显存**：作者默认 batch=4×16=64@448 → 必 OOM。实测 batch=**4×8=32** 峰值 **6.2GB / 8151**，稳。`--no-pin` 防 DataLoader 锁页僵死。
- **三组学习率**：LoRA 2e-4 / 局部分支 5e-4 / head 1e-3（局部和头新初始化，学得快一点）。EMA 0.999。OneCycleLR。
- **速度**：~14-18 min/epoch（83k 训练图），6 轮 ~1.5-2h + 满推理 ~10min。

### 8.1 实训抓到并修掉的两个 bug
1. **采样器 off-by-one → OneCycleLR 越界崩溃**：`BalancedBatchSampler.__len__` 报 N，`__iter__` 的 `while seen+bs<=total` 在某 epoch 多吐 1 个 batch（stepped 269 > total_steps 268）。修法：`__iter__` 用 `for _ in range(len(self))` **精确吐 len 个 batch**。
2. **推理 strict-load 崩溃（靠推理预判抓到）**：checkpoint 存的是 EMA 模型（`use_pfi=True`，含 `pfi_norm` 参数），而推理建模型 `use_pfi=False` → `load_state_dict` 严格模式拒绝多余的 `pfi_*` 键。修法：`strict=False` + 断言只丢 `pfi_*`（PFI 是训练专属，推理 forward 提前 return，用不到）。

> 教训：**训练能跑 ≠ 推理能跑**。PFI 这种"训练期专属模块"会让 save/load 的键不一致，smoke（只训练）抓不到，必须单独做推理 sanity。

---

## 9. 为什么它*可能*比汤强（也为什么要保持怀疑）

**乐观面**：
- 汤是"把多个同款模型平均"，本质还是同一种全局特征，**天花板受限于单模型架构**（我们看到汤 77.7 就上不去了，扩池/多样化都失败）。
- FET 换的是**架构本身**（加局部判别能力），是**正交**的提升维度——理论上能突破汤的天花板。
- 细粒度任务局部信息确实关键，CBAM+part 关系是被验证过的有效设计。

**怀疑面（不臆断，等榜分）**：
- "90+"是学长在**别的数据/可能别的约束**下拿到的，未必迁移到我们这个噪声+均衡测试榜。
- 移植版 PFI 是**近似**，未必复现原效果。
- `patch_feats.detach()` 限制了局部分支对骨干的反向影响。
- "part = 通道分组"是弱假设。
- 多 17.58M 参数在噪声标签上有**过拟合风险**。

**现实目标**：先看 90/10 val + 满推理榜分能不能**稳过 77.7**；90+ 当上限期待。哪怕只 +1pt 也是汤给不了的正交收益。

---

## 9.5 首训结果与解读（2026-06-17，重要负向信号）

**FET baseline（90/10，6轮，batch 4×8=32，PFI on，local_scale0.5）实测：**

| epoch | loss | noisy_all | mid_03_06 | high |
|--:|--:|--:|--:|--:|
| 1 | 3.96 | 0.688 | 0.8316 | 0.989 |
| 2 | 3.60 | 0.727 | 0.8736 | 0.993 |
| 3 | 3.30 | 0.753 | 0.8990 | 0.996 |
| 4 | 3.01 | 0.768 | 0.9162 | 0.996 |
| **5** | 2.80 | 0.776 | **0.9184** ⭐ | 0.995 |
| 6 | 2.70 | 0.777 | 0.9177↓ | 0.996 |

**对比我们的 LoRA 线**：FET val mid **0.9184** < LoRA 单模型 **0.9244** < soup **0.9267**。**FET 在两个本地代理上都略低于现有 LoRA。**

**解读（学习要点）**：
1. **loss 一路降(3.96→2.70)、noisy_all 一路升到 ep6、mid 在 ep5 见顶后 ep6 回落** → 典型的**后期开始记忆噪声标签**（early-learning 之后进入 memorization 阶段）。
2. **多的 17.58M 参数(LocalBranch/part-Transformer)在 37% 噪声上倾向过拟合**——容量越大越能记住错标，干净泛化反而被拖。这解释了为何 FET < LoRA。
3. **印证战略诊断：瓶颈是噪声不是架构**。学长的 90+ 大概率是在**干净数据**上——干净时局部增强是纯增益；噪声时额外容量是双刃。
4. 推论：**FET 的潜力要在标签更干净后才放得出来** → 先用 ELR/迭代去噪压住记忆，再让 FET 的局部判别发力。
5. ⚠️ val 代理不可靠预测榜分，FET baseline 已提交，等干净测试榜分可能与 val 不同向——但本地双代理一致偏低，是需要正视的信号。

**对策（已/将执行）**：
- ✅ 接 **ELR**（直接打"后期记忆噪声"——把预测锁在 early-learning 的干净目标上）。FET+ELR vs FET baseline 干净 A/B。
- 降 `local_scale`(0.5→0.3)、`pfi_weight=0` 消融，看是不是局部分支/PFI 在噪声上添乱。
- 缩容量（`num_parts`/`local_depth`/`part_channels`）降过拟合。

## 10. 待办 / 可调旋钮（出 val 后按榜分推进）

- [ ] FET 首训出 val mid → 满推理（多尺度+均衡）→ 上榜 A/B vs soup_sweep 77.73。
- [ ] 若有潜力：`--full` 满数据重训（峰值轮）+ 多 seed 进汤（FET 同架构也能 soup）。
- [ ] 旋钮扫：`local_scale`(0.3/0.5/0.7)、`num_parts`(4/8/16)、`pfi_weight`(0/0.3/0.5)、`local_depth`(1/2/3)。
- [ ] 验证 PFI 是否真有用：`pfi_weight=0` 做消融（关掉 PFI 只留 LocalBranch）。
- [ ] 还原"真注意力"版 PFI（用 ViT 真实 attention 而非幅值近似），看能否逼近 90+。
- [ ] 让局部分支**不 detach**，看反向影响骨干能否进一步提升（小心显存/稳定性）。

---

## 附：组件速查
- **CBAM** = 通道注意力(avg+max→MLP) ⊕ 空间注意力(max+mean→7×7conv×mask)
- **LocalBranch** = 8 通道组 × CBAM → 8 part token → part-Transformer → 局部向量
- **PFI** = batch 内同类/异类最近对 → 注意力特征交换 → 辅助 CE（需 P×K 采样器）
- **融合** = cls + 0.5×local → cosine 头
- **去噪** = 完全复用 LoRA 线的 prepare_targets
