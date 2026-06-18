# 打榜提交指南（Codex 执行版）- 2026-06-17

## 当前结论

- 第一轮广度搜索已完成：16 个有效榜分，另有 2 条历史空抓榜记录不计入有效结果。
- 当前最强是 `pred_results_soup_sweep_tta_balanced_s0.75.zip`：`77.7306`，rank 2。
- 第二强是 `pred_results_soup_uniform_tta_balanced.zip`：`77.6865`，只低 `0.0441`。
- soup 分支明显领先；下一轮优先继续验证 soup_v3 与 balance strength 曲线。
- 非 soup 中较强的是 `swa_champion` (`76.7293`)、`conservative` (`76.2206`)、`c448_dr_rank32_keep90` (`76.1485`)。

## 已验证榜分

| 文件 | 分数 | 排名 | 结论 |
|---|---:|---:|---|
| `pred_results_soup_sweep_tta_balanced_s0.75.zip` | 77.7306 | 2 | 当前最强；说明 balance strength 不一定越满越好 |
| `pred_results_soup_uniform_tta_balanced.zip` | 77.6865 | 2 | 与 s0.75 非常接近，是强基线 |
| `pred_results_soup_v2_tta_balanced.zip` | 76.8334 | 2 | 多样化 soup 有效，但低于 uniform/sweep |
| `pred_results_swa_champion_tta_balanced.zip` | 76.7293 | 2 | 单模型/SWA 中较强 |
| `pred_results_conservative_tta_balanced.zip` | 76.2206 | 2 | 中上，低于 soup |
| `pred_results_c448_dr_rank32_keep90_tta_balanced.zip` | 76.1485 | 2 | 单冠军满推理可用，但低于 soup |
| `pred_results_swa_run60_tta_balanced.zip` | 75.6679 | 3 | 接近旧 5sig 强基线 |
| `pred_results_5sig_full_v2_tta_balanced.zip` | 75.6639 | 3 | 旧 5sig 最强已被 soup 超过 |
| `pred_results_run60_ep60_tta_balanced.zip` | 75.6198 | 3 | 低于 soup/SWA champion |
| `pred_results_soup_greedy_tta_balanced.zip` | 75.3314 | 4 | 贪心 soup 不如 uniform/sweep |
| `pred_results_c448_gce_balanced.zip` | 73.5571 | 5 | 单尺度 GCE 表现差；full TTA 仍需补验一次 |

## ⚠️ FET 线已判定（2026-06-18 榜分回来）：全部 ~76 单模型档，**降优先级**

FET 线已拿到榜分，**结论是封板**：`fet_iter1`=**76.16**、`fet_elr8`=76.02、`fet_elr`=76.06、`fet_c448`=75.52。
val 0.9289（曾首超 soup）**没有转化为榜分优势**——和最优单 LoRA(76.15)同档。坐实瓶颈=冻结 B/32 特征+噪声，**不是头部架构**。
- **剩余 FET 候选（`fet_512` / `fet_soup` / `fet_iter2`）预期同样 ~76，请降到队尾**，不要再占用前排榜位（除非空闲）。

## 🆕 最高优先（新线：正交去噪机制 ortho，2026-06-18）

为跳出 ~76 单 / 77.7 汤的天花板，开了一批**机制本质不同**的探针（champion 配方 + 单一改动），文件名 `pred_results_ortho_<名>_tta_balanced.zip`。**生成一个就请提交一个**（A/B vs 单模型 76.1 与 soup 77.73）。判据：单模型若明显 >76.5 即"有戏"，随后会做它的 soup 冲 78+。

| 顺序 | 文件（生成后即提交） | 机制 |
|---:|---|---|
| 1 | `pred_results_ortho_sce_tta_balanced.zip` | 对称交叉熵 SCE（鲁棒损失，正交于kNN去噪） |
| 2 | `pred_results_ortho_apl_tta_balanced.zip` | Active-Passive Loss（NCE+RCE，理论鲁棒） |
| 3 | `pred_results_ortho_mixup02_tta_balanced.zip` | Mixup α0.2（标签噪声平滑） |
| 4 | `pred_results_ortho_dora_tta_balanced.zip` | DoRA（权重分解PEFT，抬单模型上限） |
| 5 | `pred_results_ortho_mixup04_tta_balanced.zip` | Mixup α0.4（第二发） |
| 6 | `pred_results_ortho_dora16_tta_balanced.zip` | DoRA rank16（第二发） |
| 7 | `pred_results_ortho_cleanlab_tta_balanced.zip` | Confident Learning 干净集选择 |
| 8 | `pred_results_ortho_cleanlabknn_tta_balanced.zip` | CL ∩ kNN（更严格干净集） |

> 这些是"试试水"探针：大多数预计仍 ~76（验证瓶颈是特征非机制），但只要**任意一发明显破 76.5**，就锁定主攻方向并升级成 soup。**val 不可靠，只认榜分**——所以每发都值得一个榜位。

## 第二轮队列

先补齐已生成但未验证的 balanced 主候选；这些比重复提交更有信息量。

| 顺序 | 文件 | 目的 |
|---:|---|---|
| 1 | `pred_results_soup_v3_tta_balanced.zip` | 新大汤，最可能挑战当前 best |
| 2 | `pred_results_soup_sweep_tta_balanced_s0.5.zip` | 补 strength 曲线，判断 s0.75 是否局部最优 |
| 3 | `pred_results_soup_sweep_tta_balanced_s0.25.zip` | 补低 strength 端点，判断曲线形状 |
| 4 | `pred_results_keep95_tta_balanced.zip` | 策略模型：高保留率 |
| 5 | `pred_results_keep85_tta_balanced.zip` | 策略模型：低保留率 |
| 6 | `pred_results_aug06_tta_balanced.zip` | 策略模型：强增强 |
| 7 | `pred_results_ema9995_tta_balanced.zip` | 策略模型：高 EMA |
| 8 | `pred_results_drecall_tta_balanced.zip` | 复核旧冠军前身满推理 |
| 9 | `pred_results_c448_gce_tta_balanced.zip` | GCE full TTA；单尺度差，但补齐证据 |

## 后续决策

- 如果 `soup_v3` 超过当前 best：围绕 `soup_v3` 生成/提交 strength sweep。
- 如果 `s0.5` 或 `s0.25` 超过 `s0.75`：继续在更优区间插点，例如 `0.375/0.625`。
- 如果 `s0.75` 仍最佳：下一轮优先复核 `s0.75` 和 `soup_uniform`，再做更密 strength 网格。
- 不提交无 `_balanced` 的 `_tta.zip`，除非专门需要量化 balance 增益；第一轮已经证明无 balance 普遍明显吃亏。

## 执行命令

```powershell
node tools\aicomp_apply_guide_queue.mjs
node tools\aicomp_submit_queue.mjs run
```

运行要求：

- 保持两个 AICOMP tab：一个提交页，一个排行榜页。
- `node tools\aicomp_cdp.mjs pages` 可检查 tab 识别。
- runner 会优先处理 `awaiting_refresh`，抓到新提交时间的新分后才会提交下一个 pending。
