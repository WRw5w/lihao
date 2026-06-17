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

## 🚀 最高优先（新线：FET + 迭代重打标，2026-06-17）

**强烈建议下一发就提交 FET 候选**——这是新架构线（CLIP冻结+LoRA+CBAM局部分支+PFI）+ 迭代重打标去噪的产物，**首个在 90/10 val 上超过 soup 的单模型**，最可能突破 77.7。详见 [docs/冲分路线_80-85.md](../docs/冲分路线_80-85.md)。

| 顺序 | 文件 | val mid | 目的 |
|---:|---|---:|---|
| **0a** | `pred_results_fet_iter1_tta_balanced.zip` | **0.9289** | **迭代重打标round1，首超soup(0.9267)，最想验证的代表** |
| **0b** | `pred_results_fet_512_tta_balanced.zip` | **0.9293** | FET@512+干净标签,val 最高;若 0a 上榜>77.7 则跟上 |
| 0c | `pred_results_fet_soup_tta_balanced.zip` | ~0.929 | FET 强checkpoint汤 |

> **FET 线 val 已平台在 ~0.929**（iter1 0.9289 / iter2 0.9289 / 512 0.9293），全部略超 soup 的 0.9267。**最缺也最关键的是其中任意一个的榜分**——只要 1 发就能判定这条线能否破 77.7。强烈建议 codex **下一发就从 0a/0b 选一个打**（重跑 `aicomp_apply_guide_queue.mjs` 以读取本优先级）。

> 说明：val 不可靠预测榜分，但这是首个 val 超 soup 的单模型；**最缺的就是它的榜分**。优先用 1-2 发验证 FET 线能否转化为 >77.7。若 iter1 上榜 >77.7 → 这条线是冲 80+ 的主攻方向。

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
