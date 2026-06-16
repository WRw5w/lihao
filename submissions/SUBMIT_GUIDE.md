# 打榜提交指南（给 codex）— 2026-06-16 修正版

## 原则（重要修正）
- **已验证的只是"推理配方"**：满均衡(strength 1.0) 比无均衡 **+2.3 分**、满 TTA(448/512/576) 比精简TTA **+1.1 分**（实测）。→ 推理一律用「满 TTA + 满均衡」。
- **"模型/策略之间谁更强"还没验证**：除了 5合一(75.66) 和 soup_uniform(76.7)，**其他策略都没上过榜**。第一轮是**广度搜索**：尽量多测不同策略，每策略 **2 个 seed**（消除训练方差）。
- ⚠️ 不要臆断"soup 一定 > 单模型"——那些都要实测。

## 15 发广度分配（今晚→明天中午）
推理固定满TTA+满均衡，只变策略；每个单模型策略跑 2 个 seed：

| 策略 | 发数 | 说明 |
|---|---|---|
| soup_uniform | 1 | 76.7 基准复核 |
| 单冠军 LoRA(rank32/keep0.90/CE) | 2 | seed42+seed1 |
| GCE 鲁棒损失 | 2 | seed42+seed1 |
| conservative(加权和+软伪标签) | 2 | seed42+seed1 |
| keep0.85（弱召回） | 2 | seed42+seed1 |
| keep0.95（强召回） | 2 | seed42+seed1 |
| soup_v2 / soup_v3（多样化汤） | 2 | 各1 |
| 5合一 | 1 | 75.66 复核 |
| 备用 | 1 | |

---

## ✅ 现在就能交的广度候选（满推理、不同策略，已就绪）
codex 可立即从这些开交（每个不同策略 = 一条独立信息）：
- `pred_results_soup_uniform_tta_balanced.zip`     # soup（=76.7 基准）
- `pred_results_soup_v2_tta_balanced.zip`          # 多样化汤
- `pred_results_conservative_tta_balanced.zip`     # 李洋保守化（加权和+软伪标签）
- `pred_results_swa_champion_tta_balanced.zip`     # 满数据 SWA
- `pred_results_c448_dr_rank32_keep90_tta_balanced.zip`  # 单冠军模型（之前误判skip，实为待测策略）
- `pred_results_soup_greedy_tta_balanced.zip`      # 贪心汤（不同融合策略）

## ⏳ 我会补生成的广度候选（凑齐 15 发）
- `soup_v3`（Phase1 多样化大汤，跑完自动 push）
- GCE 满TTA版（现有 gce 只有单尺度，要重出满TTA）
- keep0.85 / keep0.95 的满推理版
- 上述单模型策略的 **seed1 第二份**（补训）

---

## ❌ 确认更差·别交（仅"推理降级/旧基线"，不含任何待测的不同模型）
### 无均衡 `_tta`（缺 balance，实测 −2.3 分）—— 全部跳过
pred_results_*_tta.zip（所有不带 `_balanced` 的：c448_dr_rank32_keep90_tta / run60_ep60_tta / swa_run60_tta / swa_champion_tta / soup_uniform_tta / soup_greedy_tta / 5sig_tta / 5sig_ep8_tta / 5sig_full_tta / 5sig_full_v2_tta / soup_sweep_tta / conservative_tta / soup_v2_tta）
### 5合一精简/弱均衡变体（已上榜或被 full_v2 75.66 支配）
pred_results_5sig_tta_balanced.zip
pred_results_5sig_ep8_tta_balanced.zip        # 已上榜 73.56
pred_results_5sig_full_tta_balanced.zip       # 已上榜 74.23
pred_results_5sig_full_v2_tta.zip             # 已上榜 73.36
### 旧基线 / 单尺度推理（被满推理同模型支配）
pred_results.zip / pred_results_head.zip / pred_results_lora.zip   # 旧版(~61)
pred_results_c448_dr_rank32_keep90.zip               # 单冠军的"基础推理"（无均衡）→ 用上面 `_tta_balanced` 版代替
pred_results_c448_dr_rank32_keep90_balanced.zip      # 单尺度+均衡（−1.1 vs 多尺度）
pred_results_c448_dr_rank32_keep90_balanced_s05.zip  # 单尺度+弱均衡
### balance λ<1（~90% 更差，非100%；有富余只试 s0.75）
pred_results_soup_sweep_tta_balanced_s0.25.zip / s0.5 / s0.75

## 🟡 可选低优先（更弱模型/旧模型，时间够再测）
c448 / drecall / run60_ep60 / swa_run60 等——单模型且偏弱，要测得先重出满推理版；优先级低于上面 6 个。

---
**一句话**：推理降级版安全跳过；**所有不同的"模型/策略"都进广度测、不臆断**；每策略 2 seed 看稳定性。
