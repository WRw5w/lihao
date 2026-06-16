# 打榜提交指南（给 codex）— 2026-06-16 修正版 v2

## 原则
- **已验证的只是"推理配方"**：满均衡(strength 1.0) 比无均衡 **+2.3 分**、满 TTA(448/512/576) 比精简TTA **+1.1 分**（实测）。→ 推理一律「满 TTA + 满均衡」。
- **"模型/策略谁更强"基本没验证**：除 5合一(75.66) 和 soup_uniform(76.7，且这是口头记录、未正式落 CSV)，其他都没上过榜。
- **第一轮 = 广度搜索，每候选 1 发**（不是 2 seed）。理由：新产出还在陆续出（Phase1 的 soup_v3 + 5 个策略模型），1 发/候选能覆盖最多不同策略；额度优先用在"没验证过的不同策略"和"复核老的人工记录"上。
- ⚠️ 不臆断"soup 一定 > 单模型"——实测说话。

## 15 发分配（每候选 1 发，今晚→明天中午）
推理固定满TTA+满均衡，只变策略：

| # | 候选 | 类型 |
|---|---|---|
| 1 | soup_uniform | **复核老记录**(确认 76.7 并正式落 CSV) |
| 2 | soup_v2 | 新·多样化汤 |
| 3 | soup_greedy | 新·贪心汤 |
| 4 | 单冠军 c448_dr_rank32_keep90 | 新·单模型(之前误skip) |
| 5 | conservative | 新·李洋保守化 |
| 6 | swa_champion | 新·满数据SWA |
| 7 | swa_run60 | 新·run60 SWA |
| 8 | soup_v3 | 新·Phase1大汤(补生成) |
| 9 | gce(满TTA) | 新·GCE鲁棒损失(补生成) |
| 10 | keep0.85 | 新·弱召回(补生成) |
| 11 | keep0.95 | 新·强召回(补生成) |
| 12 | aug06 | 新·强增强(补生成) |
| 13 | ema9995 | 新·高EMA(补生成) |
| 14 | drecall(满TTA) | 新·旧冠军前身(补生成,选) |
| 15 | 5sig_full_v2 | 复核(已知75.66,选) |

---

## ✅ 现在就能交（满推理、已就绪，codex 从这 6 个开交）
- `pred_results_soup_uniform_tta_balanced.zip`     # 复核 76.7
- `pred_results_soup_v2_tta_balanced.zip`
- `pred_results_conservative_tta_balanced.zip`
- `pred_results_swa_champion_tta_balanced.zip`
- `pred_results_c448_dr_rank32_keep90_tta_balanced.zip`  # 单冠军
- `pred_results_soup_greedy_tta_balanced.zip`
- （`pred_results_swa_run60_tta_balanced.zip` 也可，run60 SWA）

## ⏳ 我会补生成（满推理，凑齐 15）
soup_v3(Phase1) / gce满TTA / keep0.85 / keep0.95 / aug06 / ema9995 / drecall满TTA —— 生成后 push，文件名形如 `pred_results_<name>_tta_balanced.zip`。

---

## ❌ 确认更差·别交（仅"推理降级/旧基线"）
### 无均衡 `_tta`（缺 balance，实测 −2.3）—— 所有不带 `_balanced` 的一律跳过
### 5合一精简/弱均衡变体（已上榜或被 75.66 支配）
pred_results_5sig_tta_balanced / 5sig_ep8_tta_balanced(73.56) / 5sig_full_tta_balanced(74.23) / 5sig_full_v2_tta(73.36)
### 旧基线/单尺度（被满推理同模型支配）
pred_results.zip / head / lora；c448_dr_rank32_keep90.zip(基础推理) / _balanced.zip(单尺度) / _balanced_s05.zip
### balance λ<1（~90%更差，富余只试 s0.75）
pred_results_soup_sweep_tta_balanced_s0.25 / s0.5 / s0.75

---
**一句话**：推理降级版跳过；**每个不同"模型/策略"交 1 发**覆盖最大广度；老记录(soup_uniform 76.7)复核落账；2 seed 留到第二轮（找到强策略后再做稳定性确认）。
