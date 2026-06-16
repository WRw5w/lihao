# 打榜提交指南（给 codex）— 2026-06-16

**依据（已落地的真实榜分）**：
- 满均衡(balance strength=1.0) 比无均衡 **+2.3 分**（5sig_full_v2: 75.66 vs 73.36，实测）。
- 满 TTA(448/512/576) 比精简TTA(仅448) **+1.1 分**（实测）。
- `soup_uniform`=**76.7** 是当前最优基准；5合一线已死（满推理也只 75.66 < 76.7）。
- **结论：只交「满TTA(448/512/576)+满均衡(1.0)」的 `_tta_balanced` 强候选。**

---

## ✅ 该交（按优先级，每小时额度只花这里）
- `pred_results_soup_v3_tta_balanced.zip`   ← Phase1 多样化大汤（生成后最看好，未上榜）
- `pred_results_soup_v2_tta_balanced.zip`   ← 多样化汤（未上榜）
- `pred_results_conservative_tta_balanced.zip` ← 李洋保守化模型（未上榜）
- `pred_results_swa_champion_tta_balanced.zip` ← 满数据 SWA（未上榜）
- `pred_results_soup_uniform_tta_balanced.zip` ← =76.7 基准（已知，可复核）
- `pred_results_c448_gce_balanced.zip`      ← GCE 模型（唯一未测的独立模型；注：单尺度推理非最优，仅供参考）

---

## ❌ 别交（确认/几乎确认更差，浪费每小时额度）

### 1) 无均衡版（文件名 `_tta` 不带 `_balanced`）—— 实测 −2.3 分，确认更差
pred_results_c448_dr_rank32_keep90_tta.zip
pred_results_run60_ep60_tta.zip
pred_results_swa_run60_tta.zip
pred_results_swa_champion_tta.zip
pred_results_soup_uniform_tta.zip
pred_results_soup_greedy_tta.zip
pred_results_5sig_tta.zip
pred_results_5sig_ep8_tta.zip
pred_results_5sig_full_tta.zip
pred_results_5sig_full_v2_tta.zip
pred_results_soup_sweep_tta.zip
pred_results_conservative_tta.zip
pred_results_soup_v2_tta.zip

### 2) 5合一系（精简TTA+弱均衡，被 full_v2 的 75.66 碾压，且5合一线已死）
pred_results_5sig_tta_balanced.zip
pred_results_5sig_ep8_tta_balanced.zip          # 已上榜=73.56
pred_results_5sig_full_tta_balanced.zip         # 已上榜=74.23
pred_results_5sig_full_v2_tta_balanced.zip      # 已上榜=75.66（无需重交）

### 3) 旧/基线/单尺度推理（都被满推理碾压，确认更差）
pred_results.zip                                  # 旧交错文件(~61)
pred_results_head.zip                             # 旧头部基线
pred_results_lora.zip                             # 旧 v2 LoRA
pred_results_c448.zip                             # 基础推理(flip-only,无均衡)
pred_results_c448_drecall.zip
pred_results_c448_gce.zip
pred_results_c448_dr_rank32_keep90.zip            # 基础推理,无均衡
pred_results_c448_dr_rank32_keep90_balanced.zip   # 单尺度+均衡(−1.1 vs 多尺度)
pred_results_c448_dr_rank32_keep90_balanced_s05.zip
pred_results_c448_dr_rank32_keep90_tta_balanced.zip  # 单模型满推理,被 soup(76.7) 碾压

### 4) 弱模型的满推理版（已知/极可能 <76.7）
pred_results_run60_ep60_tta_balanced.zip    # 90/10 单轮,弱
pred_results_swa_run60_tta_balanced.zip     # 欠佳 run60 的 SWA
pred_results_soup_greedy_tta_balanced.zip   # =单模型,被 uniform 汤碾压

### 5) balance λ<1（很可能 <λ=1 的 76.7；~90% 确信，非 100% 直接验证）
pred_results_soup_sweep_tta_balanced_s0.25.zip
pred_results_soup_sweep_tta_balanced_s0.5.zip
pred_results_soup_sweep_tta_balanced_s0.75.zip

---

**确定性说明**：第 1/2/3 组是**确认更差**（直接榜分证据或被同模型更优推理完全支配）；第 4 组弱模型大概率更差；第 5 组 λ<1 是 ~90% 确信（无直接 soup-λ A/B，若有富余额度可只试 s0.75）。
