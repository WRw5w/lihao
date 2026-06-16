# 打榜提交指南（Codex 执行版）— 2026-06-16

## 复核原则

- 只有 `aicomp_results.csv` / `aicomp_leaderboard_snapshots.log` 里由脚本抓到的结果，才算已验证榜分。
- 人工记录只作为线索，不作为跳过依据；`soup_uniform=76.7`、`c448_dr_rank32_keep90 显著提升`、`run60_ep60 -1` 都需要重新落到 CSV。
- 第一轮目标是广度搜索：每个不同模型/策略先 1 发，覆盖最大策略空间；2 seed 留到第二轮稳定性确认。
- 推理降级版跳过：所有无 balance 的 `_tta.zip`、旧基线、单尺度替代品，除非它本身就是要验证的唯一可用策略。

## 已脚本确认

| 文件 | 分数 | 结论 |
|---|---:|---|
| `pred_results_5sig_full_v2_tta_balanced.zip` | 75.6639 | 5sig 最强已确认 |
| `pred_results_5sig_full_v2_tta.zip` | 73.3648 | 无 balance 比 full balance 低 2.2991 |
| `pred_results_5sig_full_tta_balanced.zip` | 74.2300 | 5sig full 旧版低于 v2 |
| `pred_results_5sig_full_tta.zip` | 72.2634 | 无 balance 继续劣化 |
| `pred_results_5sig_ep8_tta_balanced.zip` | 73.5571 | 5sig ep8 不够强 |
| `pred_results_5sig_tta_balanced.zip` | 73.6212 | 5sig lean 版低于 full v2 |

5sig 分支第一轮已结束，后续转向 soup / c448 / SWA / 策略差异验证。

## 今晚执行队列

当前状态：

- `pred_results_soup_v2_tta_balanced.zip` 可能已在 2026-06-16 22:07 左右提交，runner 正在等待 23:05 抓榜确认；只有榜单提交时间晚于本次提交开始时间才算有效。
- `pred_results_soup_uniform_tta_balanced.zip` 的 22:05 抓榜是旧分 `73.6212`，不算验证，已退回 pending，等当前 awaiting 项确认后再重交。

后续顺序按“信息量 + 已就绪 + 策略差异”排序。

| 顺序 | 文件 | 策略目的 |
|---:|---|---|
| 0 | `pred_results_5sig_tta_balanced.zip` | 已提交，只抓结果 |
| 1 | `pred_results_soup_uniform_tta_balanced.zip` | 复核人工 76.7，正式落 CSV |
| 2 | `pred_results_soup_v2_tta_balanced.zip` | 多样化 soup，新增强候选 |
| 3 | `pred_results_soup_greedy_tta_balanced.zip` | 最强单模型/贪心 soup 锚点 |
| 4 | `pred_results_c448_dr_rank32_keep90_tta_balanced.zip` | 单冠军 LoRA 满推理，复核结构主线 |
| 5 | `pred_results_conservative_tta_balanced.zip` | 保守化训练策略 |
| 6 | `pred_results_swa_champion_tta_balanced.zip` | 满数据 SWA 是否有效 |
| 7 | `pred_results_swa_run60_tta_balanced.zip` | 长训练后段 SWA 是否能救回 |
| 8 | `pred_results_soup_v3_tta_balanced.zip` | 若生成完成则自动纳入下一轮重排 |
| 9 | `pred_results_c448_gce_tta_balanced.zip` | 若补生成 full TTA GCE，则优先于单尺度 GCE |
| 10 | `pred_results_keep85_tta_balanced.zip` | 若补生成，验证弱召回 |
| 11 | `pred_results_keep95_tta_balanced.zip` | 若补生成，验证强召回 |
| 12 | `pred_results_aug06_tta_balanced.zip` | 若补生成，验证强增强 |
| 13 | `pred_results_ema9995_tta_balanced.zip` | 若补生成，验证高 EMA |
| 14 | `pred_results_drecall_tta_balanced.zip` | 若补生成，复核旧冠军前身 |

## 低优先备用

这些不是第一选择，但如果补生成文件暂时未出现、又需要继续占用整点窗口，可以作为 fallback：

- `pred_results_c448_gce_balanced.zip`：GCE 单尺度 balanced，低于 full TTA GCE。
- `pred_results_run60_ep60_tta_balanced.zip`：复核人工 `-1` 记录。
- `pred_results_soup_sweep_tta_balanced_s0.75.zip`：balance strength 曲线最高 λ 备用点。

## 明确跳过

- 所有不带 `_balanced` 的 `_tta.zip`：缺 balance。
- `pred_results.zip`、`pred_results_head.zip`、`pred_results_lora.zip`、`pred_results_c448.zip`、`pred_results_c448_drecall.zip`：旧基线/旧单尺度。
- `pred_results_5sig_ep8_tta.zip`、`pred_results_5sig_tta.zip`、`pred_results_5sig_full_v2_tta.zip`：5sig 无 balance 或已验证弱。
- `pred_results_c448_dr_rank32_keep90.zip`、`pred_results_c448_dr_rank32_keep90_balanced.zip`、`pred_results_c448_dr_rank32_keep90_balanced_s05.zip`：被同模型 full TTA balanced 目标支配。

## 执行命令

浏览器建议用专用启动脚本打开，避免 Chrome 后台节流/内存节省影响长时间等待：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\aicomp_start_chrome.ps1 -Restart
node tools\aicomp_cdp.mjs wait-login 300000
```

如果已经手动登录且页面正常，不必强制重启；runner 等待整点时会每 2 分钟 heartbeat 一次，并在遇到 `共 0 条数据`、排行榜 `暂无数据` 时自动刷新重试。默认保持两个 AICOMP tab：一个提交页，一个排行榜页；提交和抓榜不会再反复复用同一个 tab。runner 会优先处理 `awaiting_refresh`，抓到新提交时间的新分后才会提交下一个 pending，防止跨整点提交被下一发覆盖。

查看当前 tab 识别：

```powershell
node tools\aicomp_cdp.mjs pages
```

```powershell
node tools\aicomp_apply_guide_queue.mjs
node tools\aicomp_submit_queue.mjs run
```

运行中如果新文件生成，先暂停 runner，再执行 `node tools\aicomp_apply_guide_queue.mjs` 重排，最后恢复 runner。
