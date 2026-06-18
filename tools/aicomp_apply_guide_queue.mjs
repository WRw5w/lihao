import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const ROOT = "D:\\02_Projects\\ML\\jinyinsai";
const SUBMISSIONS = path.join(ROOT, "submissions");
const QUEUE_PATH = path.join(SUBMISSIONS, "aicomp_submit_queue.json");
const MANIFEST_SCRIPT = path.join(ROOT, "tools", "aicomp_manifest.mjs");

const GUIDE_ORDER = [
  // 2026-06-18 ortho 正交机制探针(最高优先): 生成后即提交, A/B vs 76.1 单 / 77.73 汤
  "pred_results_ortho_sce_tta_balanced.zip",
  "pred_results_ortho_apl_tta_balanced.zip",
  "pred_results_ortho_mixup02_tta_balanced.zip",
  "pred_results_ortho_dora_tta_balanced.zip",
  "pred_results_ortho_mixup04_tta_balanced.zip",
  "pred_results_ortho_dora16_tta_balanced.zip",
  "pred_results_ortho_cleanlab_tta_balanced.zip",
  "pred_results_ortho_cleanlabknn_tta_balanced.zip",
  "pred_results_ortho_fuse4_tta_balanced.zip",
  "pred_results_ortho_fuse6_tta_balanced.zip",
  "pred_results_ortho_attnpool_tta_balanced.zip",
  "pred_results_ortho_mmixup_tta_balanced.zip",
  "pred_results_ortho_curriculum_tta_balanced.zip",
  "pred_results_soup_v3_tta_balanced.zip",
  "pred_results_soup_sweep_tta_balanced_s0.5.zip",
  "pred_results_soup_sweep_tta_balanced_s0.25.zip",
  "pred_results_keep95_tta_balanced.zip",
  "pred_results_keep85_tta_balanced.zip",
  "pred_results_aug06_tta_balanced.zip",
  "pred_results_ema9995_tta_balanced.zip",
  "pred_results_drecall_tta_balanced.zip",
  "pred_results_c448_gce_tta_balanced.zip",
  "pred_results_fet_iter1_tta_balanced.zip",
  "pred_results_fet_elr8_tta_balanced.zip",
  "pred_results_fet_elr_tta_balanced.zip",
  "pred_results_fet_c448_tta_balanced.zip",
  "pred_results_fet_iter1_tta.zip",
  "pred_results_fet_elr8_tta.zip",
  "pred_results_fet_elr_tta.zip",
  "pred_results_fet_c448_tta.zip",
  "pred_results_ema9995_tta.zip",
  "pred_results_drecall_tta.zip",
  "pred_results_c448_dr_rank32_keep90_balanced_s05.zip",
];

const FALLBACK_ORDER = [];

function nowIso() {
  return new Date().toISOString();
}

function readJson(filePath, fallback) {
  if (!fs.existsSync(filePath)) return fallback;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  const tmp = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2), "utf8");
  fs.renameSync(tmp, filePath);
}

function priority(name) {
  const n = name.toLowerCase();
  // ortho 正交机制探针: 最高优先(新机制 > 重复的 ~76 单模型/封板FET)
  if (n.includes("ortho_") && n.includes("balanced")) return 1200;
  if (n.includes("ortho_")) return 1190;
  if (n.includes("soup_uniform")) return 1000;
  if (n.includes("soup_v2")) return 990;
  if (n.includes("soup_greedy")) return 980;
  if (n.includes("c448_dr_rank32_keep90_tta_balanced")) return 970;
  if (n.includes("conservative")) return 960;
  if (n.includes("swa_champion")) return 950;
  if (n.includes("swa_run60")) return 940;
  if (n.includes("soup_v3") && n.includes("balanced")) return 1100;
  if (n.includes("soup_sweep") && n.includes("s0.5")) return 1090;
  if (n.includes("soup_sweep") && n.includes("s0.25")) return 1080;
  if (n.includes("keep95") && n.includes("balanced")) return 1070;
  if (n.includes("keep85") && n.includes("balanced")) return 1060;
  if (n.includes("aug06") && n.includes("balanced")) return 1050;
  if (n.includes("ema9995") && n.includes("balanced")) return 1040;
  if (n.includes("drecall") && n.includes("balanced")) return 1030;
  if (n.includes("c448_gce_tta_balanced")) return 1020;
  if (n.includes("fet_iter1") && n.includes("balanced")) return 1010;
  if (n.includes("fet_elr8") && n.includes("balanced")) return 1008;
  if (n.includes("fet_elr") && n.includes("balanced")) return 1006;
  if (n.includes("fet_c448") && n.includes("balanced")) return 1004;
  if (n.includes("fet_iter1")) return 850;
  if (n.includes("fet_elr8")) return 848;
  if (n.includes("fet_elr")) return 846;
  if (n.includes("fet_c448")) return 844;
  if (n.includes("soup_v3")) return 930;
  if (n.includes("gce")) return 920;
  if (n.includes("keep85")) return 910;
  if (n.includes("keep95")) return 900;
  if (n.includes("aug06")) return 890;
  if (n.includes("ema9995")) return 880;
  if (n.includes("drecall")) return 870;
  if (n.includes("soup_sweep")) return 500;
  if (n.includes("run60")) return 490;
  return 100;
}

function existingZip(name) {
  const filePath = path.join(SUBMISSIONS, name);
  if (!fs.existsSync(filePath)) return null;
  const st = fs.statSync(filePath);
  return {
    name,
    path: filePath,
    size: st.size,
    priority: priority(name),
  };
}

function cloneItem(item) {
  return JSON.parse(JSON.stringify(item));
}

function normalizeItem(item, index) {
  item.index = index;
  item.name ||= path.basename(item.path || "");
  item.path ||= path.join(SUBMISSIONS, item.name);
  if (fs.existsSync(item.path)) {
    const st = fs.statSync(item.path);
    item.size = st.size;
  }
  item.priority = item.priority || priority(item.name);
  item.exitCode ??= "";
  item.submittedAt ??= "";
  item.note ??= "";
  return item;
}

function main() {
  const doc = readJson(QUEUE_PATH, { refreshDelayMinutes: 5, queue: [] });
  const oldQueue = Array.isArray(doc.queue) ? doc.queue : [];
  const byName = new Map(oldQueue.map((item) => [String(item.name || "").toLowerCase(), item]));
  const selected = [];
  const selectedNames = new Set();

  for (const item of oldQueue) {
    if (item.status === "done" || item.status === "awaiting_refresh" || item.status === "submitting") {
      selected.push(cloneItem(item));
      selectedNames.add(String(item.name || "").toLowerCase());
    }
  }

  const addPending = (name, source) => {
    const key = name.toLowerCase();
    if (selectedNames.has(key)) return false;
    const file = existingZip(name);
    if (!file) return false;
    const prev = byName.get(key);
    selected.push({
      ...(prev ? cloneItem(prev) : file),
      ...file,
      status: prev && ["done", "awaiting_refresh", "submitting"].includes(prev.status)
        ? prev.status
        : "pending",
      submittedAt: prev?.submittedAt || "",
      exitCode: prev?.exitCode || "",
      note: prev?.note || `guide_queue:${source}`,
      guideSource: source,
    });
    selectedNames.add(key);
    return true;
  };

  for (const name of GUIDE_ORDER) addPending(name, "primary");
  for (const name of FALLBACK_ORDER) addPending(name, "fallback");

  const queue = selected.map((item, i) => normalizeItem(item, i + 1));
  const nextDoc = {
    updatedAt: nowIso(),
    refreshDelayMinutes: doc.refreshDelayMinutes ?? 5,
    guide: {
      source: path.join(SUBMISSIONS, "SUBMIT_GUIDE.md"),
      primaryOrder: GUIDE_ORDER,
      fallbackOrder: FALLBACK_ORDER,
      note: "Curated breadth-search queue. Re-run this script after new planned zip files are generated.",
    },
    queue,
  };
  writeJson(QUEUE_PATH, nextDoc);

  if (fs.existsSync(MANIFEST_SCRIPT)) {
    spawnSync(process.execPath, [MANIFEST_SCRIPT, "sync"], {
      cwd: ROOT,
      stdio: "ignore",
      windowsHide: true,
    });
  }

  console.log(`queue=${QUEUE_PATH}`);
  for (const item of queue) {
    console.log(`${String(item.index).padStart(2, "0")} ${String(item.status).padEnd(16)} ${item.name}`);
  }

  const missingPrimary = GUIDE_ORDER.filter((name) => !fs.existsSync(path.join(SUBMISSIONS, name)));
  if (missingPrimary.length) {
    console.log("missing_primary:");
    for (const name of missingPrimary) console.log(`  ${name}`);
  }
}

main();
