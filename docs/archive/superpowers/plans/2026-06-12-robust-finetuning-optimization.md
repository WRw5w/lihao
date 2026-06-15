# Robust Fine-Tuning Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a strict-validation, reproducible, single-model CLIP ViT-B/32 LoRA pipeline with stronger noisy-label training.

**Architecture:** Isolate pure robust-training decisions in `robust_utils.py`, then use them from `finetune_lora.py` after splitting data before any label-driven preparation. Preserve the existing frozen-feature workflow while adding continuous reliability weighting, consensus pseudo-labels, explicit checkpoints, and reproducibility metadata.

**Tech Stack:** Python, PyTorch, torchvision, timm, unittest.

---

### Task 1: Pure Robust Decision Helpers

**Files:**
- Create: `robust_utils.py`
- Create: `tests/test_robust_utils.py`

- [ ] Write failing tests for split overlap rejection, continuous weights,
  consensus pseudo-label masks, and checkpoint selection.
- [ ] Run `python -m unittest tests.test_robust_utils -v` and confirm RED.
- [ ] Implement minimal helpers.
- [ ] Run the tests and confirm GREEN.

### Task 2: Strict Split-First LoRA Preparation

**Files:**
- Modify: `finetune_lora.py`
- Test: `tests/test_lora_contract.py`

- [ ] Write a source-contract test proving split indices are passed into target
  preparation and validation indices are excluded from teacher preparation.
- [ ] Run the contract test and confirm RED.
- [ ] Refactor `prepare_targets` to accept train/validation indices and compute
  the training-gallery-only statistics.
- [ ] Run all tests and confirm GREEN.

### Task 3: Robust Weighted Targets

**Files:**
- Modify: `finetune_lora.py`
- Modify: `robust_utils.py`
- Test: `tests/test_robust_utils.py`

- [ ] Add failing tests for reliability-weight bounds and consensus recovery.
- [ ] Implement continuous kept-sample weights and consensus pseudo-labels.
- [ ] Save target statistics in checkpoints.
- [ ] Run all tests and confirm GREEN.

### Task 4: Single-Model Training And Checkpoint Safety

**Files:**
- Modify: `finetune_lora.py`
- Test: `tests/test_lora_contract.py`

- [ ] Add failing tests for explicit `best`, `last`, and `full` checkpoint
  policies and single-model metadata.
- [ ] Add configurable LoRA block depth, gradient clipping, early stopping,
  resumable checkpoints, and `full.pt`.
- [ ] Make prediction choose the requested checkpoint explicitly.
- [ ] Run all tests and confirm GREEN.

### Task 5: Fair Experiments And Documentation

**Files:**
- Modify: `exp_round2.py`
- Modify: `README.md`
- Modify: `EXPERIMENTS.md`

- [ ] Add an equal-step continuation control beside pseudo-label continuation.
- [ ] Document strict validation commands, full-training commands, and explicit
  prediction checkpoint commands.
- [ ] Mark historical leaky LoRA validation metrics as non-comparable.
- [ ] Run compile checks, unit tests, and submission validation.

