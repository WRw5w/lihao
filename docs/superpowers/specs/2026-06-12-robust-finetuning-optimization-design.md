# Robust Fine-Tuning Optimization Design

## Goal

Build a competition-compliant, reproducible, single-model CLIP ViT-B/32 pipeline
that improves noisy-label robustness without leaking validation labels.

## Competition Boundaries

- Use only OpenAI CLIP ViT-B/32 as the visual backbone.
- Use only the official dataset for the current competition stage.
- Never use test images for training, self-supervision, filtering, or tuning.
- Produce predictions from one model and one inference flow. Flip TTA is allowed
  because it is one model, while checkpoint/model ensembling is forbidden.
- Make filtering, training, validation, checkpoint selection, and prediction
  reproducible from code.

## Architecture

The frozen CLIP feature cache remains the basis for fast noise analysis. For a
validation run, the split is created before all label-driven operations. kNN
agreement, teacher training, reliability scores, and pseudo-labels are computed
from the training partition only. Validation samples query the training gallery
only and never initialize or supervise the model.

The LoRA training set uses a hybrid reliability policy:

- Keep the per-class top fraction by kNN agreement, with a high-agreement floor.
- Give kept samples a continuous reliability weight based on kNN agreement,
  teacher confidence for the observed label, and teacher prediction margin.
- Recover dropped samples only when the teacher is confident and its predicted
  class agrees with the kNN majority class.

The final submission flow trains the same single-model recipe on all training
data and saves `full.pt`. Prediction requires an explicit checkpoint policy and
does not silently prefer a stale validation checkpoint.

## Components

- `robust_utils.py`: pure decision helpers for split validation, reliability
  weighting, pseudo-label eligibility, and checkpoint selection.
- `finetune_lora.py`: strict split-first target preparation, configurable LoRA
  depth, robust training controls, explicit checkpoint prediction.
- `exp_head.py`: reusable kNN statistics and fair continuation controls.
- `tests/`: lightweight regression tests for leakage guards, weighting,
  checkpoint selection, and submission rules.

## Validation

Primary selection uses the strict held-out mid-agreement band. Secondary
metrics include noisy-label accuracy, high-agreement accuracy, per-class
accuracy, and retained/pseudo-labelled sample counts. Reported historical
leaky LoRA results must be marked as non-comparable.

## Error Handling And Reproducibility

- Refuse invalid split overlaps and empty training partitions.
- Refuse unsupported checkpoint names or missing explicit checkpoints.
- Store split indices, robust-target statistics, arguments, and checkpoint
  metadata.
- Use deterministic seeds and record the exact model name and competition
  compliance settings.

