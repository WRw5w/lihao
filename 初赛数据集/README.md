# Robust CLIP ViT-B/32 Fine-Tuning

Competition-compliant single-model pipeline for fine-grained image
classification with noisy labels.

## Compliance

- Backbone: OpenAI CLIP ViT-B/32 only.
- Training data: official current-stage training set only.
- Test images are used for inference only.
- Final prediction uses one LoRA model and one inference flow.
- Filtering, pseudo-labeling, training, and prediction are fully scripted.

## Pipeline

1. Extract and cache frozen CLIP features.
2. Create the validation split before all label-driven processing.
3. Compute kNN agreement using the training partition as the only gallery.
4. Train a frozen-feature teacher on reliable training samples.
5. Train one LoRA model with continuous reliability weights and teacher-kNN
   consensus pseudo-labels.
6. Train the selected recipe on all training data and predict from `full.pt`.

## Commands

```bash
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main.py
python finetune_lora.py --epochs 15 --lora-blocks 12 --crop-min-scale 0.8
python finetune_lora.py --epochs 15 --resume outputs/lora/last.pt
python finetune_lora.py --full --epochs 15 --lora-blocks 12
python finetune_lora.py --predict --checkpoint full --output-csv pred_results.csv
python check_submission.py
python -m unittest discover -s tests -v
```

Do not run `pip install PyTorch`: the package named `PyTorch` is only an error
placeholder. The real package is `torch`. On an NVIDIA training machine, install
`requirements-cuda.txt` first; a plain PyPI install may select a CPU-only build.

## Important Outputs

- `outputs/cache/`: frozen feature caches with extraction metadata.
- `outputs/lora/best.pt`: best strict-validation checkpoint.
- `outputs/lora/last.pt`: latest strict-validation checkpoint.
- `outputs/lora/full.pt`: final checkpoint trained on all training data.
- `outputs/lora/history.json`: epoch metrics.
- `pred_results.csv` and `pred_results.zip`: submission artifacts.
