# auto-opt: c448_rank32

- config: `--img-size 448 --batch-size 64 --lora-rank 32 --lora-alpha 64 --epochs 8`
- best: 0.9136|ep6|noisy=0.7605|low=0.4863|high=0.9942
- baseline to beat: c448_drecall mid_03_06=0.9184 (> C448 0.9091)
