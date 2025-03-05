#!/bin/bash
#PJM -L jobenv=singularity
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -j
#PJM -o ./job_outputs/output_%j.out

cd ~/delphi

sparsity_ratio=0.0005

uv run -m delphi \
  --model meta-llama/Llama-3.2-1B \
  --sparse_model meta-llama/Llama-3.2-1B \
  --sparse_model_type mlp \
  --explainer_model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --explainer_model_max_len 8192 \
  --hookpoints layers.5.mlp \
  --name mlp_interpretation_$sparsity_ratio \
  --overwrite cache scores \
  --max_latents 50 \
  --filter_bos True \
  --mlp_activation_threshold 0.3 \
  --sparsity_ratio $sparsity_ratio \
  --n_tokens 10_000_000 \
  --n_examples_test 50 \
  --n_non_activating 50 \
  --batch_size 32