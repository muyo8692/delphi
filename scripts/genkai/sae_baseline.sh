#!/bin/bash
#PJM -L jobenv=singularity
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -j
#PJM -o ./job_outputs/output_%j.out

cd ~/delphi

uv run -m delphi \
  --model meta-llama/Llama-3.2-1B \
  --sparse_model EleutherAI/sae-Llama-3.2-1B-131k \
  --sparse_model_type sae \
  --explainer_model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --explainer_model_max_len 8192 \
  --hookpoints layers.5 \
  --name transcoder_interpretation \
  --overwrite cache scores \
  --max_latents 100 \
  --filter_bos True \
  --n_tokens 100000 \
  --batch_size 16