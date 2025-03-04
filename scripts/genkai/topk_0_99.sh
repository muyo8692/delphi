#!/bin/bash
#PJM -L jobenv=singularity
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -j
#PJM -o ./job_outputs/output_%j.out

cd ~/delphi

uv run experiments/sanity_check/top_0_99_mlp_coder.py