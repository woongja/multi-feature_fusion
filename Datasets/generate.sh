#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate dataset

python generate_audio.py

conda deactivate

conda activate auto-tune

python generate_audio_autotune.py

conda deactivate

