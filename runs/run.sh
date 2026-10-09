#!/bin/bash

#SBATCH --job-name=ExampleRunName
#SBATCH --partition=root
#SBATCH --nodes=1
#SBATCH --qos intermediate
#SBATCH --mem 32G
#SBATCH --gres=gpu:1 # request a H200 GPU
#SBATCH --time=02:00:00
#SBATCH --output=/scratch_root/SHORTCODE/logs/%x-%j.out
#SBATCH --error=/scratch_root/SHORTCODE/logs/%x-%j.err
#SBATCH -c 8
#SBATCH -D /scratch_root/SHORTCODE/scd-detection

module purge

source /scratch_root/SHORTCODE/miniconda3/etc/profile.d/conda.sh
conda activate /scratch_root/SHORTCODE/miniconda3/envs/scd_env

export TORCH_HOME=/scratch_root/SHORTCODE/.cache/torch
mkdir -p "$TORCH_HOME"

echo "python: $(which python)"
which python
which gdalwarp
nvidia-smi --query-gpu=name --format=csv,noheader
 
python -u scripts/predict.py \
    --dem_path tiles \
    --model_state_path models/MaskRCNN_NoUK.pt \
    --veto_model_state_path models/DEMFineTunedUK.pt \
    --resolutions 1 3 5 \
    --device cuda \
    --run_name ExampleRunName \
    --score_threshold 0.75 \
    --veto_threshold 0.80 \
    --native_res 1.0 \


