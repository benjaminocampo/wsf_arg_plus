#!/bin/bash
#SBATCH --job-name=llama-70B-big-checkworthy
#SBATCH --partition=gpu_h100
#SBATCH --time=00-08:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --output=./out/llama-70B-big-checkworthy.out

source ../cs_hs_misinfo_env/bin/activate
python predict_checkworthy.py \
    llm=llama_big \
    input.data_path=../data/wsf_annotations_misinformation.csv \
    input.data_size=0.1 \
    input.run_name=llama-70B-big-checkworthy \
    input.output_dir=./out \
    input.uri_path=./out