#!/bin/bash
#SBATCH --job-name=llama-7B-small-checkworthy
#SBATCH --partition=gpu_h100
#SBATCH --time=00-04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --output=./out/llama-7B-small-checkworthy.out

source ./bin/activate
python predict_checkworthy.py \
    llm=llama_small \
    input.data_path=../data/wsf_annotations_misinformation.csv \
    input.data_size=0.1 \
    input.run_name=llama-7B-small-checkworthy \
    input.output_dir=./out \
    input.uri_path=./out