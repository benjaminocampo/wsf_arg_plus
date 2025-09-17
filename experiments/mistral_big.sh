#!/bin/bash
#SBATCH --job-name=mixtral-8x22B-big-checkworthy
#SBATCH --partition=gpu_h100
#SBATCH --time=00-08:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=320G
#SBATCH --output=./out/mixtral-8x22B-big-checkworthy.out

source ./bin/activate
python predict_checkworthy.py \
    llm=mistral_big \
    input.data_path=../data/wsf_annotations_misinformation.csv \
    input.data_size=0.1 \
    input.run_name=mixtral-8x22B-big-checkworthy \
    input.output_dir=./out \
    input.uri_path=./out