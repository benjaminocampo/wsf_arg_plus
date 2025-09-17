# Misinformation in Hate Speech Detection

This repository has the code to run the paper (TODO: Decide title of the paper).

## Installation

Create a python environment on your machine. You can use the environment manager
of preference. For our experiments we used `venv`. Then proceed to install the
requirements of the project.

```bash
python -m venv cs_hs_misinfo_env
source cs_hs_misinfo_env/bin/activate
pip install -r requirements.txt
```

You must create as well a `.env` file that contains your READ token to
huggingface that allows you access to the following models
`meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`mistralai/Mixtral-8x22B-Instruct-v0.1`, `mistralai/Mistral-7B-Instruct-v0.3`.
In order to gain access to these models you can request it through huggingface
and fill in their corresponding access forms. In general, they have a quick
response time. The file should contain the following environment variable:

```
HF_TOKEN=<your-token>
```

## Run Experiments

The `experiments` directory contains the runs carried out in this paper. In
particular contains a main script called `predict_checkworthy.py` that runs an
specific llm with the parameters in `conf`. Command line arguments are handled
through Hydra (https://hydra.cc/docs/intro/). Hydra requires yaml files to where
default parameters are specified. In order to one of the models locally, for
example, an small mistra version `mistralai/Mistral-7B-Instruct-v0.3` you can do
it through;

```bash
python predict_checkworthy.py \
    llm=mistral_small \
    input.data_path=../data/wsf_annotations_misinformation.csv \
    input.data_size=0.1 \
    input.run_name=mistral-7B-small-checkworthy \
    input.output_dir=./out \
    input.uri_path=./out
```

In this case `mistral_small` is the configuration file containing the parameters
used in our experiments for this model. Other models' configurations can be
found in `confg/llm`. `config.yaml` contains the default parameters for the
experiment that remain the same across all models.

Output files are handled with `mlflow`. The directory `out` registers each of
the runs results containing the logs of parameters, and output stats of the
script.

## Slurm

In case to be using Slurm as a job manager, each configuration has a bash script
with the configuration for each model. For example to execute mistral
`mistralai/Mistral-7B-Instruct-v0.3`:

```bash
sbatch mistral_small.sh
```