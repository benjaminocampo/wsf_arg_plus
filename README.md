# When Hate Meets Facts: LLMs-in-the-Loop for Checkworthiness Detection in Hate Speech

This repository contains the dataset WSF-ARG+ and the source code for each of the
experiments carried out in the paper entitled "When Hate Meets Facts:
LLMs-in-the-Loop for Checkworthiness Detection in Hate Speech". We also released
the guidelines and steps necessary to apply our LLM-in-the-loop framework on
WSF-ARG+ and on other datasets.

## Index

- [Installation](#installation)
- [Reproducibility of Experiments](#reproducibility-of-experiments)
- [Results and logging](#results-and-logging)
- [WSF-ARG+ Dataset](#wsf-arg-dataset)
- [How-To Guidelines for Running LLM-in-the-Loop](#how-to-guidelines-for-running-llm-in-the-loop)

## Installation

Create a python environment on your machine. You can use the environment manager
of preference. For our experiments we used `venv`. Then proceed to install the
requirements of the project.

```bash
python -m venv wsf_arg_plus_env
source wsf_arg_plus_env/bin/activate
pip install -r requirements.txt
```

You must create as well a `.env` file that contains your READ token to
huggingface that allows you access to the following models:

- Mistral-7B: `mistralai/Mistral-7B-Instruct-v0.3`
- Llama-8B: `meta-llama/Llama-3.1-8B-Instruct`
- Olmo2-7B: `allenai/OLMo-2-1124-7B-Instruct`
- Qwen2.5-7B: `Qwen/Qwen2.5-7B-Instruct`
- Command-r-7B: `CohereLabs/c4ai-command-r7b-12-2024`
- Mixtral-8x7B: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Mistral-22B: `mistralai/Mistral-Small-Instruct-2409`
- Olmo2-32B: `allenai/OLMo-2-0325-32B-Instruct`
- Mixtral-8x22B: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Llama-70B: `meta-llama/Llama-3.3-70B-Instruct`
- Qwen2.5-72B: `Qwen/Qwen2.5-72B-Instruct`
- Command-r-104B: `CohereLabs/c4ai-command-r-plus-08-2024`

In order to gain access to these models you can request it through huggingface
and fill in their corresponding access forms. In general, they have a quick
response time.

The `.env` file should contain the following environment variable:

```
HF_TOKEN=<your-token>
```

## Reproducibility of Experiments

The `experiments/` directory contains all the runs conducted for this paper. The
directory `analysis/` contains all the scripts that are necessary to obtain each
of the tables and results presented in the paper. Their numeration indicates its
associated table. For example the script `01_descriptive_statistics.py`
calculates the results presented in Table 1 of the paper,
`02_cw_percent_agreement.py` calculates the results presented in Table 2 of the
paper, and so on. The script `06_D_hs_detection_with_cw.py` for convenience it
contains the results of Tables 6 and D of the paper. The script
`03.1_judge_decisions.py` contains some of the results that have not an
corresponding table but were related to Table 3 and were properly described in
the paper. These scripts are self contained and can be run directly using
`python percent cells`. The script `00_data_preprocessing.py` contains the steps
that we carried out to preprocess and clean the data that was released and each
of the formats available in `data/`. Scripts in `analysis/` use the preprocessed
data to obtain the results. We encourage researchers to use the released data to
be used in future findings.

To predict on check-worthiness detection we used the main entry script
`predict_cw.py` which executes a chosen LLM using parameters defined in the
`conf/` directory. `conf/` is a directory that contains all the configuration
files that are used to run our experiments, and are organized in `experiment/`
and `llm/`. `experiment/` contains configuration files related to the task such
as the prompt defining check-worthiness detection, the output labels, the
prompting strategy (zero or one shot), the examples used for those settings.
`llm/` contains LLM configuration files that were used in the paper, they
contain the exact version used, the hardware resources claimed to properly run
that LLM, and extra LLM related parameters for prediction. The files for the
check-worthiness detection task are `checkworthiness_zero_shot.yaml` and
`check-worthiness_one_shot.yaml`. There's an extra file in `conf/` that is
`config.yaml`. This file contains configuration that are shared and are kept the
same across all runs.

To predict on hate speech detection using check-worthiness labels we used the
entry script `predict_hs.py` that uses the LLM configurations as the one of the
check-worthiness detection task, but the task experiment changes. configuration
files regarding this task are `experiment/detect_hs_base_noarg.yaml` for the
baseline hate speech detection (without check-worthiness labels), and
`experiment/detect_hs_with_cw_noarg.yaml` for the detection with
check-worthiness labels.

Command-line arguments are managed with
**[Hydra](https://hydra.cc/docs/intro/)**, a configuration framework. Hydra uses
`.yaml` files to specify default parameters and organize different experiment
settings.

To launch an experiment with a specific LLM configuration, use the
**`run_job.sh`** script.

This Bash script:  
- Takes as input three parameters: the script to launch the experiment that can
  be either `predict_hs.py` or `predict_cw.py` and the two following `.yaml`
  configuration files.
- Submits the run to **Slurm**, a job manager that allocates the necessary
  computing resources.  
- Reads the `slurm_params` section from each LLM config file to properly
  configure the job submission.  

If Slurm is **not available**, the script will attempt to run the experiment
locally. However, local execution is not guaranteed to succeed, since some
experiments may require specific hardware (e.g., GPUs). Small models required,
one GPU A100, medium size one to two GPUs H100, and big models were run with
four GPUs sometimes A100 or H100.

For example if we want to predict check-worthiness on WSF-ARG+ with
`Mistral-7B`, we can give `run_job.sh` three parameters, one for the
`predict_cw.py` script, one for the experiment configuration (mainly describing
the task parameters) and one for the llm configuration (mainly describing the
name, parameters, and hardware to be used for that LLM).

```bash
cd experiments/
bash run_job.sh predict_cw.py conf/experiment/checkworthiness_zero_shot.yaml conf/llm/mistral-7B-small.yaml
```

## Results and Logging

All experiment outputs are tracked using `MLflow`. The results directory stores:

- Run logs
- Parameters used
- Output statistics generated by the scripts

The `predict_cw.py` and `predict_hs.py` scripts use MLFlow to track a run. This
recording is later saved in a directory called `mlruns`.

Those runs can be later read using the MLFlow interface by running on the same
directory `mlruns` is stored:

```bash
mlflow ui
```

There you can visualize the recorded generations of your run and the associated
metrics.

## WSF-ARG+ Dataset

The dataset is preprocessed from `wsf_arg_plus_raw.csv` using the script in `experiments/analysis/00_data_preprocessing.py`. We organized the data into two tabular formats:

1. Message-level format:
In this format, each row corresponds to an entire message, which can be either hate speech or non-hate speech. Each message is divided into one or more claims. If the message is argumentative, its premises and conclusion are explicitly annotated; otherwise, it is stored as one or more claims. Consequently, each row represents a full message, with columns corresponding to the individual claims. Additional information includes:

- The overall hatefulness of the message
- Whether the message is argumentative
- Annotations per claim (hatefulness and check-worthiness labels)
- Existing annotations from WSF-ARG

You can find the dataset in the directory `data/`. The file
`wsf_arg_plus_per_message.csv` contains the aggreggated labels through majority
voting. It has both gold (obtained through LLM-in-the-loop) and platinum
(obtained through full human annotation) check-worthiness annotations.
`wsf_arg_plus_per_message_gold_disagg.csv` and
`wsf_arg_plus_per_message_platinum_disagg.csv` contain the annotations
disaggregated given by all annotators for both gold and platinum. We also
indicate which claims required to be judged. We also released
`wsf_arg_plus_per_message_all_llms.csv` that contains all the predictions carried
out in the check-worthiness detection task for all configurations (per LLM and
prompt). We release the three runs per configuration and the majority voting
label. Consider that we have 12 LLMs tested, and 2 prompt strategies being in
total 12x2 = 24 configurations. Each configuration is run three times meaning a
total of 24x3 = 96 runs. Therefore, the `.csv` file contains 96 columns with
each of these runs and 12 other columns for the majority voting label. As a
helper of how to handle these datasets, the scripts in `experiments/analysis/`
can be of inspiration to see how to work with this amount of columns properly.

2. Claim-level format: This format transforms the message-level table into one
where each row represents a single claim. Each claim may originate from a full
message. For each claim, we record:

- The message it comes from
- Whether the claim itself is hateful
- Whether it comes from a hate speech or non-hate speech message
- Check-worthiness annotations provided by the LLM-in-the-loop
- Annotations from all human annotators
- Annotations from each model included in our experimental study

You can find the dataset in the directory `data/`. Similarly to the
message-level format, we have:
- `wsf_arg_plus_per_claim.csv` as the general dataset organized in claim-level
  format.
- `wsf_arg_plus_per_claim_gold_disagg.csv` and
  `wsf_arg_plus_per_claim_platinum_disagg.csv` for the disaggregated gold and
  platinum labels.
- `wsf_arg_plus_per_claim_all_llms.csv` that contains the disaggregated and
  aggregated annotations carried out by all our 24 configuration (12 LLMs and 2
  prompting strategies).

## How-To Guidelines for Running LLM-in-the-Loop

As a resource for the NLP community, we release step-by-step instructions to run the proposed LLM-in-the-loop framework not only on WSF-ARG+ but also on other datasets. As one of our main contributions, we aim for LLM-in-the-loop to be easily reproducible, serving as a future direction to be tested across several datasets and domains in order to achieve high-quality labeling through an LLM-as-annotator strategy. We describe the LLM-in-the-loop framework for a task that has $N=3$ labels. As future perspectives we aim to extend the framework for tasks with $N=2$ and $N>3$.

The first step is to have a dataset $D$ that specifies a detection task with $N=3$ labels. For a task with three classes, we may require at least four annotators in a full human annotation setting. Three annotators independently annotate the dataset, and the remaining one acts as a judge for instances where the three first annotators disagree.

In this setting, we aim for reducing the human effort from four annotators to two. We suggest annotating the entire dataset $D$ with one human annotator, one chosen LLM, and adjudicating disagreements with a human judge. This LLM should receive, as instructions, the definition of the task and the classes, as well as prompting strategies such as zero- or one-shot prompting (this can also be extended to more than one shot, i.e., few-shot prompting).

We recommend running our approach using several LLMs, in particular those described in our paper, which establish a good test bed of models and provide an indication across small, medium, and large models, as well as across different LLM families. We suggest running each configuration three times to test prediction variability. Majority voting is used to decide the final labeling; in cases where there is no clear majority-voted class, that is, when the three runs produce three different labels, a judge is required.

- **Step 1**: One annotator labels the entire dataset.
- **Step 2**: Several LLMs annotate the entire dataset using different prompting strategies (zero-, one-, and few-shot). Configurations depend on model family and size. A possible pool of LLMs configurations can be the one described in our paper and in the `installation` section of this README. Run each configuration three times to measure prediction variability. The final labeling is decided through majority voting. If there is no unique majority-voted label, a judge intervenes.
- **Step 3**: Calculate alignment metrics between the LLMs and the human annotator, as well as prediction variability across different configuration runs.

We then select the most "suitable" configuration, where suitability depends on percent agreement and inter-annotator agreement (IAA) with the human annotator, low prediction variability, and high IAA among LLMs.

Once the most "suitable" configuration for the dataset $D$ has been selected, we proceed to resolve disagreements between this configuration and the human annotator by asking another human annotator to judge disagreement cases, as well as unstable predictions produced by the LLM. The judge should be blind to the source of the previous annotations. This can be achieved by presenting the labels in random order with anonymous sources, or by completely hiding their origin. Afterwards, the decisions should be computed and reported, measuring agreement with the human annotator, the LLM, or both (if this setting is possible).

- **Step 4**: Choose the most "suitable" configuration for dataset $D$ based on percent agreement, IAA with humans and other LLMs, prediction variability, and prompting strategy.
- **Step 5**: Ask a human judge to annotate instances where there is disagreement between the human annotator and the chosen configuration. The judge should be blind to the source of the previous labels. A clear report of the judge’s preferences should be carried out.

Finally, if available, these labels should be compared with those obtained from a full human annotation setting to assess the extent to which quality is maintained or lost.

- **Step 6**: If available, compare LLM-in-the-loop annotations with full human annotations to measure and report annotation quality.

As anticipated in the previous sections of this README, `predict_cw.py` is a use case example of Step 2, defining a script for running several configurations and generating annotations with LLMs. The analysis carried out in Steps 3, 4, 5, and 6 can be found in the directory `experiments/analysis/`, where we provide python percent files (similar to notebooks) in which we performed each of the remaining steps.

More specifically, Step 3 is implemented in the files `02_cw_percent_agreement.py`, `03_iaa_olmo2-32B_vs_rest.py`, and `B_pred_variability.py`. Step 4 is carried out by analyzing the results of Step 3. The analysis for Step 5 can be found in `03.1_judge_decisions.py`. Finally, Step 6 is implemented in `04_data_distribution.py` and `A_iaa_full_human.py`