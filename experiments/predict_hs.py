from tempfile import TemporaryDirectory
from pathlib import Path
from dotenv import load_dotenv
from utils import flatten_dict
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow
import os
import math

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Use LLMs to generate the labels for the Check Worthyness task. That is,
    verify if a message can be considered relevant to be fact-checked.
    """
    logger.info("Command-line Arguments:")
    logger.info(f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}")

    df = pd.read_csv(cfg.input.data_path)

    assert 0 <= cfg.input.data_size <= 1

    df = df.head(int(len(df) * cfg.input.data_size))

    llm = LLM(
        model=cfg.llm.params.model,
        max_model_len=cfg.llm.params.max_model_len,
        max_num_batched_tokens=cfg.llm.params.max_num_batched_tokens,
        tensor_parallel_size=cfg.llm.params.tensor_parallel_size,
        dtype=cfg.llm.params.dtype,
    )
    guided_decoding_params = GuidedDecodingParams(json=cfg.experiment.output_schema)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
        max_tokens=cfg.llm.params.max_tokens,
    )

    logger.info("generate labels...")

    arg_cols = ["premise0", "premise1", "premise2", "conclusion"]
    cw_cols = ["premise0_final_cw", "premise1_final_cw", "premise2_final_cw", "conclusion_final_cw"]

    df["premise0_final_cw"] = df["annA_premise0"]
    df["premise1_final_cw"] = df["annA_premise1"]
    df["premise2_final_cw"] = df["annA_premise2"]
    df["conclusion_final_cw"] = df["annA_conclusion"]

    df.loc[~df["annC_premise0_cw_final"].isna(), "premise0_final_cw"] = df.loc[~df["annC_premise0_cw_final"].isna(), "annC_premise0_cw_final"]
    df.loc[~df["annC_premise1_cw_final"].isna(), "premise1_final_cw"] = df.loc[~df["annC_premise1_cw_final"].isna(), "annC_premise1_cw_final"]
    df.loc[~df["annC_premise2_cw_final"].isna(), "premise2_final_cw"] = df.loc[~df["annC_premise2_cw_final"].isna(), "annC_premise2_cw_final"]
    df.loc[~df["annC_conclusion_cw_final"].isna(), "conclusion_final_cw"] = df.loc[~df["annC_conclusion_cw_final"].isna(), "annC_conclusion_cw_final"]

    repl = {
        "NFS": "Non-Factual",
        "UFS": "Unimportant Factual",
        "CFS": "Check-worthy Factual"
    }
    df["premise0_final_cw"] = df["premise0_final_cw"].replace(repl)
    df["premise1_final_cw"] = df["premise1_final_cw"].replace(repl)
    df["premise2_final_cw"] = df["premise2_final_cw"].replace(repl)
    df["conclusion_final_cw"] = df["conclusion_final_cw"].replace(repl)
    arg_comps = []

    if cfg.experiment.use_checkworthiness:
        for _, row in df.iterrows():
            ss = ""
            for c in arg_cols:
                if ((not isinstance(row[c], str)) and (math.isnan(row[c]))):
                    continue
                ss = ss + f"{c} [{row[f'{c}_final_cw']}]: {row[c]}\n"
            arg_comps.append(ss)
    else:
        for _, row in df[arg_cols].iterrows():
            ss = ""
            for c in arg_cols:
                if ((not isinstance(row[c], str)) and (math.isnan(row[c]))):
                    continue
                ss = ss + f"{c}: {row[c]}\n"
            arg_comps.append(ss)

    df["arg_comps"] = arg_comps
    df["arg_comps_prompt"] = df["arg_comps"].apply(
        lambda t: [
            {"role": "system", "content": cfg.experiment.system},
            {"role": "user", "content": cfg.experiment.user.format(arg_comps=t)},
        ]
    )
    responses = llm.chat(
        messages=df["arg_comps_prompt"].tolist(),
        sampling_params=sampling_params,
        response_format={"type": "json_schema", "json_schema": cfg.experiment.output_schema},
    )
    df["arg_comps_pred"] = [r.outputs[0].text for r in responses]

    out_file = f"{cfg.input.run_name}_llm_pred.csv"
    df.to_csv(out_file, index=False)
    mlflow.log_artifact(out_file)
    os.remove(out_file)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    if cfg.input.uri_path is not None:
        mlflow.set_tracking_uri(cfg.input.uri_path)
        assert cfg.input.uri_path == mlflow.get_tracking_uri()

    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    mlflow.set_experiment(cfg.experiment.experiment_name)
    mlflow.set_experiment_tag(
        "mlflow.note.content", cfg.experiment.experiment_description
    )

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Log params expects a flatten dictionary, since the configuration has
        # nested configurations (e.g. train.model), we need to use flatten_dict
        # in order to transform it into something that can be easilty logged by
        # MLFlow.
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run)


if __name__ == "__main__":
    main()
