from tempfile import TemporaryDirectory
from pathlib import Path
from dotenv import load_dotenv
from utils import flatten_dict
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from vllm import LLM

import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow
import os
import numpy as np

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
        tensor_parallel_size=cfg.llm.params.tensor_parallel_size,
        dtype=cfg.llm.params.dtype,
        runner="pooling"
    )

    outputs = llm.embed(df["claim"].to_list())
    embeds = [o.outputs.embedding for o in outputs]

    np.save(f"{cfg.experiment.path_embedding}/{cfg.input.run_name}.npy", embeds)


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
