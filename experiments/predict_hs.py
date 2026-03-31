from tempfile import TemporaryDirectory
from pathlib import Path
from dotenv import load_dotenv
from utils import flatten_dict
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from sklearn.metrics import precision_recall_fscore_support

import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


replacements = {
    "CFS": "Check-worthy Factual",
    "UFS": "Unimportant Factual",
    "NFS": "Non-Factual"
}


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
    guided_decoding_params = GuidedDecodingParams(choice=cfg.experiment.output_labels)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
        max_tokens=cfg.llm.params.max_tokens,
    )

    logger.info("generate labels...")

    cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

    for col in cols:
        df[f"{col}_cw_{cfg.experiment.cw_quality}_repl"] = df[f"{col}_cw_{cfg.experiment.cw_quality}"].replace(replacements)

    concat_texts = []
    for _, row in df.iterrows():
        concat_text = ""
        for col in cols:
            # All row[col] are annotated so this check can be done as well on
            # the full text. However, we prefer to do it on the label directly
            if pd.isna(row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]):
                continue

            if row["is_argument"] == "yes":
                if cfg.experiment.use_checkworthiness:
                    concat_text += (
                        f'[{df[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}] ' + # Start of the wrapping
                        f'{row[col].strip(".")} ' + # Claim
                        f'[/{row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}]. ' # end of the wrapping (space for next claim)
                    )
                else:
                    concat_text += f'{row[col].strip(".")}. '
            else:
                if col == "conclusion":
                    if cfg.experiment.use_checkworthiness:
                        concat_text += (
                            'Therefore, ' +
                            f'[{row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}] ' +
                            f'{row[col].strip(".")} ' +
                            f'[/{row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}].'
                        )
                    else:
                        concat_text += f'Therefore, {row[col].strip(".")}.'
                else:
                    if cfg.experiment.use_checkworthiness:
                        concat_text += (
                            f'[{row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}] ' +
                            f'{row[col].strip(".")} ' +
                            f'[/{row[f"{col}_cw_{cfg.experiment.cw_quality}_repl"]}]. '
                        )
                    else:
                        concat_text += f'{row[col].strip(".")}. '
        concat_texts.append(concat_text)

    df["new_concat"] = concat_texts

    df["new_concat_prompt"] = df["new_concat"].apply(
        lambda t: [
            {"role": "system", "content": cfg.experiment.system},
            {"role": "user", "content": cfg.experiment.user.format(input_text=t)},
        ]
    )
    responses = llm.chat(
        messages=df["new_concat_prompt"].tolist(),
        sampling_params=sampling_params,
    )
    df["concat_pred_hate"] = [r.outputs[0].text for r in responses]

    df["concat_pred_hate"] = df["concat_pred_hate"].replace({"hateful": 1, "non-hateful": 0})
    y_true = df["concat_hate"].astype(int)
    y_pred = df["concat_pred_hate"].astype(int)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    results = {
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
        "p_micro": p_micro,
        "r_micro": r_micro,
        "f1_micro": f1_micro,
        "p_weighted": p_weighted,
        "r_weighted": r_weighted,
        "f1_weighted": f1_weighted,
    }
    mlflow.log_metrics(results)
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
