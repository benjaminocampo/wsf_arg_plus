from tempfile import TemporaryDirectory
from pathlib import Path
from dotenv import load_dotenv
from utils import flatten_dict
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from vllm import LLM
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

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
        #max_model_len=cfg.llm.params.max_model_len,
        #max_num_batched_tokens=cfg.llm.params.max_num_batched_tokens,
        tensor_parallel_size=cfg.llm.params.tensor_parallel_size,
        dtype=cfg.llm.params.dtype,
        runner="pooling"
    )

    embeds = llm.embed(df["claim"].to_list())
    df["embed"] = embeds

    sss = StratifiedShuffleSplit(
        n_splits=5,
        train_size=0.30,
        test_size=0.70,
        random_state=0
    )

    fractions = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    N = len(df)

    models = {
            "lgr": LogisticRegression(max_iter=1000, random_state=0),
            "rforest": RandomForestClassifier(random_state=0),
            "svm": SVC(probability=True, random_state=0),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                random_state=0
            ),
            "xgb": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=0
            ),
        }

    y = df["claim_cw_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})

    def to_cat_labels(y):
        if y == 0:
            return "NFS"
        elif y == 1:
            return "UFS"
        elif y == 2:
            return "CFS"
        else:
            assert False

    for split_id, (train_idx, test_idx) in enumerate(sss.split(df, y)):
        # single shuffle -> nested subsets
        rng = np.random.RandomState(split_id)
        order = rng.permutation(len(train_idx))
        
        # We declare test data, training will be done after
        y_test = y.iloc[test_idx]
        X = df["embed"]

        X_pool = X.iloc[train_idx] 
        y_pool = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        X_pool = X_pool.iloc[order].reset_index(drop=True)
        y_pool = y_pool.iloc[order].reset_index(drop=True)
        
        for frac in fractions[1:]:
            n_samples = int(frac * N)

            X_sub = X_pool.iloc[:n_samples]
            y_sub = y_pool.iloc[:n_samples]

            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(X_sub, y_sub)
                preds = clf.predict(X_test)
                predict_proba = clf.predict_proba(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
                kappa = cohen_kappa_score(y_test, preds, weights="linear", labels=[0, 1, 2])

                res = {
                    "split": split_id,
                    "llm": cfg.llm.name,
                    "model": model_name,
                    "train_frac": frac,
                    "n_train": n_samples,
                    "p": p,
                    "r": r,
                    "f1": f1,
                    "kappa": kappa,
                    "y_test": y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                    "pred": pd.Series([to_cat_labels(p) for p in preds]),
                    "predict_proba": predict_proba
                }
                mlflow.log_metrics(res)


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
