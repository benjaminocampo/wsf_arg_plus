# %%
import pandas as pd
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
all = {
	"Mistral-7B": ["zero","one"],
	"Llama-8B": ["zero", "one"],
	"Olmo2-7B": ["zero", "one"],
	"Qwen2.5-7B": ["zero", "one"],
	"Command-r-7B": ["zero", "one"],
	"Mixtral-8x7B": ["zero", "one"],
	"Mistral-22B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
	"Mixtral-8x22B": ["zero", "one"],
	"Llama-70B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
	"Command-r-104B": ["zero", "one"],
}
# %%
all_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in all.items() for shot in shots]
# %%
for c in all_flatten:
    df[c] = pd.Categorical(df[c], categories=["NFS", "UFS", "CFS"])
# %%
X = pd.get_dummies(df[all_flatten]).astype(int)
y = df["claim_cw_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})
# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def kfoldres(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    models = {
        "lgr": LogisticRegression(max_iter=1000),
        "rforest": RandomForestClassifier(),
        "svm": SVC(),
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

    results = {}

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            p, r, f1, sup = precision_recall_fscore_support(y_test, preds, average="macro")
            results[f"kf={split_idx}_{model_name}"] = {}
            results[f"kf={split_idx}_{model_name}"]["p"] = p
            results[f"kf={split_idx}_{model_name}"]["r"] = r
            results[f"kf={split_idx}_{model_name}"]["f1"] = f1
    
    return results
# %%
results = kfoldres(X, y)
df_results = pd.DataFrame(results).T
# %%
df_results.round(3)
# %%
df_results = df_results.reset_index().rename(columns={"index": "kfold"})
# %%
df_results["model_name"] = df_results["kfold"].apply(lambda fold: fold.split("_")[1])
df_results["kfold"] = df_results["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_kfold_cw = (
    df_results
    .drop(columns=["kfold"])
    .groupby("model_name")
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_kfold_cw
# %%
# Combination with ISHate
# %%
import pandas as pd

splits = {'train': 'ishate_train.parquet.gzip', 'validation': 'ishate_dev.parquet.gzip', 'test': 'ishate_test.parquet.gzip'}
df_ishate_train = pd.read_parquet("hf://datasets/BenjaminOcampo/ISHate/" + splits["train"])
df_ishate_dev = pd.read_parquet("hf://datasets/BenjaminOcampo/ISHate/" + splits["validation"])
df_ishate_test = pd.read_parquet("hf://datasets/BenjaminOcampo/ISHate/" + splits["test"])
# %%
df_ishate = pd.concat([
    df_ishate_train[df_ishate_train["aug_method"] == "orig"].drop(columns=["aug_method"]),
    df_ishate_dev,
    df_ishate_test
    ])
# %%
df_ishate_wsf_hate = df_ishate[(df_ishate["source"] == "wsf") & 
                               (df_ishate["hateful_layer"] == "HS")]
# %%
df_wsf_arg_plus = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
# %%
df_wsf_arg_plus_implicit = df_wsf_arg_plus.merge(df_ishate_wsf_hate, left_on="file_id", right_on="message_id", how="left")
# %%
df_wsf_arg_plus_implicit.reset_index()
# %%
df["message_id"] = df["claim_idx"].apply(lambda sid: int(sid.split("_")[0]))
# %%
df_claim_implicit = df.merge(df_wsf_arg_plus_implicit.reset_index(), left_on="message_id", right_on="index", how="left")
# %%
def kfoldres_implicit(X, y, idx_implicit, idx_subtle):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    models = {
        "lgr": LogisticRegression(max_iter=1000),
        "rforest": RandomForestClassifier(),
        "svm": SVC(),
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

    results = {}

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            p, r, f1, sup = precision_recall_fscore_support(y_test, preds, average="macro")
            
            results[f"kf={split_idx}_{model_name}"] = {}
            results[f"kf={split_idx}_{model_name}"]["p"] = p
            results[f"kf={split_idx}_{model_name}"]["r"] = r
            results[f"kf={split_idx}_{model_name}"]["f1"] = f1

            results[f"kf={split_idx}_{model_name}"]["acc_implicit"] = (preds[idx_implicit] == y_test.iloc[idx_implicit]).sum() / len(idx_implicit)
            results[f"kf={split_idx}_{model_name}"]["acc_subtle"] = (preds[idx_subtle] == y_test.iloc[idx_subtle]).sum() / len(idx_subtle)
    
    return results
# %%
idx_subtle = df_claim_implicit[df_claim_implicit["subtlety_layer"] == "Subtle"].index 
# %%
idx_implicit = df_claim_implicit[df_claim_implicit["implicit_layer"] == "Implicit HS"].index
# %%
results_imp_subt = kfoldres_implicit(X, y, idx_implicit, idx_subtle)
# %%
df_wsf_arg_plus_implicit[df_wsf_arg_plus_implicit["implicit_layer"] == "Implicit HS"]
# %%
df_wsf_arg_plus_implicit[df_wsf_arg_plus_implicit["subtlety_layer"] == "Subtle"]
# %%
pd.crosstab(df_claim_implicit["implicit_layer"], df_claim_implicit["claim_cw_platinum"])
# %%
pd.crosstab(df_claim_implicit["subtlety_layer"], df_claim_implicit["claim_cw_platinum"])
# %%
df_wsf_arg_plus_implicit[df_wsf_arg_plus_implicit["subtlety_layer"] == "Subtle"].to_csv("wsf_arg_plus_subtle.csv", index=False)
# %%
# Kfold validation using little data
# %%
import pandas as pd

df_claim = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
df_platinum_disagg = pd.read_csv("../../data/wsf_arg_plus_per_claim_platinum_disagg.csv")
# %%
df = pd.concat([df_claim, df_platinum_disagg[["claim_cw_annA_platinum", "claim_cw_annB_platinum", "claim_cw_annC_platinum"]]], axis=1)
# %%
all = {
	"Mistral-7B": ["zero","one"],
	"Llama-8B": ["zero", "one"],
	"Olmo2-7B": ["zero", "one"],
	"Qwen2.5-7B": ["zero", "one"],
	"Command-r-7B": ["zero", "one"],
	"Mixtral-8x7B": ["zero", "one"],
	"Mistral-22B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
	"Mixtral-8x22B": ["zero", "one"],
	"Llama-70B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
	"Command-r-104B": ["zero", "one"],
}
small = {
	"Mistral-7B": ["zero","one"],
	"Llama-8B": ["zero", "one"],
	"Olmo2-7B": ["zero", "one"],
	"Qwen2.5-7B": ["zero", "one"],
	"Command-r-7B": ["zero", "one"]
}
medium = {
	"Mixtral-8x7B": ["zero", "one"],
	"Mistral-22B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
	"Mixtral-8x22B": ["zero", "one"],

}
large = {
	"Llama-70B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
	"Command-r-104B": ["zero", "one"],
}

mistral = {
	"Mistral-7B": ["zero","one"],
	"Mistral-22B": ["zero", "one"],
}
mixtral = {
	"Mixtral-8x22B": ["zero", "one"],
	"Mixtral-8x7B": ["zero", "one"],
}
llama = {
    "Llama-8B": ["zero", "one"],
	"Llama-70B": ["zero", "one"],
}
olmo = {
    "Olmo2-7B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
}
qwen = {
    "Qwen2.5-7B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
}
commandr = {
    "Command-r-7B": ["zero", "one"],
    "Command-r-104B": ["zero", "one"],
}

all_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in all.items() for shot in shots]
small_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in small.items() for shot in shots]
medium_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in medium.items() for shot in shots]
large_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in large.items() for shot in shots]
mistral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mistral.items() for shot in shots]
mixtral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mixtral.items() for shot in shots]
llama_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in llama.items() for shot in shots]
olmo_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in olmo.items() for shot in shots]
qwen_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in qwen.items() for shot in shots]
commandr_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in commandr.items() for shot in shots]
# %%
from collections import Counter

def majority_vote(row, cols, group_name):
    values = row[cols].tolist()
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({group_name: most_common_value, f"{group_name}_agreement_count": count})
    else:
        return pd.Series({group_name: "All unequal", f"{group_name}_agreement_count": 1})
# %% [markdown]
# Comparison of Majority Voting vs Ensemble methods with training data
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone
import numpy as np

sss = StratifiedShuffleSplit(
    n_splits=5,
    train_size=0.30,
    test_size=0.70,
    random_state=0
)

fractions = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

results = []

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

results = []

mv_strats = {
    "all": all_flatten,
    "small": small_flatten,
    "medium": medium_flatten,
    "large": large_flatten,
    "mistral": mistral_flatten,
    "mixtral": mixtral_flatten,
    "llama": llama_flatten,
    "olmo": olmo_flatten,
    "qwen": qwen_flatten,
    "commandr": commandr_flatten,
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

    # Results of LLM-in-the-loop across the folds
    kappa_llm_loop = cohen_kappa_score(y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}), df.iloc[test_idx]["claim_cw_gold"], weights="linear", labels=["NFS", "UFS", "CFS"])
    p_macro_llm_loop, r_macro_llm_loop, f1_macro_llm_loop, _ = precision_recall_fscore_support(y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                                                                                               df.iloc[test_idx]["claim_cw_gold"],
                                                                                               average='macro',
                                                                                               labels=["NFS", "UFS", "CFS"])
    results.append({
        "split": split_id,
        "model": "llm_loop",
        "encoding": "",
        "encoding_annotator": "",
        "train_frac": 0,
        "n_train": None,
        "p": p_macro_llm_loop,
        "r": r_macro_llm_loop,
        "f1": f1_macro_llm_loop,
        "kappa": kappa_llm_loop,
        "y_test": y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
        "pred": df.iloc[test_idx]["claim_cw_gold"],
        "annA": df.iloc[test_idx]["claim_cw_annA_platinum"],
        "annB": df.iloc[test_idx]["claim_cw_annB_platinum"],
        "annC": df.iloc[test_idx]["claim_cw_annC_platinum"],
        "predict_proba": None
    })

    # No training data here, use majority voting and test on the test kfold split
    for mv_strat, flatten in mv_strats.items():
        for ann in ["only_llms", "annA", "annB", "annC"]:
            if ann == "only_llms":
                mv = df.iloc[test_idx].apply(lambda row: majority_vote(row, flatten, mv_strat), axis=1)
            else:
                mv = df.iloc[test_idx].apply(lambda row: majority_vote(row, flatten + [f"claim_cw_{ann}_platinum"], mv_strat), axis=1)
            y_pred_mv = mv[mv_strat]
            p_macro_mv, r_macro_mv, f1_macro_mv, _ = precision_recall_fscore_support(y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                                                                                    y_pred_mv,
                                                                                    average='macro',
                                                                                    labels=["NFS", "UFS", "CFS"])

            kappa_mv = cohen_kappa_score(y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                                         y_pred_mv, weights="linear", labels=["NFS", "UFS", "CFS"])
            results.append({
                "split": split_id,
                "model": "mv",
                "encoding": mv_strat,
                "encoding_annotator": ann,
                "train_frac": 0,
                "n_train": None,
                "p": p_macro_mv,
                "r": r_macro_mv,
                "f1": f1_macro_mv,
                "kappa": kappa_mv,
                "y_test": y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                "pred": y_pred_mv,
                "annA": df.iloc[test_idx]["claim_cw_annA_platinum"],
                "annB": df.iloc[test_idx]["claim_cw_annB_platinum"],
                "annC": df.iloc[test_idx]["claim_cw_annC_platinum"],
                "predict_proba": None
            })

    # Now train small models
    for mv_strat, flatten in mv_strats.items():
        for ann in ["only_llms", "only_annA", "only_annB", "only_annC", "annA", "annB", "annC"]:
            if ann == "only_llms":
                # Encode of only llms
                X = pd.get_dummies(df[flatten]).astype(int)
            elif ann.startswith("only_"):
                # Encode of only ann{A, B, C}
                X = pd.get_dummies(df[[f"claim_cw_{ann.split("_")[1]}_platinum"]]).astype(int)
            else:
                # Encode of llms + annotators
                X = pd.get_dummies(df[flatten + [f"claim_cw_{ann}_platinum"]]).astype(int)

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

                    results.append({
                        "split": split_id,
                        "model": model_name,
                        "encoding": mv_strat,
                        "encoding_annotator": ann,
                        "train_frac": frac,
                        "n_train": n_samples,
                        "p": p,
                        "r": r,
                        "f1": f1,
                        "kappa": kappa,
                        "y_test": y_test.replace({0: "NFS", 1: "UFS", 2: "CFS"}),
                        "pred": pd.Series([to_cat_labels(p) for p in preds]),
                        "annA": df.iloc[test_idx]["claim_cw_annA_platinum"],
                        "annB": df.iloc[test_idx]["claim_cw_annB_platinum"],
                        "annC": df.iloc[test_idx]["claim_cw_annC_platinum"],
                        "predict_proba": predict_proba
                    })
# %%
df_results = pd.DataFrame(results)
# %%
df_results.loc[df_results["predict_proba"].notna(), "pred_with_proba"] = df_results.loc[df_results["predict_proba"].notna(), "predict_proba"].apply(lambda y_prob: [to_cat_labels(p) for p in np.argmax(y_prob, axis=1)])
# %%
df_results.loc[df_results["predict_proba"].notna(), "pred_with_proba_ok"] = df_results.loc[df_results["predict_proba"].notna(), ["pred", "pred_with_proba"]].apply(lambda row: (row["pred"] == row["pred_with_proba"]).all(), axis=1)
# %%
df_results["errors"] = df_results[["y_test", "pred"]].apply(lambda row: (row["y_test"].reset_index(drop=True) != row["pred"].reset_index(drop=True)).astype(int).tolist(), axis=1)
# %%
from scipy.stats import pointbiserialr

df_results.loc[df_results["predict_proba"].notna(), ["corr_errors_prob", "p_value_errors_prob"]] = df_results.loc[df_results["predict_proba"].notna(), ["errors", "predict_proba"]].apply(lambda row: pointbiserialr(np.array(row["errors"]), row["predict_proba"].max(axis=1)), axis=1).tolist()
# %%
df_results.loc[df_results["corr_errors_prob"].notna(), "corr_errors_prob"].astype(float).describe()
# %%
df_results["corr_errors_prob"] = df_results["corr_errors_prob"].astype(float)
# %%
df_results.loc[df_results["corr_errors_prob"].notna(), "p_value_errors_prob"].astype(float).describe()
# %%
df_results["p_value_errors_prob"] = df_results["p_value_errors_prob"].astype(float)
# %%
# %%
pd.crosstab(df_results["model"], df_results["pred_with_proba_ok"])
# %%
df_results.loc[df_results["predict_proba"].notna(), "pred_with_proba_error"] = df_results.loc[df_results["predict_proba"].notna(), ["pred", "pred_with_proba"]].apply(lambda row: len(row["pred"]) - (row["pred"] == row["pred_with_proba"]).sum(), axis=1)
# %%
from itertools import product

show = [
    ("llm_loop", 0, "", ""),
    *product(
        ["mv"],
        [0.0],
        ["all", "small", "medium", "large", "mixtral", "mistral", "olmo", "qwen", "llama", "commandr"],
        ["only_llms", "annA", "annB", "annC"]
    ),
    *product(
        ["svm", "lgr", "mlp", "rforest", "xgb"],
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        ["all", "small", "medium", "large", "mixtral", "mistral", "olmo", "qwen", "llama", "commandr"],
        ["only_llms", "annA", "annB", "annC"]
    )
]
# %%
(
    df_results
    .loc[:, ["model", "train_frac", "encoding", "encoding_annotator", "n_train", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["model", "train_frac", "encoding", "encoding_annotator"])
    .agg(["mean", "std"])
    .round(3)
    .loc[show]
    #.to_csv("results_all_encodings_and_annotators.csv")
    #.loc[show]
)
# %%
# Finding one: All encodings and models benefit from platinum training data
(
    df_results
    .loc[df_results["encoding_annotator"] == "only_llms", ["model", "train_frac", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["model", "train_frac"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding1.csv")
)
# %%
# Finding two: You can choose any ensemble method, you will be fine (lgr, mlp, rforest, svm, xgb, etc)
(
    df_results
    .loc[df_results["encoding_annotator"] == "only_llms", ["model", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["model"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding2.csv")
)
# %%
# Finding three: Large and all encodings are the best and second best
(
    df_results
    .loc[df_results["encoding_annotator"] == "only_llms", ["encoding", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["encoding"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding3.csv")
)
# %%
# Finding four: Adding one of the human annotations in the encoding will lead to higher F1 and Kappa rather than using only llms
(
    df_results
    .loc[:, ["encoding_annotator", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["encoding_annotator"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding4.csv")
)
# %%
# Finding four point 2: Adding one of the human annotations in the encoding will lead to higher F1 and Kappa rather than using only llms
(
    df_results
    .loc[:, ["train_frac", "encoding_annotator", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["encoding_annotator", "train_frac"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding4.2.csv")
)
# %%
# Finding five: Results are stable across folds where variability mainly on the types of encodings and the jump of using only LLMs vs human annotations
(
    df_results
    .loc[:, ["split", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["split"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding5.1.csv")
)
# %%
(
    df_results
    .loc[:, ["split", "encoding_annotator", "p", "r", "f1", "kappa", "corr_errors_prob", "p_value_errors_prob"]]
    .groupby(["split", "encoding_annotator"])
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("finding5.2.csv")
)
# %%
# %%
from sklearn.metrics import f1_score

df_results_th = df_results.copy()
df_results_th["th"] = "base"

threshs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
res_th = []
res_th.append(df_results_th)
for th in threshs:
    
    for ann in ["annA", "annB", "annC"]:
        df_results_th = df_results.copy()
        df_results_th["th"] = th
        mask = df_results_th["predict_proba"].notna()

        df_results_th.loc[mask, "ann"] = ann
        df_results_th.loc[mask, "y_test_enc"] = df_results_th.loc[mask, "y_test"].apply(lambda pred: pred.replace({"NFS": 0, "UFS": 1, "CFS": 2}).tolist())
        df_results_th.loc[mask, "pred_enc"] = df_results_th.loc[mask, "pred"].apply(lambda pred: pred.replace({"NFS": 0, "UFS": 1, "CFS": 2}).tolist())
        df_results_th.loc[mask, f"{ann}_enc"] = df_results_th.loc[mask, ann].apply(lambda ann: ann.replace({"NFS": 0, "UFS": 1, "CFS": 2}).tolist())
        df_results_th.loc[mask, "pred"] = df_results_th.loc[mask, ["pred_enc", "predict_proba", f"{ann}_enc"]].apply(lambda row: [row["pred_enc"][i] if row["predict_proba"][i, row["pred_enc"][i]] > th else row[f"{ann}_enc"][i] for i in range(len(row["pred_enc"]))], axis=1)
        df_results_th.loc[mask, "kappa"] = df_results_th.loc[mask, ["pred", "y_test"]].apply(lambda row: cohen_kappa_score(row["pred"], row["y_test"].replace({"NFS": 0, "UFS": 1, "CFS": 2}), weights="linear", labels=[0, 1, 2]), axis=1)
        df_results_th.loc[mask, "f1"] = df_results_th.loc[mask, ["pred", "y_test"]].apply(lambda row: f1_score(row["pred"], row["y_test"].replace({"NFS": 0, "UFS": 1, "CFS": 2}), average="macro", labels=[0,1,2]), axis=1)
        df_results_th.loc[mask, "n_below_th"] = df_results_th.loc[mask, ["pred_enc", "predict_proba", f"{ann}_enc"]].apply(lambda row: sum([row["predict_proba"][i, row["pred_enc"][i]] <= th for i in range(len(row["pred_enc"]))]), axis=1)
        df_results_th.loc[mask, "n_below_th_%"] = df_results_th.loc[mask, ["pred_enc", "predict_proba", f"{ann}_enc"]].apply(lambda row: sum([row["predict_proba"][i, row["pred_enc"][i]] <= th for i in range(len(row["pred_enc"]))]) / len(row["pred_enc"]) * 100, axis=1)
        df_results_th.loc[mask, "n_below_th_ok"] = df_results_th.loc[mask, ["y_test_enc", "pred", "pred_enc", "predict_proba"]].apply(lambda row: sum([((row["pred"][i] == row["y_test_enc"][i]) & (row["predict_proba"][i, row["pred_enc"][i]] <= th)) for i in range(len(row["pred_enc"]))]), axis=1)
        df_results_th.loc[mask, "n_below_th_ok_%"] = df_results_th.loc[mask, ["y_test_enc", "pred", "pred_enc", "predict_proba"]].apply(lambda row: sum([((row["pred"][i] == row["y_test_enc"][i]) & (row["predict_proba"][i, row["pred_enc"][i]] <= th)) for i in range(len(row["pred_enc"]))]) / len(row["pred_enc"]) * 100, axis=1)
        
        df_results_th.loc[mask, "n_below_th_nc_ok"] = df_results_th.loc[mask, ["y_test_enc", "pred", "pred_enc", "predict_proba"]].apply(lambda row: sum([((row["pred_enc"][i] == row["y_test_enc"][i]) & (row["predict_proba"][i, row["pred_enc"][i]] <= th)) for i in range(len(row["pred_enc"]))]), axis=1)
        df_results_th.loc[mask, "n_below_th_nc_ok_%"] = df_results_th.loc[mask, ["y_test_enc", "pred", "pred_enc", "predict_proba"]].apply(lambda row: sum([((row["pred_enc"][i] == row["y_test_enc"][i]) & (row["predict_proba"][i, row["pred_enc"][i]] <= th)) for i in range(len(row["pred_enc"]))]) / len(row["pred_enc"]) * 100, axis=1)
        #df_results_th.loc[mask, "n_above_th_ok"] = df_results_th.loc[mask, ["y_test_enc", "pred", "pred_enc", "predict_proba"]].apply(lambda row: sum([((row["pred"][i] == row["y_test_enc"][i]) & (row["predict_proba"][i, row["pred_enc"][i]] > th)) for i in range(len(row["pred_enc"]))]), axis=1)
        df_results_th.loc[mask, "n_test"] = df_results_th.loc[mask, "pred_enc"].apply(lambda pred: len(pred))

        res_th.append(df_results_th)
    #df_results_th.loc[mask, "kappa_with_th"] = cohen_kappa_score(df_results_th.loc[mask, "pred_with_annA"], df_results_th.loc[mask, "y_true"], weights="linear", labels=[0, 1, 2])
    #df_results_th.loc[mask, "kappa_with_th"] = cohen_kappa_score(df_results_th.loc[mask, "pred_with_annB"], df_results_th.loc[mask, "y_true"], weights="linear", labels=[0, 1, 2])
    #df_results_th.loc[mask, "kappa_with_th"] = cohen_kappa_score(df_results_th.loc[mask, "pred_with_annC"], df_results_th.loc[mask, "y_true"], weights="linear", labels=[0, 1, 2])
# %%
df_res_th = pd.concat(res_th, ignore_index=True)
# %%
df_res_th[(df_res_th["ann"] == df_res_th["encoding_annotator"]) | (df_res_th["encoding_annotator"] == "only_llms")]
# %%
(
    df_res_th
    .loc[(df_res_th["ann"] == df_res_th["encoding_annotator"]) |
         (df_res_th["encoding_annotator"] == "only_llms"),
         ["th", "f1", "kappa", "n_below_th", "n_below_th_%", "n_below_th_ok", "n_below_th_ok_%", "n_below_th_nc_ok", "n_below_th_nc_ok_%", "n_test"]]
    .groupby(["th"])
    .agg(["mean", "std"])
    .round(3)
    .to_csv("finding6.1.csv")
)
# %%
(
    df_res_th
    .loc[(df_res_th["ann"] == df_res_th["encoding_annotator"]) |
         (df_res_th["encoding_annotator"] == "only_llms") |
         (df_res_th["ann"].isna() & (df_res_th["encoding_annotator"] != "")),
         ["th", "encoding_annotator", "f1", "kappa", "n_below_th", "n_below_th_%", "n_below_th_ok", "n_below_th_ok_%", "n_below_th_nc_ok", "n_below_th_nc_ok_%", "n_test"]]
    .groupby(["th", "encoding_annotator"])
    .agg(["mean", "std"])
    .round(3)
    .to_csv("finding6.2.csv")
)
# %% [markdown]
# Ensemble methods with different group of predictions
# %%
df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")

sss = StratifiedShuffleSplit(
    n_splits=5,
    train_size=0.30,
    test_size=0.70,
    random_state=0
)

fractions = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

results = []


models = {
        "lgr": LogisticRegression(max_iter=1000),
        "rforest": RandomForestClassifier(),
        "svm": SVC(probability=True),
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


N = len(df)
results = []

y = df["claim_cw_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})

for strat, flatten in mv_strats.items():
    X = pd.get_dummies(df[flatten]).astype(int)

    for split_id, (train_idx, test_idx) in enumerate(sss.split(X, y)):

        X_pool = X.iloc[train_idx]
        y_pool = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # single shuffle -> nested subsets
        rng = np.random.RandomState(split_id)

        order = rng.permutation(len(X_pool))

        X_pool = X_pool.iloc[order].reset_index(drop=True)
        y_pool = y_pool.iloc[order].reset_index(drop=True)

        # Now train small models
        for frac in fractions[1:]:
            n_samples = int(frac * N)

            X_sub = X_pool.iloc[:n_samples]
            y_sub = y_pool.iloc[:n_samples]

            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(X_sub, y_sub)
                preds = clf.predict(X_test)
                p, r, f1, sup = precision_recall_fscore_support(y_test, preds, average="macro")

                results.append({
                    "split": split_id,
                    "model": model_name,
                    "train_frac": frac,
                    "n_train": n_samples,
                    "p": p,
                    "r": r,
                    "f1": f1
                })
# %%
df_platinum = pd.read_csv("../../data/wsf_arg_plus_per_claim.csv")
# %%
from sklearn.metrics import classification_report

print(classification_report(df_platinum["claim_cw_gold"], df_platinum["claim_cw_platinum"], digits=3))
# %%
