# %% [markdown]
# ## HS Detection on Check-worthiness
# This notebook reproduces the results of the table 6 and table C (appendix) in
# the paper concerning the HS detection
# %%
# Preprocessing of runs
from glob import glob
import pandas as pd

all_runs = glob("../../data/raw_generations/hs_detection/**/**/**/**/**.csv")
all_runs_platinum = glob("../../data/raw_generations/hs_detection_platinum/**/**/**/**/**.csv")
# %%
df_orig = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
# %%
order = {
    "mistral-7B-small": "Mistral-7B",
    "llama-8B-small": "Llama-8B",
    "olmo-7B-small": "Olmo2-7B",
    "qwen2.5-7B-small": "Qwen2.5-7B",
    "commandr-7B-small": "Command-r-7B",
    "mixtral-8x7B-small": "Mixtral-8x7B", # We later considered it medium in the paper
    "mistral-22B-medium": "Mistral-22B",
    "olmo2-32B-medium": "Olmo2-32B",
    "mixtral-8x22B-medium": "Mixtral-8x22B",
    "llama-70B-big": "Llama-70B",
    "qwen2.5-72B-big": "Qwen2.5-72B",
    "commandr-104B-big": "Command-r-104B",
}
# %%
order.keys()
# %%
hs_res_wo_cw = [r for r in all_runs if "base_noarg" in r]
hs_res_w_cw  = [r for r in all_runs if "with_cw_noarg" in r]
hs_res_w_cw_platinum  = [r for r in all_runs_platinum if "with_cw_noarg" in r]
# %%
def sort_key(path, order):
    for i, b in enumerate(order):
        if b in path:
            return i
    return len(order)  # if no match, push to end

hs_res_wo_cw = sorted(hs_res_wo_cw, key=lambda path: next(
        (i for i, b in enumerate(order.keys()) if b in path),
        len(order.keys())  # if no match → goes to the end
    ))
hs_res_w_cw = sorted(hs_res_w_cw, key=lambda path: next(
        (i for i, b in enumerate(order.keys()) if b in path),
        len(order.keys())  # if no match → goes to the end
    ))
hs_res_w_cw_platinum = sorted(hs_res_w_cw_platinum, key=lambda path: next(
        (i for i, b in enumerate(order.keys()) if b in path),
        len(order.keys())  # if no match → goes to the end
    ))
# %%
hs_res_wo_cw
# %%
assert len(hs_res_wo_cw) == 36 # 12 models run 3 times
assert len(hs_res_w_cw) == 36  # 12 models run 3 times
assert len(hs_res_w_cw_platinum) == 36  # 12 models run 3 times
# %% [markdown]
# Results of HS detection without Check-worthiness labels
# %%
from sklearn.metrics import precision_recall_fscore_support

hs_det_res = {}
for path in hs_res_wo_cw:
    run_name = path.split("/")[-1].removesuffix("_detect_hs_base_noarg_llm_pred.csv")
    run_name = order[run_name]
    run_id = path.split("/")[-3]
    df_run = pd.read_csv(path)
    y_true = df_run["concat_hate"]
    y_pred = df_run["concat_pred_hate"]
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
    hs_det_res[f"{run_name}_{run_id}"] = results
    df_orig[f"{run_name}_{run_id}_wo_cw_concat_pred_hate"] = y_pred

# %%
hs_det_res_df = pd.DataFrame(hs_det_res).T
# %%
hs_det_res_df = (
    hs_det_res_df
    .reset_index()
    .rename(columns={"index":"run_name"})
)
hs_det_res_df["model_name"] = hs_det_res_df["run_name"].apply(lambda s: s.split("_")[0])
# %%
hs_det_res_df_goupby = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[order.values()]
    .round(3)
)
# %%
small_models = ["Mistral-7B", "Llama-8B", "Olmo2-7B", "Qwen2.5-7B", "Command-r-7B"]
medium_models = ["Mixtral-8x7B", "Mistral-22B", "Olmo2-32B", "Mixtral-8x22B"]
large_models = ["Llama-70B", "Qwen2.5-72B", "Command-r-104B",]
# %%
hs_det_res_df_goupby.loc["Avg Small"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[small_models]
    .mean()
    .round(3)
)
# %%
# Medium Models
hs_det_res_df_goupby.loc["Avg Medium"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[medium_models]
    .mean()
    .round(3)
)
# %%
# Large Models
hs_det_res_df_goupby.loc["Avg Large"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[large_models]
    .mean()
    .round(3)
)
# %%
hs_det_res_df_goupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"]]
# %%
hs_det_res_df_goupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %% [markdown]
# Results of HS detection with check-worthiness
# %%
# %%
from sklearn.metrics import precision_recall_fscore_support

hs_det_w_cw_res = {}
for path in hs_res_w_cw:
    run_name = path.split("/")[-1].removesuffix("_detect_hs_with_cw_noarg_llm_pred.csv")
    run_name = order[run_name]
    run_id = path.split("/")[-3]
    df_run = pd.read_csv(path)
    y_true = df_run["concat_hate"]
    y_pred = df_run["concat_pred_hate"]
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
    hs_det_w_cw_res[f"{run_name}_{run_id}"] = results
    df_orig[f"{run_name}_{run_id}_w_cw_concat_pred_hate"] = y_pred

# %%
hs_det_w_cw_res_df = pd.DataFrame(hs_det_w_cw_res).T
# %%
hs_det_w_cw_res_df = (
    hs_det_w_cw_res_df
    .reset_index()
    .rename(columns={"index":"run_name"})
)
hs_det_w_cw_res_df["model_name"] = hs_det_w_cw_res_df["run_name"].apply(lambda s: s.split("_")[0])
# %%
hs_det_w_cw_res_df_groupby = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[order.values()]
    .round(3)
)
# %%
hs_det_w_cw_res_df_groupby.loc["Avg Small"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[small_models]
    .mean()
    .round(3)
)
# %%
# Medium Models
hs_det_w_cw_res_df_groupby.loc["Avg Medium"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[medium_models]
    .mean()
    .round(3)
)
# %%
# Large Models
hs_det_w_cw_res_df_groupby.loc["Avg Large"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[large_models]
    .mean()
    .round(3)
)
# %%
hs_det_w_cw_res_df_groupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"]]
# %%
hs_det_w_cw_res_df_groupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %%
# Tables in the paper
# %%
# Macro results of HS Detection with check-worthiness
hs_det_w_cw_res_df_groupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %%
# Macro results of HS Detection without check-worthiness
hs_det_res_df_goupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %%
# Delta difference on Macro F1 of HS detection: Comparison of check-worthiness labels vs without them
(
    hs_det_w_cw_res_df_groupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], "f1_macro"]
    -
    hs_det_res_df_goupby.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], "f1_macro"]
)
# %% [markdown]
# ## Standard Deviation, HS Detection without Check-worthiness
# %%
hs_det_res_df_goupby_std = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[order.values()]
    .round(3)
)
# %%
hs_det_res_df_goupby_std.loc["Avg Small"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[small_models]
    .mean() # The mean of the standard deviations per size
    .round(3)
)
# %%
# Medium Models
hs_det_res_df_goupby_std.loc["Avg Medium"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[medium_models]
    .mean()
    .round(3)
)
# %%
# Large Models
hs_det_res_df_goupby_std.loc["Avg Large"] = (
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[large_models]
    .mean()
    .round(3)
)
# %%
hs_det_res_df_goupby_std.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %%
# %% [markdown]
# ## Standard Deviation, HS Detection with Check-worthiness
# %%
hs_det_w_cw_res_df_groupby_std = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[order.values()]
    .round(3)
)
# %%
hs_det_w_cw_res_df_groupby_std.loc["Avg Small"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[small_models]
    .mean()
    .round(3)
)
# %%
# Medium Models
hs_det_w_cw_res_df_groupby_std.loc["Avg Medium"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[medium_models]
    .mean()
    .round(3)
)
# %%
# Large Models
hs_det_w_cw_res_df_groupby_std.loc["Avg Large"] = (
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[large_models]
    .mean()
    .round(3)
)
# %% [markdown]
# ## Final Tables of Standard Deviation
# %%
# Standard on HS detection with check-worthiness
hs_det_w_cw_res_df_groupby_std.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %%
# Standard on HS detection without check-worthiness
hs_det_res_df_goupby_std.loc[small_models + ["Avg Small"] + medium_models + ["Avg Medium"] + large_models + ["Avg Large"], ["p_macro", "r_macro", "f1_macro"]]
# %% [markdown]
# ## Statistical Significance
# In this part we calculate the majority voting for the 3 runs per model we did
# on HS detection for check-worthiness and without check-worthiness labels. We
# report F1 and statistical significance test.
# %%
df_orig
# %%
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

results_mv = {}
for model_name in order.values():
    pred_cols = [c for c in df_orig.columns if c.startswith(model_name)]
    pred_cols_w_cw = [c for c in pred_cols if c.endswith("_w_cw_concat_pred_hate")]
    pred_cols_wo_cw = [c for c in pred_cols if c.endswith("_wo_cw_concat_pred_hate")]
    y_pred_mv_w_cw = df_orig[pred_cols_w_cw].apply(lambda row: row.sum() >= 2, axis=1).astype(int)
    y_pred_mv_wo_cw = df_orig[pred_cols_wo_cw].apply(lambda row: row.sum() >= 2, axis=1).astype(int)
    y_true = df_orig["concat_hate"]

    p_macro_w_cw, r_macro_w_cw, f1_macro_w_cw, _ = precision_recall_fscore_support(y_true, y_pred_mv_w_cw, average='macro')
    p_macro_wo_cw, r_macro_wo_cw, f1_macro_wo_cw, _ = precision_recall_fscore_support(y_true, y_pred_mv_wo_cw, average='macro')
    
    # Two-sided McNemar test
    b = np.sum((y_pred_mv_w_cw == 1) & (y_pred_mv_wo_cw == 0))
    c = np.sum((y_pred_mv_w_cw == 0) & (y_pred_mv_wo_cw == 1))

    table = [[0, b],
             [c, 0]]

    stat_test = mcnemar(table, exact=True)

    res = {
        "p_macro_w_cw": p_macro_w_cw,
        "r_macro_w_cw": r_macro_w_cw,
        "f1_macro_w_cw": f1_macro_w_cw,
        "p_macro_wo_cw": p_macro_wo_cw,
        "r_macro_wo_cw": r_macro_wo_cw,
        "f1_macro_wo_cw": f1_macro_wo_cw,
        "stat": stat_test.statistic,
        "p-value": stat_test.pvalue,
    }

    results_mv[model_name] = res
# %%
pd.DataFrame(results_mv).T.round(3)
# %% [markdown]
# Results of HS with check-worthiness in platinum labels
# %%
hs_det_w_cw_res_platinum = {}
for path in hs_res_w_cw_platinum:
    run_name = path.split("/")[-1].removesuffix("_detect_hs_with_cw_noarg_llm_pred.csv")
    run_name = order[run_name]
    run_id = path.split("/")[-3]
    df_run = pd.read_csv(path)
    y_true = df_run["concat_hate"]
    y_pred = df_run["concat_pred_hate"]
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
    hs_det_w_cw_res_platinum[f"{run_name}_{run_id}"] = results
    df_orig[f"{run_name}_{run_id}_w_cw_concat_pred_hate_platinum"] = y_pred

# %%
hs_det_w_cw_res_platinum_df = pd.DataFrame(hs_det_w_cw_res_platinum).T
# %%
hs_det_w_cw_res_platinum_df
# %%
hs_det_w_cw_res_platinum_df = (
    hs_det_w_cw_res_platinum_df
    .reset_index()
    .rename(columns={"index":"run_name"})
)
hs_det_w_cw_res_platinum_df["model_name"] = hs_det_w_cw_res_platinum_df["run_name"].apply(lambda s: s.split("_")[0])
# %%
hs_det_w_cw_res_platinum_df
# %%
hs_det_w_cw_res_platinum_df_groupby = (
    hs_det_w_cw_res_platinum_df
    .drop(columns=["run_name"])
    .groupby("model_name")
    .agg(["mean","std"])
    .loc[order.values()]
    .round(3)
)
# %%
hs_det_w_cw_res_platinum_df_groupby.loc["Avg Small"] = (
    hs_det_w_cw_res_platinum_df_groupby
    .loc[small_models]
    .mean()
    .round(3)
)
# %%
# Medium Models
hs_det_w_cw_res_platinum_df_groupby.loc["Avg Medium"] = (
    hs_det_w_cw_res_platinum_df_groupby
    .loc[medium_models]
    .mean()
    .round(3)
)
# %%
# Large Models
hs_det_w_cw_res_platinum_df_groupby.loc["Avg Large"] = (
    hs_det_w_cw_res_platinum_df_groupby
    .loc[large_models]
    .mean()
    .round(3)
)
# %%
hs_det_w_cw_res_platinum_df_groupby.loc[
    small_models +
    ["Avg Small"] +
    medium_models +
    ["Avg Medium"] +
    large_models +
    ["Avg Large"],
    ["p_macro", "r_macro", "f1_macro"]
]
# %%
# %% [markdown]
# ## Statistical Significance with platinum
# In this part we calculate the majority voting for the 3 runs per model we did
# on HS detection for check-worthiness and without check-worthiness labels. We
# report F1 and statistical significance test.
# %%
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

results_mv_platinum = {}
for model_name in order.values():
    pred_cols = [c for c in df_orig.columns if c.startswith(model_name)]
    pred_cols_w_cw_platinum = [c for c in pred_cols if c.endswith("_w_cw_concat_pred_hate_platinum")]
    pred_cols_wo_cw = [c for c in pred_cols if c.endswith("_wo_cw_concat_pred_hate")]
    y_pred_mv_w_cw_platinum = df_orig[pred_cols_w_cw_platinum].apply(lambda row: row.sum() >= 2, axis=1).astype(int)
    y_pred_mv_wo_cw = df_orig[pred_cols_wo_cw].apply(lambda row: row.sum() >= 2, axis=1).astype(int)
    y_true = df_orig["concat_hate"]

    p_macro_w_cw_platinum, r_macro_w_cw_platinum, f1_macro_w_cw_platinum, _ = precision_recall_fscore_support(y_true, y_pred_mv_w_cw_platinum, average='macro')
    p_macro_wo_cw, r_macro_wo_cw, f1_macro_wo_cw, _ = precision_recall_fscore_support(y_true, y_pred_mv_wo_cw, average='macro')
    
    # Two-sided McNemar test
    b = np.sum((y_pred_mv_w_cw_platinum == 1) & (y_pred_mv_wo_cw == 0))
    c = np.sum((y_pred_mv_w_cw_platinum == 0) & (y_pred_mv_wo_cw == 1))

    table = [[0, b],
             [c, 0]]

    stat_test = mcnemar(table, exact=True)

    res = {
        "p_macro_w_cw_platinum": p_macro_w_cw_platinum,
        "r_macro_w_cw_platinum": r_macro_w_cw_platinum,
        "f1_macro_w_cw_platinum": f1_macro_w_cw_platinum,
        "p_macro_wo_cw": p_macro_wo_cw,
        "r_macro_wo_cw": r_macro_wo_cw,
        "f1_macro_wo_cw": f1_macro_wo_cw,
        "stat": stat_test.statistic,
        "p-value": stat_test.pvalue,
    }

    results_mv_platinum[model_name] = res
# %%
pd.DataFrame(results_mv_platinum).T.round(3)
# %%
