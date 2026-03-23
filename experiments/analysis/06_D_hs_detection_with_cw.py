# %% [markdown]
# ## HS Detection on Check-worthiness
# This notebook reproduces the results of the table 6 and table C (appendix) in the paper concerning the HS detection
# %%
# Preprocessing of runs
from glob import glob
import pandas as pd

all_runs = glob("../../data/raw_generations/hs_detection/**/**/**/**/**.csv")
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
# %%
hs_res_wo_cw
# %%
assert len(hs_res_wo_cw) == 36 # 12 models run 3 times
assert len(hs_res_w_cw) == 36  # 12 models run 3 times
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
(
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
)
# %%
(
    hs_det_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
)
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
(
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").mean()
    .loc[order.values()]
)
# %%
(
    hs_det_w_cw_res_df
    .drop(columns=["run_name"])
    .groupby("model_name").std()
    .loc[order.values()]
)
# %%
