# %%[markdown]
# ## Percent Agreement
# This notebook reproduces the results of Table 2 of the paper.
# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
claim_gold_annA = pd.read_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv")[["concat_hate", "claim_cw_annA_gold"]]
# %%
runs = {
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
runs = [f"{model_name}_{shot}" for model_name, shots in runs.items() for shot in shots]
# %% [markdown]
# We calculate for all the run configurations (12 LLMs x 2 prompts (zero + one
# shot) = 24 configurations) the percent agreement with the first annotator
# (named annA)
# %%
claim_gold_annA_hs = claim_gold_annA[claim_gold_annA["concat_hate"] == 1]
claim_gold_annA_non_hs = claim_gold_annA[claim_gold_annA["concat_hate"] == 0]
llm_mv_hs = df[df["concat_hate"] == 1]
llm_mv_non_hs = df[df["concat_hate"] == 0]
percent_agreement_tab = {}
for r in runs:
    percent_agreement_tab[r] = {}
    percent_agreement_tab[r]["all_claims_hs"] = (claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]).sum() / len(claim_gold_annA_hs["claim_cw_annA_gold"])
    percent_agreement_tab[r]["CFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")
    percent_agreement_tab[r]["NFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")
    percent_agreement_tab[r]["UFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")

    percent_agreement_tab[r]["all_claims_non_hs"] = (claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]).sum() / len(claim_gold_annA_non_hs["claim_cw_annA_gold"])
    percent_agreement_tab[r]["CFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")
    percent_agreement_tab[r]["NFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")
    percent_agreement_tab[r]["UFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")
# %%
# We are using rounding to 3 digits in the paper.
# Note: This percent agreement is calculated with the majority voting labels.
pd.DataFrame(percent_agreement_tab).T.round(3)
# %% [markdown]
# For each of the 24 configurations, we run it 3 times obtaining 96 runs which
# we calculate the percent agreement and we calculate the standard deviation.
# %%
percent_agreement_tab_disagg = {}
for r in runs:
    for m in range(3):
        percent_agreement_tab_disagg[f"{r}_{m}"] = {}
        percent_agreement_tab_disagg[f"{r}_{m}"]["all_claims_hs"] = (claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw_m{m}"]).sum() / len(claim_gold_annA_hs["claim_cw_annA_gold"])
        percent_agreement_tab_disagg[f"{r}_{m}"]["CFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")
        percent_agreement_tab_disagg[f"{r}_{m}"]["NFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")
        percent_agreement_tab_disagg[f"{r}_{m}"]["UFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")

        percent_agreement_tab_disagg[f"{r}_{m}"]["all_claims_non_hs"] = (claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw_m{m}"]).sum() / len(claim_gold_annA_non_hs["claim_cw_annA_gold"])
        percent_agreement_tab_disagg[f"{r}_{m}"]["CFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")
        percent_agreement_tab_disagg[f"{r}_{m}"]["NFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")
        percent_agreement_tab_disagg[f"{r}_{m}"]["UFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw_m{m}"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")
# %%
df_percent_agreement_tab_disagg = pd.DataFrame(percent_agreement_tab_disagg).T
# %%
df_percent_agreement_tab_disagg = df_percent_agreement_tab_disagg.reset_index().rename(columns={"index": "run_name"})
# %%
df_percent_agreement_tab_disagg["model"] = df_percent_agreement_tab_disagg["run_name"].apply(lambda s: "_".join(s.split("_")[:2]))
# %%
(
    df_percent_agreement_tab_disagg
    .drop(columns=["run_name"])
    .groupby("model")
    .std()
    .round(3)
    .loc[runs]
)
# %% [markdown]
# Majority voting of the runs with platinum using F1-score.
# %%
from sklearn.metrics import precision_recall_fscore_support

results_f1 = {}
for r in runs:
    y_pred = df[f"{r}_claim_cw"]
    y_true = df["claim_cw_platinum"]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])
    
    res = {
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
    }
    results_f1[r] = res
# %%
pd.DataFrame(results_f1).T
# %%
# Mean and std of the runs when compared with platinum labels using F1-score.
# %%
results_f1_disagg = {}
for r in runs:
    for i in range(3): # We predicted cw labels 3 times per run.
        y_pred = df[f"{r}_claim_cw_m{i}"]
        y_true = df["claim_cw_platinum"]

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])

        res = {
            "p_macro": p_macro,
            "r_macro": r_macro,
            "f1_macro": f1_macro,
        }
        results_f1_disagg[f"{r}_m{i}"] = res
# %%
df_f1_res_disagg = pd.DataFrame(results_f1_disagg).T
# %%
df_f1_res_disagg = df_f1_res_disagg.reset_index().rename(columns={"index": "run"})
df_f1_res_disagg["model_name"] = df_f1_res_disagg["run"].apply(lambda r: "_".join(r.split("_")[:-1]))
# %%
(
    df_f1_res_disagg[["model_name", "p_macro", "r_macro", "f1_macro"]]
    .groupby("model_name")
    .agg(["mean", "std"])
    .loc[runs]
    .round(3)
)
# %%
