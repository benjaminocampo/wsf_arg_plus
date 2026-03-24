# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim.csv")
df_disagg_gold = pd.read_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv")
# %%
from sklearn.metrics import cohen_kappa_score
# %%
# In this case annB is Olmo2-32B for the gold annotations (Results of Section
# 5.1, IAA of Olmo2-32B with human annotator)
# %%
# Overall IAA w/ 3 labels NFS, UFS, CFS
cohen_kappa_score(df_disagg_gold["claim_cw_annA_gold"], df_disagg_gold["claim_cw_annB_gold"], weights="linear", labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
# IAA w/ 3 labels NFS, UFS, CFS on HS messages
cohen_kappa_score(
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 1, "claim_cw_annA_gold"],
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 1, "claim_cw_annB_gold"],
    weights="linear",
    labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
# IAA w/ 3 labels NFS, UFS, CFS on Non-HS messages
cohen_kappa_score(
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 0, "claim_cw_annA_gold"],
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 0, "claim_cw_annB_gold"],
    weights="linear",
    labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
# Overall IAA w/ 2 labels NFS + UFS = Non-check-worthy, CFS = Check-worthy
cohen_kappa_score(
    df_disagg_gold["claim_cw_annA_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_disagg_gold["claim_cw_annB_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
# IAA w/ 2 labels NFS + UFS = Non-check-worthy, CFS = Check-worthy on HS messages
cohen_kappa_score(
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 1, "claim_cw_annA_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 1, "claim_cw_annB_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
# IAA w/ 2 labels NFS + UFS = Non-check-worthy, CFS = Check-worthy on Non-HS messages
cohen_kappa_score(
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 0, "claim_cw_annA_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_disagg_gold.loc[df_disagg_gold["concat_hate"] == 0, "claim_cw_annB_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %% [markdown]
# Judge Results for the LLM-in-the-loop with Olmo2-32B
# We also include here the ones for the full human evaluation
# %%
llm_loop_needed_judge = df.loc[df["claim_needed_judge_gold"] == 1, ["claim_idx", "concat_hate", "claim_hate"]]
# %%
# Quick Test, checking that the claims that need a judge are the difference between annA and annB
assert all(
    df_disagg_gold[df_disagg_gold["claim_cw_annA_gold"] != df_disagg_gold["claim_cw_annB_gold"]]["claim_idx"] ==
    llm_loop_needed_judge["claim_idx"]
)
# %%
len(llm_loop_needed_judge)
# %%
(len(llm_loop_needed_judge) / len(df)).__round__(4) * 100
# %%
# Number of claims to be judged in Non-HS messages
len(llm_loop_needed_judge[llm_loop_needed_judge["concat_hate"] == 0])
# %%
# Number of claims to be judged in HS messages
len(llm_loop_needed_judge[llm_loop_needed_judge["concat_hate"] == 1])
# %%
# Percent of claims to be judged in HS messages
(len(llm_loop_needed_judge[llm_loop_needed_judge["concat_hate"] == 1]) / len(df[df["concat_hate"] == 1])).__round__(4) * 100
# %%
# Percent of claims to be judged in Non-HS messages
(len(llm_loop_needed_judge[llm_loop_needed_judge["concat_hate"] == 0]) / len(df[df["concat_hate"] == 0])).__round__(4) * 100
# %%
llm_loop_needed_judge_disagg = llm_loop_needed_judge.merge(df_disagg_gold[["claim_idx", "claim_cw_annA_gold", "claim_cw_annB_gold", "claim_cw_gold"]], on="claim_idx", how="left")
# %%
# Number of cases the Judge agreed with Human
(llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annA_gold"]).sum()
# %%
# Percent of cases the Judge agreed with Human
((llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annA_gold"]).sum() / len(llm_loop_needed_judge_disagg)).__round__(4) * 100
# %%
# Number of cases the Judge agreed with LLM (Olmo2-32B zero)
(llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annB_gold"]).sum()
# %%
# Percent of cases the Judge agreed with LLM (Olmo2-32B zero)
((llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annB_gold"]).sum() / len(llm_loop_needed_judge_disagg)).__round__(4) * 100
# %%
# From the total number of messages to be judged we calculate the difference
# with the number of claims the judge agreed with the LLM. We call these cases
# "disagreements" of the LLM w.r.t to the humans. To-be-judged claims are
# disagreements with the first annotator (annA) + unstable predictions of the
# LLM (All unequal cases).
len(llm_loop_needed_judge_disagg) - (llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annB_gold"]).sum()
# %%
# Percent of disagreements after the adjudication phase (after the judge)
((len(llm_loop_needed_judge_disagg) - (llm_loop_needed_judge_disagg["claim_cw_gold"] == llm_loop_needed_judge_disagg["claim_cw_annB_gold"]).sum()) / len(df)).__round__(4) * 100