# %% [markdown]
# ## IAA of full human annotation strategy
# This notebook reproduces table A in the appendix of the paper.
# %%
import pandas as pd

df_platinum = pd.read_csv("../../data/wsf_arg_plus_per_claim_platinum_disagg.csv")
df_gold = pd.read_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv")
# %%
from sklearn.metrics import cohen_kappa_score
# %%
# Ann1 vs Ann2
cohen_kappa_score(df_platinum["claim_cw_annA_platinum"], df_platinum["claim_cw_annB_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
cohen_kappa_score(
    df_platinum["claim_cw_annA_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_platinum["claim_cw_annB_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
# Ann1 vs Ann3
cohen_kappa_score(df_platinum["claim_cw_annA_platinum"], df_platinum["claim_cw_annC_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
cohen_kappa_score(
    df_platinum["claim_cw_annA_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_platinum["claim_cw_annC_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
# Ann2 vs Ann3
# TODO: In the paper we put this setting as Ann1 vs Ann3. Correct it.
cohen_kappa_score(df_platinum["claim_cw_annB_platinum"], df_platinum["claim_cw_annC_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
cohen_kappa_score(
    df_platinum["claim_cw_annB_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_platinum["claim_cw_annC_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
import krippendorff
import numpy as np

ann1_3l = df_platinum["claim_cw_annA_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})
ann2_3l = df_platinum["claim_cw_annB_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})
ann3_3l = df_platinum["claim_cw_annC_platinum"].replace({"NFS": 0, "UFS": 1, "CFS": 2})

ann1_2l = df_platinum["claim_cw_annA_platinum"].replace({"NFS": 0, "UFS": 0, "CFS": 1})
ann2_2l = df_platinum["claim_cw_annB_platinum"].replace({"NFS": 0, "UFS": 0, "CFS": 1})
ann3_2l = df_platinum["claim_cw_annC_platinum"].replace({"NFS": 0, "UFS": 0, "CFS": 1})

reliability_data_3l = np.vstack([ann1_3l, ann2_3l, ann3_3l])
reliability_data_2l = np.vstack([ann1_2l, ann2_2l, ann3_2l])

krippendorff.alpha(reliability_data=reliability_data_3l, level_of_measurement="ordinal").__round__(3)
# %%
krippendorff.alpha(reliability_data=reliability_data_2l, level_of_measurement="ordinal").__round__(3)
# %%
llm_loop_needed_judge = df_gold.loc[df_gold["claim_needed_judge_gold"] == 1, ["claim_idx", "concat_hate", "claim_hate"]]
full_human_needed_judge = df_platinum.loc[df_platinum["claim_needed_judge_platinum"] == 1, ["claim_idx", "concat_hate", "claim_hate"]]
# %%
# Number of cases to be judged in the full human setting
len(full_human_needed_judge)
# %%
# From the total of claims, percent of cases to be judged in the full human setting
# %%
(len(full_human_needed_judge) / len(df_platinum)).__round__(4) * 100
# %%
# Number of cases to be judged in the full human setting that are also to be judged in the LLM-in-the-loop setting
sum(claim in llm_loop_needed_judge["claim_idx"].to_list() for claim in full_human_needed_judge["claim_idx"].to_list())
# %%
# Percent of cases to be judged in the full human setting that are also to be judged in the LLM-in-the-loop setting
(sum(claim in llm_loop_needed_judge["claim_idx"].to_list() for claim in full_human_needed_judge["claim_idx"].to_list()) / len(full_human_needed_judge)).__round__(4) * 100
# %%
# 1: all annotators disagree, 2: two annotators agree, one disagree, 3: all agree
df_platinum["claim_agreement_count_platinum"].value_counts()
# %%
# Cohen's Kappa between Gold and Platinum annotations w/ 3 labels NFS, UFS, CFS
cohen_kappa_score(df_gold["claim_cw_gold"], df_platinum["claim_cw_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"]).__round__(3)
# %%
# Cohen's Kappa between Gold and Platinum annotations w/ 2 labels NFS + UFS =
# "Non-Check-worthy", CFS="Check-worthy"
cohen_kappa_score(
    df_gold["claim_cw_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_platinum["claim_cw_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"]).__round__(3)
# %%
