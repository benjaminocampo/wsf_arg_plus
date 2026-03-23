# %% [markdown]
# ## IAA of full human annotation strategy
# This notebook reproduces table A in the appendix of the paper.
# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_platinum_disagg.csv")
# %%
from sklearn.metrics import cohen_kappa_score
# %%
# Ann1 vs Ann2
cohen_kappa_score(df["claim_cw_annA_platinum"], df["claim_cw_annB_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(
    df["claim_cw_annA_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df["claim_cw_annB_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])
# %%
# Ann2 vs Ann3
# TODO: In the paper we put this setting as Ann1 vs Ann3. Correct it.
cohen_kappa_score(df["claim_cw_annB_platinum"], df["claim_cw_annC_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(
    df["claim_cw_annB_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df["claim_cw_annC_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])
# %%
# Ann1 vs Ann3
cohen_kappa_score(df["claim_cw_annA_platinum"], df["claim_cw_annC_platinum"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(
    df["claim_cw_annA_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df["claim_cw_annC_platinum"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])
# %%
