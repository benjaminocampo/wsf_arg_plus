# %%
import pandas as pd

df = pd.read_csv("../data/wsf_arg_plus_gold.csv")
# %%
import numpy as np

cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]

df = pd.read_csv("../data/wsf_arg_plus_gold.csv")
df["premise3_hate"] = np.nan
df["premise4_hate"] = np.nan
df["premise5_hate"] = np.nan

df_claims_gold = pd.concat([df[[f"{c}_cw_ann_final_gold", "concat_hate", f"{c}_hate", f"{c}_agreement_count_gold"]].rename(columns={f"{c}_cw_ann_final_gold": "claim_cw_ann_final_gold", f"{c}_hate": "claim_hate", f"{c}_agreement_count_gold": "claim_agreement_count_gold"}) for c in cols])
df_claims_gold = df_claims_gold.loc[~df_claims_gold["claim_cw_ann_final_gold"].isna()]
# %%
pd.crosstab(df_claims_gold["claim_cw_ann_final_gold"], df_claims_gold["claim_hate"]).T
# %%
df_claims_gold["claim_cw_ann_final_gold"].value_counts()
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

iscws = []
for _, row in df.iterrows():
    cws = [row[f"{col}_cw_ann_final_gold"] for col in cols]
    iscw = any(cw == "CFS" for cw in cws)
    iscws.append(iscw)
df["contains_cw"] = iscws
# %%
df[df["concat_hate"] == 1]["contains_cw"].value_counts()
# %%
df_claims_gold["claim_agreement_count_gold"].value_counts()
# %%
df_claims_gold["claim_agreement_count_gold"].value_counts()
# %%
df.columns
# %%
from collections import Counter

def majority_vote_per_claim(row, col):
    values = [row[f"{col}_cw_annA"], row[f"{col}_cw_annB"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({f"{col}_cw_final_aux": most_common_value, f"{col}_agreement_count": count})
    else:
        return pd.Series({f"{col}_cw_final_aux": "All unequal", f"{col}_agreement_count_silver": 1})

# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]
for c in cols:
    df[[f"{c}_cw_final_aux", f"{c}_agreement_count_silver"]] = df.apply(lambda row: majority_vote_per_claim(row, c), axis=1)
# %%
df.columns
# %%
def majority_vote_per_claim(row, col):
    values = [row[f"{col}_cw_annA"], row[f"{col}_cw_annB"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({f"{col}_cw_ann_final_aux": most_common_value, f"{col}_agreement_count_aux": count})
    else:
        return pd.Series({f"{col}_cw_ann_final_aux": "All unequal", f"{col}_agreement_count_aux": 1})
# %%
for c in cols:
    df[[f"{c}_cw_ann_final_aux", f"{c}_agreement_count_aux"]] = df.apply(lambda row: majority_vote_per_claim(row, c), axis=1)
# %%
for c in cols:
    df.loc[df[f"{c}_cw_ann_final_aux"] == "All unequal", f"{c}_cw_ann_final_aux"] = df.loc[df[f"{c}_cw_ann_final_aux"] == "All unequal", f"{c}_cw_final"]
# %%
for c in cols:
    df[f"{c}_cw_final"] = df[f"{c}_cw_ann_final_aux"]
# %%
df[f"conclusion_cw_final"].value_counts()
# %%
df[f"conclusion_cw_final"].isna().sum()
# %%
df_claims_ann = pd.concat([df[[f"{c}_cw_ann_final_gold", f"{c}_cw_final",  "concat_hate", f"{c}_hate"]].rename(columns={f"{c}_cw_ann_final_gold": "claims_cw_ann_final_gold", f"{c}_cw_final": f"claims_cw_final", f"{c}_hate": "claim_hate"}) for c in cols])
df_claims_ann = df_claims_ann.loc[~df_claims_ann["claims_cw_ann_final_gold"].isna()]
# %%

df_claims_ann["claims_cw_ann_final_gold"]
# %%
df_claims_ann["claims_cw_final"]
# %%
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(df_claims_ann["claims_cw_ann_final_gold"], df_claims_ann["claims_cw_final"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(
    df_claims_ann["claims_cw_ann_final_gold"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    df_claims_ann["claims_cw_final"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])
# %%
df
# %%
pd.crosstab(df_claims_ann["claims_cw_ann_final_gold"], df_claims_ann["claim_hate"])
# %%
pd.crosstab(df_claims_ann["claims_cw_final"], df_claims_ann["claim_hate"])
# %%
pd.crosstab(df_claims_ann["claims_cw_final"], df_claims_ann["concat_hate"])
# %%
pd.crosstab(df_claims_ann["claims_cw_ann_final_gold"], df_claims_ann["concat_hate"])
# %%
