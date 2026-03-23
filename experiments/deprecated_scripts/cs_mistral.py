# %%
import pandas as pd

slot1 = pd.read_csv("../data/slot1_annotations.csv")
# %%
slot2 = pd.read_csv("../data/slot2_annotations.csv")
# %%
slot2
# %%

# df_scores = pd.read_csv("../data/openai_moderation_wsf.csv")
df = pd.read_csv("../data/wsf_arg_extension_with_nohs_cw.csv")

# %%
cols = ["premise0", "premise1", "premise2", "premise3", "conclusion"]

iscws = []
for _, row in df.iterrows():
    cws = [row[f"{col}_cw_final"] for col in cols]
    iscw = any(cw == "CFS" for cw in cws)
    iscws.append(iscw)
df["contains_cw"] = iscws
# %%
df
# %%
slot1_w_cw = slot1.merge(df[["idx", "contains_cw"]], on="idx", how="left")
# %%
slot2_w_cw = slot2.merge(df[["idx", "contains_cw"]], on="idx", how="left")
# %%

# %%
evals = pd.concat([slot1_w_cw, slot2_w_cw])
# %%
evals["strat"].value_counts()
# %%
pd.crosstab(evals["strat"], evals["contains_cw"])
# %%
evals.columns
# %%
evals[["contains_cw", "suitableness", "informativeness", "cogency", "relevance"]].groupby("contains_cw").describe()
# %%
