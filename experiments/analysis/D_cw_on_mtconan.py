
# %%
import pandas as pd

df = pd.read_csv("../../data/mtconan_olmo2-32B_cw_predictions.csv")
# %%
df["cw"] = df["cw"].replace({
    "Non-Factual": "NFS",
    "Unimportant Factual": "UFS",
    "Check-worthy Factual": "CFS"
})
# %%
df["cw"].value_counts()
# %%
df["cw"].value_counts() / len(df["cw"])
# %%
mtconan_count = pd.crosstab(df["TARGET"], df["cw"]).T
# %%
targets = mtconan_count.columns
for t in targets:
    nof_messages_p_target = mtconan_count[t].sum()
    mtconan_count[f"{t}_%"] = mtconan_count[t] / nof_messages_p_target
# %%
mtconan_count
# %%
