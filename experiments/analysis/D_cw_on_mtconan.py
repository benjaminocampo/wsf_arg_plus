
# %%
import pandas as pd

df = pd.read_csv("../../data/mtconan_olmo2-32B_cw_predictions_only.csv")
# %% [markdown]
# For licensing matters we can not put in the `data/` directory the dataset
# MT-CONAN as the creators of the datasets did not allow for re-distribution. We
# can only redistribute the IDs of the data and the predicted labels of
# check-worthiness. To proceed and execute this notebook, place MT-CONAN in
# `data/` locally and using `df`, perform a merge a left join on the IDs to
# obtain the text, target, and the other columns of the dataset. To execute this
# notebook you will need the TARGET column.
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
    mtconan_count[f"{t}_%"] = mtconan_count[t] / nof_messages_p_target * 100
# %%
mtconan_count.round(2)
# %%
order = []
for t in targets:
    order.append(t)
    order.append(f"{t}_%")

mtconan_count[order].round(2)
# %%
