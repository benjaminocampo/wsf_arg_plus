# %%
import pandas as pd
df = pd.read_csv("../data/mtconan_llm_pred.csv")
# %%
df["cw"].value_counts()
# %%
pd.crosstab(df["cw"], df["TARGET"])
# %%
