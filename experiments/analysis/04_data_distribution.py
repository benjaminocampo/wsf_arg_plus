# %% [markdown]
# ## Data Distribution for Platinum and Gold
# This notebook reproduces the results for Table 4 of the paper
# %% [markdown]
# ## Platinum (Full Human)
# %%
import pandas as pd
df = pd.read_csv("../../data/wsf_arg_plus_per_claim.csv")
# %%
pd.crosstab(df["concat_hate"], df["claim_cw_platinum"]).T
# %%
# claim_hate has null values for Non-HS messages so the counting of pd.crosstab doesn't consider the Non-HS part.
pd.crosstab(df["claim_hate"], df["claim_cw_platinum"])
# %%
# %% [markdown]
# ## Gold (LLM-in-the-loop)
# %%
pd.crosstab(df["concat_hate"], df["claim_cw_gold"]).T
# %%
# claim_hate has null values for Non-HS messages so the counting of pd.crosstab doesn't consider the Non-HS part.
pd.crosstab(df["claim_hate"], df["claim_cw_gold"])