# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
# %% [markdown]
# ## Descriptive Statistics
# In this notebook we are going to reproduce the results shown in Table 1 of the paper.
# %%
# Number of Messages HS (coded as 1) and Non-HS (coded as 0)
df["concat_hate"].value_counts()
# %%
# Total Number of Messages that are arguments (and therefore have a conclusion)
pd.crosstab(df["is_argument"], df["concat_hate"])
# %%
premises = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5"]

# Premises on HS messages
(
    df.loc[(df["is_argument"] == "yes") &
           (df["concat_hate"] == 1), premises]
      .notna()
      .sum()
      .sum()
)
# %%
# Premises on Non-HS messages
(
    df.loc[(df["is_argument"] == "yes") &
           (df["concat_hate"] == 0), premises]
      .notna()
      .sum()
      .sum()
)
# %%
# Total Number of Claims on HS messages
(
    df.loc[(df["concat_hate"] == 1), premises + ["conclusion"]]
      .notna()
      .sum()
      .sum()
)
# %%
# Total Number of Claims on Non-HS messages
(
    df.loc[(df["concat_hate"] == 0), premises + ["conclusion"]]
      .notna()
      .sum()
      .sum()
)
# %%
# Mean, Std, Min, Max Total Number of Claims on HS messages
df.loc[df["concat_hate"] == 1, premises + ["conclusion"]].apply(lambda row: row[premises + ["conclusion"]].notna().sum(), axis=1).describe()
# %%
# Mean, Std, Min, Max Total Number of Claims on Non-HS messages
df.loc[df["concat_hate"] == 0, premises + ["conclusion"]].apply(lambda row: row[premises + ["conclusion"]].notna().sum(), axis=1).describe()
# %%
