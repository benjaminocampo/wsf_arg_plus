# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
df.columns
# %%
runs = {
	"Mistral-7B": ["zero","one"],
	"Llama-8B": ["zero", "one"],
	"Olmo2-7B": ["zero", "one"],
	"Qwen2.5-7B": ["zero", "one"],
	"Command-r-7B": ["zero", "one"],
	"Mixtral-8x7B": ["zero", "one"],
	"Mistral-22B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
	"Mixtral-8x22B": ["zero", "one"],
	"Llama-70B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
	"Command-r-104B": ["zero", "one"],
}
runs = [f"{model_name}_{shot}" for model_name, shots in runs.items() for shot in shots]
# %%
from collections import Counter

def majority_vote(row, col):
    values = [row[f"{col}_m0"], row[f"{col}_m1"], row[f"{col}_m2"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({col: most_common_value, f"{col}_agreement_count": count})
    else:
        return pd.Series({col: "All unequal", f"{col}_agreement_count": 1})
# %%
# HS Table of prediction variability
df_hs = df[df["concat_hate"] == 1]
variability_hs = {}
for r in runs:
    col = f"{r}_claim_cw"
    df_variability = df_hs.apply(lambda row: majority_vote(row, col), axis=1)
    variability_hs[r] = df_variability[f"{col}_agreement_count"].value_counts().to_dict()
    # If 1, 2, or 3 don't exist set agreement_count in 0
    variability_hs[r].setdefault(1, 0)
    variability_hs[r].setdefault(2, 0)
    variability_hs[r].setdefault(3, 0)
    variability_hs[r]["3_%"] = variability_hs[r][3] / len(df_hs) * 100
    variability_hs[r]["2_%"] = variability_hs[r][2] / len(df_hs) * 100
    variability_hs[r]["1_%"] = variability_hs[r][1] / len(df_hs) * 100
# %%
pd.DataFrame(variability_hs).T.round(1)[[3, "3_%", 2, "2_%", 1, "1_%"]]
# %%
# Non-HS Table of prediction variability
df_non_hs = df[df["concat_hate"] == 0]
variability_non_hs = {}
for r in runs:
    col = f"{r}_claim_cw"
    df_variability = df_non_hs.apply(lambda row: majority_vote(row, col), axis=1)
    variability_non_hs[r] = df_variability[f"{col}_agreement_count"].value_counts().to_dict()
    # If 1, 2, or 3 don't exist set agreement_count in 0
    variability_non_hs[r].setdefault(1, 0)
    variability_non_hs[r].setdefault(2, 0)
    variability_non_hs[r].setdefault(3, 0)
    variability_non_hs[r]["3_%"] = variability_non_hs[r][3] / len(df_non_hs) * 100
    variability_non_hs[r]["2_%"] = variability_non_hs[r][2] / len(df_non_hs) * 100
    variability_non_hs[r]["1_%"] = variability_non_hs[r][1] / len(df_non_hs) * 100
# %%
pd.DataFrame(variability_non_hs).T.round(1)[[3, "3_%", 2, "2_%", 1, "1_%"]]
# %%
