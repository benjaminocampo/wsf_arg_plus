# %% [markdown]
# ## IAA Agreement: Olmo2-32B-zero vs Other LLMs
# This notebook reproduces the results of Table 3 of the paper.
# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
from sklearn.metrics import cohen_kappa_score
# %%
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
olmo2_32_zero_hs = df[df["concat_hate"] == 1]["Olmo2-32B_zero_claim_cw"]
olmo2_32_zero_non_hs = df[df["concat_hate"] == 0]["Olmo2-32B_zero_claim_cw"]
iaa_res = {}
for r in runs:
    iaa_res[r] = {}
    iaa_res[r]["non_hs"] = cohen_kappa_score(olmo2_32_zero_non_hs, df[df["concat_hate"] == 0][f"{r}_claim_cw"], weights="linear", labels=["NFS", "UFS", "CFS"])
    iaa_res[r]["hs"] = cohen_kappa_score(olmo2_32_zero_hs, df[df["concat_hate"] == 1][f"{r}_claim_cw"], weights="linear", labels=["NFS", "UFS", "CFS"])
    iaa_res[r]["overall"] = np.mean([iaa_res[r]["non_hs"], iaa_res[r]["hs"]])
# %%
pd.DataFrame(iaa_res).T
# %%
iaa_res
# %%
