# %%
import pandas as pd

olmo2_mv = pd.read_csv("./cw_gens/non_hs/Olmo2-32B/zero/mj_vot.csv")
# %%
cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]
# %%
olmo_claims_ann = pd.concat([olmo2_mv[[f"{c}_cw_annA", f"{c}_cw_annB", "concat_hate"]].rename(columns={f"{c}_cw_annA": "claims_cw_annA", f"{c}_cw_annB": "claims_cw_annB"}) for c in cols])
# %%
olmo_claims_ann.dropna(subset="claims_cw_annA")
# %%
y1 = olmo_claims_ann.loc[~olmo_claims_ann["claims_cw_annA"].isna(), "claims_cw_annA"].reset_index(drop=True)
y2 = olmo_claims_ann.loc[~olmo_claims_ann["claims_cw_annA"].isna(), "claims_cw_annB"].reset_index(drop=True)
y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
# %%
non_hs_runs = {
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
# %%
# Reading new dfs
non_hs_runs_df = {}
for model_name, shot_types in non_hs_runs.items():
    non_hs_runs_df[model_name] = {}
    for shot in shot_types:
        non_hs_runs_df[model_name][shot] = pd.read_csv(f"./cw_gens/non_hs/{model_name}/{shot}/mj_vot.csv")

# %%
agreement = olmo_claims_ann[y1 == y2]
# %%
disagreement = olmo_claims_ann[y1 != y2]
# %%
agreement
# %%
disagreement
# %%
olmo2_mv
# %%
olmo_claims_ann
# %%
