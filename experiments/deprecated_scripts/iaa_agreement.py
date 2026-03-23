# %%
import pandas as pd

olmo2_mv = pd.read_csv("./cw_gens/non_hs/Olmo2-32B/zero/mj_vot.csv")
# %%
olmo2_mv.columns
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
y1 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()) & (olmo_claims_ann["concat_hate"] == 1), "claims_cw_annA"].reset_index(drop=True)
y2 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()) & (olmo_claims_ann["concat_hate"] == 1), "claims_cw_annB"].reset_index(drop=True)
# %%
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
y1.value_counts()
# %%
y2.value_counts()
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
# %%
# %% [markdown]
# Non HS data
# %%
# %%
y1 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()) & (olmo_claims_ann["concat_hate"] == 0), "claims_cw_annA"].reset_index(drop=True)
y2 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()) & (olmo_claims_ann["concat_hate"] == 0), "claims_cw_annB"].reset_index(drop=True)
y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
# %%
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
y1.value_counts()
# %%
y2.value_counts()
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
# %%
# %%[markdown]
# Overall
# %%
y1 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()), "claims_cw_annA"].reset_index(drop=True)
y2 = olmo_claims_ann.loc[(~olmo_claims_ann["claims_cw_annA"].isna()), "claims_cw_annB"].reset_index(drop=True)
y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
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
from sklearn.metrics import cohen_kappa_score

olmo2_mv = pd.read_csv("./cw_gens/non_hs/Olmo2-32B/zero/mj_vot.csv")
cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]

olmo_vs_rest = {}
for model_name, shot_types in non_hs_runs_df.items():
    olmo_vs_rest[model_name] = {}
    for shot, df in shot_types.items():
        olmo_claims_ann = pd.concat(
            [
                olmo2_mv.loc[
                    ~olmo2_mv[c].isna(), [f"{c}_cw_annB", "concat_hate"]
                ].rename(columns={f"{c}_cw_annB": "claims_cw_annB"}).reset_index(drop=True) for c in cols
            ]
        )
        df_claims_ann = pd.concat(
            [
                df.loc[
                    ~df[c].isna(), [f"{c}_cw_annB", "concat_hate"]
                ].rename(columns={f"{c}_cw_annB": "claims_cw_annB"}).reset_index(drop=True) for c in cols
            ]
        )
        y1_non_hs = olmo_claims_ann.loc[olmo_claims_ann["concat_hate"] == 0, "claims_cw_annB"].reset_index(drop=True)
        y2_non_hs = df_claims_ann.loc[df_claims_ann["concat_hate"] == 0, "claims_cw_annB"].reset_index(drop=True)
        y1_non_hs = y1_non_hs.replace({"USF": "UFS", "UFFS": "UFS"})
        y2_non_hs = y2_non_hs.replace({"USF": "UFS", "UFFS": "UFS"})
        kappa_non_hs = cohen_kappa_score(y1_non_hs, y2_non_hs, weights="linear", labels=["NFS", "UFS", "CFS"])

        y1_hs = olmo_claims_ann.loc[olmo_claims_ann["concat_hate"] == 1, "claims_cw_annB"].reset_index(drop=True)
        y2_hs = df_claims_ann.loc[df_claims_ann["concat_hate"] == 1, "claims_cw_annB"].reset_index(drop=True)
        y1_hs = y1_hs.replace({"USF": "UFS", "UFFS": "UFS"})
        y2_hs = y2_hs.replace({"USF": "UFS", "UFFS": "UFS"})
        kappa_hs = cohen_kappa_score(y1_hs, y2_hs, weights="linear", labels=["NFS", "UFS", "CFS"])
        
        y1 = olmo_claims_ann["claims_cw_annB"].reset_index(drop=True)
        y2 = df_claims_ann["claims_cw_annB"].reset_index(drop=True)
        y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
        y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
        kappa_overall = cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])

        olmo_vs_rest[model_name][shot] = {}
        olmo_vs_rest[model_name][shot]["kappa_non_hs"] = kappa_non_hs
        olmo_vs_rest[model_name][shot]["kappa_hs"] = kappa_hs
        olmo_vs_rest[model_name][shot]["kappa_overall"] = kappa_overall
        
# %%
olmo_vs_rest_df = pd.DataFrame(olmo_vs_rest).T
# %%
olmo_vs_rest_df["zero_kappa_non_hs"] = olmo_vs_rest_df["zero"].apply(lambda t: t["kappa_non_hs"])
olmo_vs_rest_df["zero_kappa_hs"] = olmo_vs_rest_df["zero"].apply(lambda t: t["kappa_hs"])
olmo_vs_rest_df["zero_kappa_overall"] = olmo_vs_rest_df["zero"].apply(lambda t: t["kappa_overall"])

olmo_vs_rest_df["one_kappa_non_hs"] = olmo_vs_rest_df["one"].apply(lambda t: t["kappa_non_hs"])
olmo_vs_rest_df["one_kappa_hs"] = olmo_vs_rest_df["one"].apply(lambda t: t["kappa_hs"])
olmo_vs_rest_df["one_kappa_overall"] = olmo_vs_rest_df["one"].apply(lambda t: t["kappa_overall"])
# %%
pd.DataFrame.from_dict(olmo_vs_rest).T["zero"].apply(lambda )
# %%
olmo_vs_rest_df.to_csv("iaa_olmo_vs_all.csv")
# %%
# %% [markdown]
# Agreement with WSF-ARG+ gold
# %%
import pandas as pd
df = pd.read_csv("/Users/nicola/Documents/projects/cs_hs_misinfo/data/wsf_arg_plus_gold.csv")
# %%
df["conclusion_cw_annA_gold"].value_counts()
# %%
cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]
# %%
df_claims_ann = pd.concat([df[[f"{c}_cw_annA_gold", f"{c}_cw_annB_gold", "concat_hate"]].rename(columns={f"{c}_cw_annA_gold": "claims_cw_annA_gold", f"{c}_cw_annB_gold": "claims_cw_annB_gold"}) for c in cols])
y1 = df_claims_ann.loc[(~df_claims_ann["claims_cw_annA_gold"].isna()) & (df_claims_ann["concat_hate"] == 1), "claims_cw_annA_gold"].reset_index(drop=True)
y2 = df_claims_ann.loc[(~df_claims_ann["claims_cw_annA_gold"].isna()) & (df_claims_ann["concat_hate"] == 1), "claims_cw_annB_gold"].reset_index(drop=True)
# %%
y1.value_counts()
# %%
y2.value_counts()
# %%
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
# %%
y1 = df_claims_ann.loc[(~df_claims_ann["claims_cw_annA_gold"].isna()) & (df_claims_ann["concat_hate"] == 0), "claims_cw_annA_gold"].reset_index(drop=True)
y2 = df_claims_ann.loc[(~df_claims_ann["claims_cw_annA_gold"].isna()) & (df_claims_ann["concat_hate"] == 0), "claims_cw_annB_gold"].reset_index(drop=True)
y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
# %%
y2.isna().sum()
# %%
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
y1.value_counts()
# %%
y2.value_counts()
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
# %%

y1 = df_claims_ann.loc[~df_claims_ann["claims_cw_annA_gold"].isna(), "claims_cw_annA_gold"].reset_index(drop=True)
y2 = df_claims_ann.loc[~df_claims_ann["claims_cw_annA_gold"].isna(), "claims_cw_annB_gold"].reset_index(drop=True)
y1 = y1.replace({"USF": "UFS", "UFFS": "UFS"})
y2 = y2.replace({"USF": "UFS", "UFFS": "UFS"})
# %%
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y1, y2, weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
cohen_kappa_score(y1.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), y2.replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}), weights="linear", labels=["no_cw", "cw"])
# %%
# %%
import pandas as pd

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
best = non_hs_runs_df["Olmo2-32B"]["zero"]
# %%
best["premise2_hate"].isna().sum()
# %%
best["premise0_cw_annA"]
# %%
import pandas as pd
import numpy as np
df = pd.read_csv("/Users/nicola/Documents/projects/cs_hs_misinfo/data/wsf_arg_plus_gold.csv")
df["premise3_hate"] = np.nan
df["premise4_hate"] = np.nan
df["premise5_hate"] = np.nan
df_claims_ann = pd.concat([df[[c, f"{c}_cw_annA_gold", f"{c}_cw_annB_gold", f"{c}_cw_annC_gold", f"{c}_hate"]].rename(columns={c: "claim" ,f"{c}_cw_annA_gold": "claims_cw_annA_gold", f"{c}_cw_annB_gold": "claims_cw_annB_gold", f"{c}_cw_annC_gold": "claims_cw_annC_gold", f"{c}_hate": "claim_hate"}) for c in cols])
df_claims_ann["claims_cw_annA_gold"] = df_claims_ann["claims_cw_annA_gold"].replace({"USF": "UFS", "UFFS": "UFS"})
df_claims_ann["claims_cw_annB_gold"] = df_claims_ann["claims_cw_annB_gold"].replace({"USF": "UFS", "UFFS": "UFS"})
df_claims_ann = df_claims_ann.loc[~df_claims_ann["claims_cw_annA_gold"].isna()]
# %%
import pandas as pd
df = pd.read_csv("/Users/nicola/Documents/projects/cs_hs_misinfo/data/wsf_arg_plus_gold.csv")
df_claims_ann = pd.concat([df[[f"{c}_cw_annA", f"{c}_cw_annB"]].rename(columns={f"{c}_cw_annA": "claims_cw_annA", f"{c}_cw_annB": "claims_cw_annB"}) for c in cols])
df_claims_ann["claims_cw_annA"] = df_claims_ann["claims_cw_annA"].replace({"USF": "UFS", "UFFS": "UFS"})
df_claims_ann["claims_cw_annB"] = df_claims_ann["claims_cw_annB"].replace({"USF": "UFS", "UFFS": "UFS"})
df_claims_ann = df_claims_ann.loc[~df_claims_ann["claims_cw_annA"].isna()]
# %%
import numpy as np

cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]

claims_per_model_dict = {}
for model_name, shot_types, in non_hs_runs.items():
    claims_per_model_dict[model_name] = {}
    for shot in shot_types:
        df_pred = non_hs_runs_df[model_name][shot]
        df_pred["idx"] = df["idx"] # fix idx
        df_pred["premise3_hate"] = np.nan
        df_pred["premise4_hate"] = np.nan
        df_pred["premise5_hate"] = np.nan
        df_concat = pd.concat(
            [df_pred[[c, f"{c}_cw_annA", f"{c}_cw_annB", "concat_hate", f"{c}_hate"]]
             .rename(index={i: f"{i}_{c}" for i in df_pred["idx"].tolist()}, columns={c: "claim", f"{c}_cw_annA": "claims_cw_annA", f"{c}_cw_annB": "claims_cw_annB", f"{c}_hate": "claim_hate"}) for c in cols]
        )
        df_concat["claims_cw_annA"] = df_concat["claims_cw_annA"].replace({"USF": "UFS", "UFFS": "UFS"})
        df_concat["claims_cw_annB"] = df_concat["claims_cw_annB"].replace({"USF": "UFS", "UFFS": "UFS"})
        df_concat = df_concat[~df_concat["claims_cw_annA"].isna()]
        claims_per_model_dict[model_name][shot] = df_concat
# %%
df_needs_judge = claims_per_model_dict["Olmo2-32B"]["zero"].copy()
# %%
df_needs_judge["needs_judge"] = (df_needs_judge["claims_cw_annA"] != df_needs_judge["claims_cw_annB"]).astype(int)
# %%
df_needs_judge = df_needs_judge[["claim", "needs_judge", "concat_hate", "claim_hate"]]
# %%
df_needs_judge
# %%
for model_name, shot_types, in claims_per_model_dict.items():
    for shot in shot_types:
        df = claims_per_model_dict[model_name][shot]
        df_needs_judge[f"{model_name}_{shot}_ann"] = df["claims_cw_annB"]
# %%
df_needs_judge
# %%
df_needs_judge["human_ann1_bo"] = claims_per_model_dict["Olmo2-32B"]["zero"]["claims_cw_annA"]
# %%
df_needs_judge
# %%
pd.crosstab(df_needs_judge["needs_judge"], df_needs_judge["claim_hate"], normalize="index")
# %%
pd.crosstab(df_needs_judge["needs_judge"], df_needs_judge["claim_hate"], margins=True)
# %%
pd.crosstab(df_needs_judge["needs_judge"], df_needs_judge["claim_hate"], margins=True, normalize="index")
# %%
import pandas as pd

cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]

df = pd.read_csv("../data/wsf_arg_plus_gold.csv")
df_claims_ann = pd.concat([df[[f"{c}_cw_annA_gold", f"{c}_cw_annB_gold", f"{c}_cw_annC_gold"]].rename(columns={f"{c}_cw_annA_gold": "claims_cw_annA_gold", f"{c}_cw_annB_gold": "claims_cw_annB_gold", f"{c}_cw_annC_gold": "claims_cw_annC_gold"}) for c in cols])
df_claims_ann = df_claims_ann.loc[~df_claims_ann["claims_cw_annA_gold"].isna()]
# %%
df_needs_judge = df_needs_judge.rename(columns={"needs_judge": "needs_judge_silver"})
# %%
df_needs_judge
# %%
# Silver is Best LLM olmo vs Human ann 1 (BO) 
#
#
#
#
#
# %%
df
# %%
for i, j in enumerate(df["text_ed"].drop_duplicates().index):
    if i != j:
        print(j)
# %%
dropped_id = df_needs_judge["claim"].drop_duplicates().index
# %%
all_id = df_needs_judge["claim"].index
# %%
set(all_id) - set(dropped_id)
# %%
set(all_id)
# %%
import pandas as pd
from sklearn.metrics import cohen_kappa_score

wsf_arg_plus = pd.read_csv("wsf_arg_plus_gold_per_claim.csv")
# %%
# Benjamin vs Greta
# 3 labels
cohen_kappa_score(wsf_arg_plus["human_annA_bo"], wsf_arg_plus["human_annB_gd"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
# 2 labels
cohen_kappa_score(
    wsf_arg_plus["human_annA_bo"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    wsf_arg_plus["human_annB_gd"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])
# %%
# Benjamin vs Davide
# 3 labels
cohen_kappa_score(wsf_arg_plus["human_annA_bo"], wsf_arg_plus["human_annC_dc"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
# 2 labels
cohen_kappa_score(
    wsf_arg_plus["human_annA_bo"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    wsf_arg_plus["human_annC_dc"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])

# %%
# Greta vs Davide
cohen_kappa_score(wsf_arg_plus["human_annB_gd"], wsf_arg_plus["human_annC_dc"], weights="linear", labels=["NFS", "UFS", "CFS"])
# %%
# 2 labels
cohen_kappa_score(
    wsf_arg_plus["human_annB_gd"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    wsf_arg_plus["human_annC_dc"]
        .replace({"NFS": "no_cw", "UFS": "no_cw", "CFS": "cw"}),
    weights="linear",
    labels=["no_cw", "cw"])

# %%
import krippendorff
import numpy as np

ann1_3l = wsf_arg_plus["human_annA_bo"].replace({"NFS": 0, "UFS": 1, "CFS": 2})
ann2_3l = wsf_arg_plus["human_annB_gd"].replace({"NFS": 0, "UFS": 1, "CFS": 2})
ann3_3l = wsf_arg_plus["human_annC_dc"].replace({"NFS": 0, "UFS": 1, "CFS": 2})

ann1_2l = wsf_arg_plus["human_annA_bo"].replace({"NFS": 0, "UFS": 0, "CFS": 1})
ann2_2l = wsf_arg_plus["human_annB_gd"].replace({"NFS": 0, "UFS": 0, "CFS": 1})
ann3_2l = wsf_arg_plus["human_annC_dc"].replace({"NFS": 0, "UFS": 0, "CFS": 1})

reliability_data_3l = np.vstack([ann1_3l, ann2_3l, ann3_3l])
reliability_data_2l = np.vstack([ann1_2l, ann2_2l, ann3_2l])

krippendorff.alpha(reliability_data=reliability_data_3l, level_of_measurement="ordinal")
# %%
krippendorff.alpha(reliability_data=reliability_data_2l, level_of_measurement="ordinal")
# %%
from collections import Counter

def majority_vote(row):
    values = [row[f"human_annA_bo"], row[f"human_annB_gd"], row[f"human_annC_dc"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({"gold_mv": most_common_value, f"gold_mv_agreement_count": count})
    else:
        return pd.Series({"gold_mv": "All unequal", f"gold_mv_agreement_count": 1})


wsf_arg_plus[["gold_mv", "gold_mv_agreement_count"]] = wsf_arg_plus[["human_annA_bo", "human_annB_gd", "human_annC_dc"]].apply(majority_vote, axis=1)
# %%
wsf_arg_plus
# %%
wsf_arg_plus["gold_mv"].value_counts()
# %%
wsf_arg_plus["gold_mv_agreement_count"].value_counts()
# %%
wsf_arg_plus
# %%
# %%
df_claims_ann
# %%
df.columns
# %%
for id in wsf_arg_plus[wsf_arg_plus["gold_mv_agreement_count"] == 1]["Unnamed: 0"].apply(lambda t: t.split("_")[0]).astype(int).to_list():
    print(id)
    df.loc[id, "to_judge"] = 1
# %%
df.to_csv("wsf_arg_plus_gold_to_judge.csv", index=False)
# %%
import pandas as pd

df = pd.read_csv("wsf_arg_plus_gold_to_judge.csv")
# %%
def majority_vote_per_claim(row, col):
    values = [row[f"{col}_cw_annA_gold"], row[f"{col}_cw_annB_gold"], row[f"{col}_cw_annC_gold"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({f"{col}_cw_ann_final_gold": most_common_value, f"{col}_agreement_count": count})
    else:
        return pd.Series({f"{col}_cw_ann_final_gold": "All unequal", f"{col}_agreement_count": 1})
# %%
cols = ["premise0",
        "premise1",
        "premise2",
        "premise3",
        "premise4",
        "premise5",
        "conclusion"]

for c in cols:
    df[[f"{c}_cw_ann_final_gold", f"{c}_agreement_count"]] = df.apply(lambda row: majority_vote_per_claim(row, c), axis=1)
# %%
df["conclusion_cw_ann_final_gold"].value_counts()
# %%
df.to_csv("wsf_arg_gold_mv.csv", index=False)
# %%
df
# %%
wsf_arg_plus["needs_judge_gold"] = (wsf_arg_plus["human_annA_bo"] != wsf_arg_plus["human_annB_gd"]).astype(int)
# %%
wsf_arg_plus["needs_judge_gold_strict"] = (wsf_arg_plus["gold_mv"] == "All unequal").astype(int)
# %%
# small llms
# medium llms
# large llms
# best + worst
# best + 2nd best
# olmo family
# mistral family
# llama family

small = ["Mistral-7B", "Llama-8B", "Olmo2-7B", "Command-r-7B", "Mixtral-8x7B"]
medium = ["Mistral-22B", "Olmo2-32B", "Mixtral-8x22B"]
large = ["Llama-70B", "Command-r-104B", "Qwen2.5-72B"]
best_worst = ["Olmo2-32B", "Llama-8B"]
best_and_2ndbest = ["Olmo2-32B", "Mixtral-8x7B"]
olmo_family = ["Olmo2-7B", "Olmo2-32B"]
mistral_family = ["Mistral-7B", "Mixtral-8x7B", "Mistral-22B"]
llama_family = ["Llama-8B", "Llama-70B"]
commandr_family = ["Command-r-7B", "Command-r-104B"]
qwen_family = ["Qwen2.5-7B", "Qwen2.5-72B"]

runs = [
    (small, "small"),
    (medium, "medium"),
    (large, "large"),
    (best_worst, "best_worst"),
    (best_and_2ndbest, "best_and_2ndbest"),
    (olmo_family, "olmo_family"),
    (mistral_family, "mistral_family"),
    (llama_family, "llama_family"),
    (commandr_family, "commandr_family"),
    (qwen_family, "qwen_family"),
]
# %%
wsf_arg_plus
# %%
from collections import Counter

def majority_vote_general(row, cols):
    values = [row[c] for c in cols]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count <= 1:
        return pd.Series({"mv": "All unequal", f"mv_agreement_count": count})
    else:
        return pd.Series({"mv": most_common_value, f"mv_agreement_count": count})
# %%
from sklearn.metrics import precision_recall_fscore_support
res = {}
for rs, r_name in runs:
    for shot in ["zero", "one"]:
        r_cols = [f"{c}_{shot}_ann" for c in rs]
        df_pred = wsf_arg_plus.apply(lambda row: majority_vote_general(row, r_cols), axis=1)
        wsf_arg_plus[f"needs_judge_{r_name}_{shot}"] = (df_pred["mv"] == "All unequal").astype(int)
        p_silver, r_silver, f1_silver, _ = precision_recall_fscore_support(wsf_arg_plus["needs_judge_silver"], wsf_arg_plus[f"needs_judge_{r_name}_{shot}"], average="macro")
        p_gold, r_gold, f1_gold, _ = precision_recall_fscore_support(wsf_arg_plus["needs_judge_gold"], wsf_arg_plus[f"needs_judge_{r_name}_{shot}"], average="macro")
        p_gold_strict, r_gold_strict, f1_gold_strict, _ = precision_recall_fscore_support(wsf_arg_plus["needs_judge_gold_strict"], wsf_arg_plus[f"needs_judge_{r_name}_{shot}"], average="macro")
        res[f"{r_name}_{shot}"] = {}
        res[f"{r_name}_{shot}"]["p_silver"] = p_silver
        res[f"{r_name}_{shot}"]["r_silver"] = r_silver
        res[f"{r_name}_{shot}"]["f1_silver"] = f1_silver
        res[f"{r_name}_{shot}"]["p_gold"] = p_gold
        res[f"{r_name}_{shot}"]["r_gold"] = r_gold
        res[f"{r_name}_{shot}"]["f1_gold"] = f1_gold
        res[f"{r_name}_{shot}"]["p_gold_strict"] = p_gold_strict
        res[f"{r_name}_{shot}"]["r_gold_strict"] = r_gold_strict
        res[f"{r_name}_{shot}"]["f1_gold_strict"] = f1_gold_strict

#majority_vote_general()
# %%
wsf_arg_plus
# %%
pd.DataFrame(res).T.to_csv("LLM_as_judge_predictors.csv")
# %%
