# %%
import pandas as pd
import numpy as np

df = pd.read_csv("../../data/wsf_arg_plus_raw.csv")
# %% [markdown]
# ## WSF-ARG+ message-level
# %%
df.columns
# %%
df["premise3_hate"] = np.nan
df["premise4_hate"] = np.nan
df["premise5_hate"] = np.nan
# %%
cols = [
    "premise0",
    "premise1",
    "premise2",
    "premise3",
    "premise4",
    "premise5",
    "conclusion"
]
# %%
for c in cols:
    df = df.rename(columns={f"{c}_cw_annA": f"{c}_cw_annA_llm_loop",
                            f"{c}_cw_annB": f"{c}_cw_annB_llm_loop",
                            f"{c}_cw_final": f"{c}_cw_llm_loop",
                            })
    df = df.rename(columns={f"{c}_cw_ann_final_gold": f"{c}_cw_gold"})
# %%
df.columns
# %%
from collections import Counter

def majority_vote(row, col):
    values = [row[f"{col}_cw_annA_llm_loop"], row[f"{col}_cw_annB_llm_loop"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({f"{col}_cw_llm_loop_aux": most_common_value, f"{col}_agreement_count_llm_loop": count})
    else:
        return pd.Series({f"{col}_cw_llm_loop_aux": "All unequal", f"{col}_agreement_count_llm_loop": 1})
# %%
for col in cols:
    df[[f"{col}_cw_llm_loop_aux", f"{col}_agreement_count_llm_loop"]] = df.apply(lambda row: majority_vote(row, col), axis=1)
    df.loc[df[col].isna(), f"{col}_agreement_count_llm_loop"] = np.nan
# %%
df.loc[df[f"{col}_cw_llm_loop_aux"] == "All unequal", f"{col}_cw_llm_loop"]
# %%
for col in cols:
    #df.loc[df[f"{col}_cw_llm_loop_aux"] == "All unequal", f"{col}_cw_llm_loop"] = df.loc[df[f"{col}_cw_llm_loop_aux"] == "All unequal", f"{col}_cw_llm_loop"]
    df.loc[df[f"{col}_cw_llm_loop_aux"] != "All unequal", f"{col}_cw_llm_loop"] = df.loc[df[f"{col}_cw_llm_loop_aux"] != "All unequal", f"{col}_cw_llm_loop_aux"]
# %%
for col in cols:
    df = df.drop(columns=[f"{col}_cw_llm_loop_aux"])
# %%
df = df.drop(columns=["to_judge_gold", "are_there_diferences", "target_ok", "arg_comps_ok"])
# %%
for col in cols:
    df.loc[(df[f"{col}_agreement_count_gold"] == 1) & (df[f"{col}_agreement_count_gold"].notna()), f"{col}_needed_judge_gold"] = 1
    df.loc[(df[f"{col}_agreement_count_gold"] != 1) & (df[f"{col}_agreement_count_gold"].notna()), f"{col}_needed_judge_gold"] = 0

    df.loc[(df[f"{col}_agreement_count_llm_loop"] == 1) & (df[f"{col}_agreement_count_llm_loop"].notna()), f"{col}_needed_judge_llm_loop"] = 1
    df.loc[(df[f"{col}_agreement_count_llm_loop"] != 1) & (df[f"{col}_agreement_count_llm_loop"].notna()), f"{col}_needed_judge_llm_loop"] = 0
# %%
df.columns
# %%
df["premise2_needed_judge_llm_loop"].isna().sum()
# %%
df["premise2_agreement_count_llm_loop"].isna().sum()
# %%
df[f"premise2_agreement_count_gold"].isna().sum()
# %%
df[f"premise2_needed_judge_gold"].isna().sum()
# %%
df["conclusion_agreement_count_llm_loop"].isna().sum()
# %%
df["conclusion_agreement_count_gold"].isna().sum()
# %%
df.columns
# %%
rel_cols = [
    "file_id",
    "idx",
    "text_ed",
    "IS",
    "weak",
    "concat",
    "concat_hate",
    "is_argument",
    "premise0",
    "premise0_hate",
    "premise0_cw_llm_loop",
    "premise0_cw_gold",
    "premise0_needed_judge_llm_loop",
    "premise0_needed_judge_gold",
    "premise1",
    "premise1_hate",
    "premise1_cw_llm_loop",
    "premise1_cw_gold",
    "premise1_needed_judge_llm_loop",
    "premise1_needed_judge_gold",
    "premise2",
    "premise2_hate",
    "premise2_cw_llm_loop",
    "premise2_cw_gold",
    "premise2_needed_judge_llm_loop",
    "premise2_needed_judge_gold",
    "premise3",
    "premise3_hate",
    "premise3_cw_llm_loop",
    "premise3_cw_gold",
    "premise3_needed_judge_llm_loop",
    "premise3_needed_judge_gold",
    "premise4",
    "premise4_hate",
    "premise4_cw_llm_loop",
    "premise4_cw_gold",
    "premise4_needed_judge_llm_loop",
    "premise4_needed_judge_gold",
    "premise5",
    "premise5_hate",
    "premise5_cw_llm_loop",
    "premise5_cw_gold",
    "premise5_needed_judge_llm_loop",
    "premise5_needed_judge_gold",
    "conclusion",
    "conclusion_hate",
    "conclusion_cw_llm_loop",
    "conclusion_cw_gold",
    "conclusion_needed_judge_llm_loop",
    "conclusion_needed_judge_gold",
]
# %%
df.columns = [c.replace("gold", "platinum") for c in df.columns]
df.columns = [c.replace("llm_loop", "gold") for c in df.columns]
# %%
rel_cols = [c.replace("gold", "platinum") for c in rel_cols]
rel_cols = [c.replace("llm_loop", "gold") for c in rel_cols]
# %%
# %%
df[rel_cols].to_csv("../../data/wsf_arg_plus_per_message.csv", index=False)
# %% [markdown]
# ## WSF-ARG+ claim-level
# %%
df_claims = pd.concat(
        #[df[["file_id", "idx", "text_ed", "IS", "weak", "concat", "concat_hate"]]] + # list 1
        [df[[c, "concat_hate", f"{c}_hate", f"{c}_cw_gold", f"{c}_cw_platinum", f"{c}_needed_judge_gold", f"{c}_needed_judge_platinum", f"{c}_agreement_count_platinum", f"{c}_agreement_count_gold"]].rename(
            index={i: f"{i}_{c}" for i in df["idx"].tolist()},
            columns={c: "claim",
                     f"{c}_hate": "claim_hate",
                     f"{c}_cw_gold": "claim_cw_gold",
                     f"{c}_cw_platinum": "claim_cw_platinum",
                     f"{c}_needed_judge_gold": "claim_needed_judge_gold",
                     f"{c}_needed_judge_platinum": "claim_needed_judge_platinum",
                     f"{c}_agreement_count_gold": "claim_agreement_count_gold",
                     f"{c}_agreement_count_platinum": "claim_agreement_count_platinum",
                     }) for c in cols], axis=0)  # list 2
df_claims = df_claims[df_claims["claim_cw_platinum"].notna()]
# %%
df_claims = df_claims.reset_index().rename(columns={"index": "claim_idx"})
# %%
df_claims.to_csv("../../data/wsf_arg_plus_per_claim.csv", index=False)
# %% [markdown]
# ## WSF-ARG+ each of the annotations (disaggregated)
# %%
df[
    ["file_id", "idx", "text_ed", "IS", "weak", "concat", "concat_hate"] +
    [f"{col}_cw_annA_gold" for col in cols] +
    [f"{col}_cw_annB_gold" for col in cols] +
    [f"{col}_cw_gold" for col in cols] +
    [f"{col}_needed_judge_gold" for col in cols]
].to_csv("../../data/wsf_arg_plus_per_message_gold_disagg.csv", index=False)
# %%
df[
    ["file_id", "idx", "text_ed", "IS", "weak", "concat", "concat_hate"] +
    [f"{col}_cw_annA_platinum" for col in cols] +
    [f"{col}_cw_annB_platinum" for col in cols] +
    [f"{col}_cw_annC_platinum" for col in cols] +
    [f"{col}_cw_platinum" for col in cols] +
    [f"{col}_needed_judge_platinum" for col in cols]
].to_csv("../../data/wsf_arg_plus_per_message_platinum_disagg.csv", index=False)
# %%
df_claims_gold_disagg = pd.concat(
        [df[[c, "concat_hate", f"{c}_hate", f"{c}_cw_annA_gold", f"{c}_cw_annB_gold", f"{c}_cw_gold", f"{c}_needed_judge_gold", f"{c}_agreement_count_gold"]].rename(
            index={i: f"{i}_{c}" for i in df["idx"].tolist()},
            columns={c: "claim",
                     f"{c}_hate": "claim_hate",
                     f"{c}_cw_annA_gold": "claim_cw_annA_gold",
                     f"{c}_cw_annB_gold": "claim_cw_annB_gold",
                     f"{c}_cw_gold": "claim_cw_gold",
                     f"{c}_needed_judge_gold": "claim_needed_judge_gold",
                     f"{c}_agreement_count_gold": "claim_agreement_count_gold",
                     }) for c in cols], axis=0)
df_claims_gold_disagg = df_claims_gold_disagg[df_claims_gold_disagg["claim_cw_gold"].notna()]
df_claims_gold_disagg = df_claims_gold_disagg.reset_index().rename(columns={"index": "claim_idx"})
df_claims_gold_disagg.to_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv", index=False)
# %%
df_claims_platinum_disagg = pd.concat(
        [df[[c, "concat_hate", f"{c}_hate", f"{c}_cw_annA_platinum", f"{c}_cw_annB_platinum", f"{c}_cw_annC_platinum", f"{c}_cw_platinum", f"{c}_needed_judge_platinum", f"{c}_agreement_count_platinum"]].rename(
            index={i: f"{i}_{c}" for i in df["idx"].tolist()},
            columns={c: "claim",
                     f"{c}_hate": "claim_hate",
                     f"{c}_cw_annA_platinum": "claim_cw_annA_platinum",
                     f"{c}_cw_annB_platinum": "claim_cw_annB_platinum",
                     f"{c}_cw_annC_platinum": "claim_cw_annC_platinum",
                     f"{c}_cw_platinum": "claim_cw_platinum",
                     f"{c}_needed_judge_platinum": "claim_needed_judge_platinum",
                     f"{c}_agreement_count_platinum": "claim_agreement_count_platinum",
                     }) for c in cols], axis=0)
df_claims_platinum_disagg = df_claims_platinum_disagg[df_claims_platinum_disagg["claim_cw_platinum"].notna()]
df_claims_platinum_disagg = df_claims_platinum_disagg.reset_index().rename(columns={"index": "claim_idx"})
df_claims_platinum_disagg.to_csv("../../data/wsf_arg_plus_per_claim_platinum_disagg.csv", index=False)
# %% [markdown]
# ## All LLM predictions
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
# %%
from itertools import chain

df_all_llms = df[["file_id", "idx", "text_ed", "concat_hate"]].copy()
# Reading new dfs
runs_df = {}
for model_name, shot_types in runs.items():
    runs_df[model_name] = {}
    for shot in shot_types:
        runs_df[model_name][shot] = pd.read_csv(f"../../data/raw_generations/check-worthiness/non_hs/{model_name}/{shot}/mj_vot.csv")
        preds = list(chain.from_iterable([[f"{col}_cw_annB_m0", f"{col}_cw_annB_m1", f"{col}_cw_annB_m2", f"{col}_cw_annB"] for col in cols]))
        df_all_llms[[f"{model_name}_{shot}_{col_pred}" for col_pred in preds]] = runs_df[model_name][shot][preds]
        df_all_llms.columns = [col.replace("_annB", "") for col in df_all_llms.columns]
# %%
runs_df["Olmo2-32B"]["zero"]
# %%
df_all_llms.to_csv("../../data/wsf_arg_plus_per_message_all_llms.csv", index=False)
# %%
# %% [markdown]
# ## All LLM predictions per claim
#%%
# df.loc[:, [f"{k}_{v}_{c}_cw_m0", f"{k}_{v}_{c}_cw_m1", f"{k}_{v}_{c}_cw_m2", f"{k}_{v}_{c}_cw"]]
# %%
df_claims_all_llms = pd.read_csv("../../data/wsf_arg_plus_per_claim.csv")
# %%
for k, values in runs.items():
    for v in values:
        df_claims_per_llm = [
            df_all_llms[[f"{k}_{v}_{c}_cw_m0", f"{k}_{v}_{c}_cw_m1", f"{k}_{v}_{c}_cw_m2", f"{k}_{v}_{c}_cw"]].rename(
                index={i: f"{i}_{c}" for i in df["idx"].tolist()},
                columns={
                    f"{k}_{v}_{c}_cw_m0": f"{k}_{v}_claim_cw_m0",
                    f"{k}_{v}_{c}_cw_m1": f"{k}_{v}_claim_cw_m1",
                    f"{k}_{v}_{c}_cw_m2": f"{k}_{v}_claim_cw_m2",
                    f"{k}_{v}_{c}_cw": f"{k}_{v}_claim_cw"})
            for c in cols]
        df_claims_per_llm = pd.concat(df_claims_per_llm, axis=0)
        df_claims_per_llm[df_claims_per_llm[f"{k}_{v}_claim_cw"].notna()]
        df_claims_per_llm = df_claims_per_llm.reset_index().rename(columns={"index": "claim_idx"})
        df_claims_all_llms[[f"{k}_{v}_claim_cw_m0", f"{k}_{v}_claim_cw_m1", f"{k}_{v}_claim_cw_m2", f"{k}_{v}_claim_cw"]] = df_claims_per_llm[[f"{k}_{v}_claim_cw_m0", f"{k}_{v}_claim_cw_m1", f"{k}_{v}_claim_cw_m2", f"{k}_{v}_claim_cw"]]
# %%
df_claims_all_llms.to_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv", index=False)
# %%
