# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
df_gold = pd.read_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv")
# %%
df_platinum = pd.read_csv("../../data/wsf_arg_plus_per_claim_platinum_disagg.csv")
# %%
all = {
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
small = {
	"Mistral-7B": ["zero","one"],
	"Llama-8B": ["zero", "one"],
	"Olmo2-7B": ["zero", "one"],
	"Qwen2.5-7B": ["zero", "one"],
	"Command-r-7B": ["zero", "one"]
}
medium = {
	"Mixtral-8x7B": ["zero", "one"],
	"Mistral-22B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
	"Mixtral-8x22B": ["zero", "one"],

}
large = {
	"Llama-70B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
	"Command-r-104B": ["zero", "one"],
}

olmo = {
    "Olmo2-7B": ["zero", "one"],
    "Olmo2-32B": ["zero", "one"],
}

mistral = {
    "Mistral-7B": ["zero","one"],
    "Mistral-22B": ["zero", "one"],
}

llama = {
    "Llama-8B": ["zero", "one"],
    "Llama-70B": ["zero", "one"],
}

qwen = {
    "Qwen2.5-7B": ["zero", "one"],
    "Qwen2.5-72B": ["zero", "one"],
}

mixtral = {
    "Mixtral-8x7B": ["zero", "one"],
    "Mixtral-8x22B": ["zero", "one"],
}

commandr = {
    "Command-r-7B": ["zero", "one"],
    "Command-r-104B": ["zero", "one"],
}

all_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in all.items() for shot in shots]
small_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in small.items() for shot in shots]
medium_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in medium.items() for shot in shots]
large_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in large.items() for shot in shots]
# %%
olmo_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in olmo.items() for shot in shots]
mistral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mistral.items() for shot in shots]
llama_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in llama.items() for shot in shots]
qwen_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in qwen.items() for shot in shots]
mixtral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mixtral.items() for shot in shots]
commandr_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in commandr.items() for shot in shots]
# %%
from collections import Counter

def majority_vote(row, cols, group_name):
    values = row[cols].tolist()
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({group_name: most_common_value, f"{group_name}_agreement_count": count})
    else:
        return pd.Series({group_name: "All unequal", f"{group_name}_agreement_count": 1})
# %%
llm_loop_all = df.apply(lambda row: majority_vote(row, all_flatten, "all"), axis=1)
# %%
llm_loop_all_label = []
for i in range(len(llm_loop_all)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_all.iloc[i]["all"]:
        llm_loop_all_label.append(llm_loop_all.iloc[i]["all"])
    else:
        llm_loop_all_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
#llm_loop_all.loc[df_gold["claim_cw_annA_gold"] == llm_loop_all["all"], "claim_cw_all"] ==
# %%
llm_loop_all["claim_cw_all"] = llm_loop_all_label
# %%
# %%
llm_loop_small = df.apply(lambda row: majority_vote(row, small_flatten, "small"), axis=1)
# %%
llm_loop_small_label = []
for i in range(len(llm_loop_small)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_small.iloc[i]["small"]:
        llm_loop_small_label.append(llm_loop_small.iloc[i]["small"])
    else:
        llm_loop_small_label.append(df_platinum.iloc[i]["claim_cw_platinum"])

# %%
llm_loop_small["claim_cw_small"] = llm_loop_small_label
# %%
# %%
llm_loop_medium = df.apply(lambda row: majority_vote(row, medium_flatten, "medium"), axis=1)
# %%
llm_loop_medium_label = []
for i in range(len(llm_loop_medium)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_medium.iloc[i]["medium"]:
        llm_loop_medium_label.append(llm_loop_medium.iloc[i]["medium"])
    else:
        llm_loop_medium_label.append(df_platinum.iloc[i]["claim_cw_platinum"])

# %%
llm_loop_medium["claim_cw_medium"] = llm_loop_medium_label
# %%
# %%
llm_loop_large = df.apply(lambda row: majority_vote(row, large_flatten, "large"), axis=1)
# %%
llm_loop_large_label = []
for i in range(len(llm_loop_large)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_large.iloc[i]["large"]:
        llm_loop_large_label.append(llm_loop_large.iloc[i]["large"])
    else:
        llm_loop_large_label.append(df_platinum.iloc[i]["claim_cw_platinum"])

# %%
llm_loop_large["claim_cw_large"] = llm_loop_large_label
# %%
res = {}
# %%
from sklearn.metrics import cohen_kappa_score

res["iaa_all_1ann"] = cohen_kappa_score(llm_loop_all["all"],
                                        df_gold["claim_cw_annA_gold"],
                                        labels=["NFS", "UFS", "CFS"],
                                        weights="linear")
# %%
res["iaa_all_platinum"] = cohen_kappa_score(llm_loop_all["claim_cw_all"],
                                            df_platinum["claim_cw_platinum"],
                                            labels=["NFS", "UFS", "CFS"],
                                            weights="linear")
# %%
res["iaa_small_1ann"] = cohen_kappa_score(llm_loop_small["small"],
                                          df_gold["claim_cw_annA_gold"],
                                          labels=["NFS", "UFS", "CFS"],
                                          weights="linear")
# %%
res["iaa_small_platinum"] = cohen_kappa_score(llm_loop_small["claim_cw_small"],
                                              df_platinum["claim_cw_platinum"],
                                              labels=["NFS", "UFS", "CFS"],
                                              weights="linear")
# %%
res["iaa_medium_1ann"] = cohen_kappa_score(llm_loop_medium["medium"],
                                          df_gold["claim_cw_annA_gold"],
                                          labels=["NFS", "UFS", "CFS"],
                                          weights="linear")
# %%
res["iaa_medium_platinum"] = cohen_kappa_score(llm_loop_medium["claim_cw_medium"],
                                              df_platinum["claim_cw_platinum"],
                                              labels=["NFS", "UFS", "CFS"],
                                              weights="linear")
# %%
res["iaa_large_1ann"] = cohen_kappa_score(llm_loop_large["large"],
                                          df_gold["claim_cw_annA_gold"],
                                          labels=["NFS", "UFS", "CFS"],
                                          weights="linear")
# %%
res["iaa_large_platinum"] = cohen_kappa_score(llm_loop_large["claim_cw_large"],
                                              df_platinum["claim_cw_platinum"],
                                              labels=["NFS", "UFS", "CFS"],
                                              weights="linear")
# %%
pd.Series(res).round(3)
# %%
res_percent = {}
# %% 
res_percent["all_agree_%"] = (llm_loop_all["all"] == df_gold["claim_cw_annA_gold"]).sum() / len(llm_loop_all)
res_percent["small_agree_%"] = (llm_loop_small["small"] == df_gold["claim_cw_annA_gold"]).sum() / len(llm_loop_small)
res_percent["medium_agree_%"] = (llm_loop_medium["medium"] == df_gold["claim_cw_annA_gold"]).sum() / len(llm_loop_medium)
res_percent["large_agree_%"] = (llm_loop_large["large"] == df_gold["claim_cw_annA_gold"]).sum() / len(llm_loop_large)
# %%
pd.Series(res_percent).round(3)
# %%
res_to_judge = {}
# %%
res_to_judge["all_needs_judge"] = (llm_loop_all["all"] != df_gold["claim_cw_annA_gold"]).sum()
res_to_judge["small_needs_judge"] = (llm_loop_small["small"] != df_gold["claim_cw_annA_gold"]).sum()
res_to_judge["medium_needs_judge"] = (llm_loop_medium["medium"] != df_gold["claim_cw_annA_gold"]).sum()
res_to_judge["large_needs_judge"] = (llm_loop_large["large"] != df_gold["claim_cw_annA_gold"]).sum()
res_to_judge["olmo_needs_judge"] = (df_gold["claim_needed_judge_gold"] == 1).sum()
res_to_judge["all_needs_judge_%"] = (llm_loop_all["all"] != df_gold["claim_cw_annA_gold"]).sum() / len(df_gold) * 100
res_to_judge["small_needs_judge_%"] = (llm_loop_small["small"] != df_gold["claim_cw_annA_gold"]).sum() / len(df_gold) * 100
res_to_judge["medium_needs_judge_%"] = (llm_loop_medium["medium"] != df_gold["claim_cw_annA_gold"]).sum() / len(df_gold) * 100
res_to_judge["large_needs_judge_%"] = (llm_loop_large["large"] != df_gold["claim_cw_annA_gold"]).sum() / len(df_gold) * 100
res_to_judge["olmo_needs_judge_%"] = (df_gold["claim_needed_judge_gold"] == 1).sum() / len(df_gold) * 100
# %%
pd.Series(res_to_judge).round(2)
# %%
cohen_kappa_score(llm_loop_all["all"],
                  df_platinum["claim_cw_annA_platinum"],
                  labels=["NFS", "UFS", "CFS"],
                  weights="linear")
# %%
cohen_kappa_score(llm_loop_all["all"],
                  df_platinum["claim_cw_annB_platinum"],
                  labels=["NFS", "UFS", "CFS"],
                  weights="linear")
# %%
cohen_kappa_score(llm_loop_all["all"],
                  df_platinum["claim_cw_annC_platinum"],
                  labels=["NFS", "UFS", "CFS"],
                  weights="linear")
# %%
agreement_per_ann = {}
agreement_per_ann["mv_all_annA"] = cohen_kappa_score(llm_loop_all["all"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_all_annB"] = cohen_kappa_score(llm_loop_all["all"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_all_annC"] = cohen_kappa_score(llm_loop_all["all"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_all_platinum"] = cohen_kappa_score(llm_loop_all["all"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_small_annA"] = cohen_kappa_score(llm_loop_small["small"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_small_annB"] = cohen_kappa_score(llm_loop_small["small"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_small_annC"] = cohen_kappa_score(llm_loop_small["small"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_small_platinum"] = cohen_kappa_score(llm_loop_small["small"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_medium_annA"] = cohen_kappa_score(llm_loop_medium["medium"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_medium_annB"] = cohen_kappa_score(llm_loop_medium["medium"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_medium_annC"] = cohen_kappa_score(llm_loop_medium["medium"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_medium_platinum"] = cohen_kappa_score(llm_loop_medium["medium"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_large_annA"] = cohen_kappa_score(llm_loop_large["large"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_large_annB"] = cohen_kappa_score(llm_loop_large["large"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_large_annC"] = cohen_kappa_score(llm_loop_large["large"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["mv_large_platinum"] = cohen_kappa_score(llm_loop_large["large"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")


agreement_per_ann["llm_loop_all_annA"] = cohen_kappa_score(llm_loop_all["claim_cw_all"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_all_annB"] = cohen_kappa_score(llm_loop_all["claim_cw_all"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_all_annC"] = cohen_kappa_score(llm_loop_all["claim_cw_all"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_small_annA"] = cohen_kappa_score(llm_loop_small["claim_cw_small"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_small_annB"] = cohen_kappa_score(llm_loop_small["claim_cw_small"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_small_annC"] = cohen_kappa_score(llm_loop_small["claim_cw_small"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_medium_annA"] = cohen_kappa_score(llm_loop_medium["claim_cw_medium"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_medium_annB"] = cohen_kappa_score(llm_loop_medium["claim_cw_medium"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_medium_annC"] = cohen_kappa_score(llm_loop_medium["claim_cw_medium"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_large_annA"] = cohen_kappa_score(llm_loop_large["claim_cw_large"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_large_annB"] = cohen_kappa_score(llm_loop_large["claim_cw_large"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann["llm_loop_large_annC"] = cohen_kappa_score(llm_loop_large["claim_cw_large"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
# %%
pd.Series(agreement_per_ann).round(3)
# %%
# %%
llm_loop_small = df.apply(lambda row: majority_vote(row, small_flatten, "small"), axis=1)
# %%
llm_loop_small_label = []
for i in range(len(llm_loop_small)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_small.iloc[i]["small"]:
        llm_loop_small_label.append(llm_loop_small.iloc[i]["small"])
    else:
        llm_loop_small_label.append(df_platinum.iloc[i]["claim_cw_platinum"])

# %%
llm_loop_small["claim_cw_small"] = llm_loop_small_label

# %%
llm_loop_olmo = df.apply(lambda row: majority_vote(row, olmo_flatten, "olmo"), axis=1)
llm_loop_olmo_label = []
for i in range(len(llm_loop_olmo)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_olmo.iloc[i]["olmo"]:
        llm_loop_olmo_label.append(llm_loop_olmo.iloc[i]["olmo"])
    else:
        llm_loop_olmo_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_olmo["claim_cw_olmo"] = llm_loop_olmo_label
# %%
llm_loop_mistral = df.apply(lambda row: majority_vote(row, mistral_flatten, "mistral"), axis=1)
llm_loop_mistral_label = []
for i in range(len(llm_loop_mistral)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_mistral.iloc[i]["mistral"]:
        llm_loop_mistral_label.append(llm_loop_mistral.iloc[i]["mistral"])
    else:
        llm_loop_mistral_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_mistral["claim_cw_mistral"] = llm_loop_mistral_label
# %%
llm_loop_qwen = df.apply(lambda row: majority_vote(row, qwen_flatten, "qwen"), axis=1)
llm_loop_qwen_label = []
for i in range(len(llm_loop_qwen)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_qwen.iloc[i]["qwen"]:
        llm_loop_qwen_label.append(llm_loop_qwen.iloc[i]["qwen"])
    else:
        llm_loop_qwen_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_qwen["claim_cw_qwen"] = llm_loop_qwen_label
# %%
llm_loop_llama = df.apply(lambda row: majority_vote(row, llama_flatten, "llama"), axis=1)
llm_loop_llama_label = []
for i in range(len(llm_loop_llama)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_llama.iloc[i]["llama"]:
        llm_loop_llama_label.append(llm_loop_llama.iloc[i]["llama"])
    else:
        llm_loop_llama_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_llama["claim_cw_llama"] = llm_loop_llama_label
# %%
llm_loop_mixtral = df.apply(lambda row: majority_vote(row, mixtral_flatten, "mixtral"), axis=1)
llm_loop_mixtral_label = []
for i in range(len(llm_loop_mixtral)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_mixtral.iloc[i]["mixtral"]:
        llm_loop_mixtral_label.append(llm_loop_mixtral.iloc[i]["mixtral"])
    else:
        llm_loop_mixtral_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_mixtral["claim_cw_mixtral"] = llm_loop_mixtral_label
# %%
llm_loop_commandr = df.apply(lambda row: majority_vote(row, commandr_flatten, "commandr"), axis=1)
llm_loop_commandr_label = []
for i in range(len(llm_loop_commandr)):
    if df_gold.iloc[i]["claim_cw_annA_gold"] == llm_loop_commandr.iloc[i]["commandr"]:
        llm_loop_commandr_label.append(llm_loop_commandr.iloc[i]["commandr"])
    else:
        llm_loop_commandr_label.append(df_platinum.iloc[i]["claim_cw_platinum"])
llm_loop_commandr["claim_cw_commandr"] = llm_loop_commandr_label
# %%
agreement_per_ann_family = {}

agreement_per_ann_family["mv_olmo_annA"] = cohen_kappa_score(llm_loop_olmo["olmo"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_olmo_annB"] = cohen_kappa_score(llm_loop_olmo["olmo"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_olmo_annC"] = cohen_kappa_score(llm_loop_olmo["olmo"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_olmo_platinum"] = cohen_kappa_score(llm_loop_olmo["olmo"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mistral_annA"] = cohen_kappa_score(llm_loop_mistral["mistral"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mistral_annB"] = cohen_kappa_score(llm_loop_mistral["mistral"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mistral_annC"] = cohen_kappa_score(llm_loop_mistral["mistral"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mistral_platinum"] = cohen_kappa_score(llm_loop_mistral["mistral"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_qwen_annA"] = cohen_kappa_score(llm_loop_qwen["qwen"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_qwen_annB"] = cohen_kappa_score(llm_loop_qwen["qwen"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_qwen_annC"] = cohen_kappa_score(llm_loop_qwen["qwen"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_qwen_platinum"] = cohen_kappa_score(llm_loop_qwen["qwen"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_llama_annA"] = cohen_kappa_score(llm_loop_llama["llama"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_llama_annB"] = cohen_kappa_score(llm_loop_llama["llama"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_llama_annC"] = cohen_kappa_score(llm_loop_llama["llama"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_llama_platinum"] = cohen_kappa_score(llm_loop_llama["llama"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mixtral_annA"] = cohen_kappa_score(llm_loop_mixtral["mixtral"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mixtral_annB"] = cohen_kappa_score(llm_loop_mixtral["mixtral"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mixtral_annC"] = cohen_kappa_score(llm_loop_mixtral["mixtral"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_mixtral_platinum"] = cohen_kappa_score(llm_loop_mixtral["mixtral"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_commandr_annA"] = cohen_kappa_score(llm_loop_commandr["commandr"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_commandr_annB"] = cohen_kappa_score(llm_loop_commandr["commandr"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_commandr_annC"] = cohen_kappa_score(llm_loop_commandr["commandr"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["mv_commandr_platinum"] = cohen_kappa_score(llm_loop_commandr["commandr"], df_platinum["claim_cw_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")


agreement_per_ann_family["llm_loop_olmo_annA"] = cohen_kappa_score(llm_loop_olmo["claim_cw_olmo"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_olmo_annB"] = cohen_kappa_score(llm_loop_olmo["claim_cw_olmo"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_olmo_annC"] = cohen_kappa_score(llm_loop_olmo["claim_cw_olmo"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mistral_annA"] = cohen_kappa_score(llm_loop_mistral["claim_cw_mistral"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mistral_annB"] = cohen_kappa_score(llm_loop_mistral["claim_cw_mistral"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mistral_annC"] = cohen_kappa_score(llm_loop_mistral["claim_cw_mistral"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_qwen_annA"] = cohen_kappa_score(llm_loop_qwen["claim_cw_qwen"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_qwen_annB"] = cohen_kappa_score(llm_loop_qwen["claim_cw_qwen"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_qwen_annC"] = cohen_kappa_score(llm_loop_qwen["claim_cw_qwen"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_llama_annA"] = cohen_kappa_score(llm_loop_llama["claim_cw_llama"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_llama_annB"] = cohen_kappa_score(llm_loop_llama["claim_cw_llama"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_llama_annC"] = cohen_kappa_score(llm_loop_llama["claim_cw_llama"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mixtral_annA"] = cohen_kappa_score(llm_loop_mixtral["claim_cw_mixtral"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mixtral_annB"] = cohen_kappa_score(llm_loop_mixtral["claim_cw_mixtral"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_mixtral_annC"] = cohen_kappa_score(llm_loop_mixtral["claim_cw_mixtral"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_commandr_annA"] = cohen_kappa_score(llm_loop_commandr["claim_cw_commandr"], df_platinum["claim_cw_annA_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_commandr_annB"] = cohen_kappa_score(llm_loop_commandr["claim_cw_commandr"], df_platinum["claim_cw_annB_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
agreement_per_ann_family["llm_loop_commandr_annC"] = cohen_kappa_score(llm_loop_commandr["claim_cw_commandr"], df_platinum["claim_cw_annC_platinum"], labels=["NFS", "UFS", "CFS"], weights="linear")
# %%
pd.Series(agreement_per_ann_family).round(3)

# %%
