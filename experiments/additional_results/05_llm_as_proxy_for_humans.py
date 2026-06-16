# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
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




mistral = {
	"Mistral-7B": ["zero","one"],
	"Mistral-22B": ["zero", "one"],
}
mixtral = {
	"Mixtral-8x22B": ["zero", "one"],
	"Mixtral-8x7B": ["zero", "one"],
}
llama = {
    "Llama-8B": ["zero", "one"],
	"Llama-70B": ["zero", "one"],
}
olmo = {
    "Olmo2-7B": ["zero", "one"],
	"Olmo2-32B": ["zero", "one"],
}
qwen = {
    "Qwen2.5-7B": ["zero", "one"],
	"Qwen2.5-72B": ["zero", "one"],
}
commandr = {
    "Command-r-7B": ["zero", "one"],
    "Command-r-104B": ["zero", "one"],
}

all_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in all.items() for shot in shots]
small_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in small.items() for shot in shots]
medium_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in medium.items() for shot in shots]
large_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in large.items() for shot in shots]
mistral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mistral.items() for shot in shots]
mixtral_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in mixtral.items() for shot in shots]
llama_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in llama.items() for shot in shots]
olmo_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in olmo.items() for shot in shots]
qwen_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in qwen.items() for shot in shots]
commandr_flatten = [f"{model_name}_{shot}_claim_cw" for model_name, shots in commandr.items() for shot in shots]
# %%
# %%
from sklearn.metrics import precision_recall_fscore_support

results_f1 = {}
for r in all_flatten:
    y_pred = df[r]
    y_true = df["claim_cw_platinum"]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])
    
    res = {
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
    }
    results_f1[r] = res
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
all_mv = df.apply(lambda row: majority_vote(row, all_flatten, "all"), axis=1)
small_mv = df.apply(lambda row: majority_vote(row, small_flatten, "small"), axis=1)
medium_mv = df.apply(lambda row: majority_vote(row, medium_flatten, "medium"), axis=1)
large_mv = df.apply(lambda row: majority_vote(row, large_flatten, "large"), axis=1)


mistral_mv = df.apply(lambda row: majority_vote(row, mistral_flatten, "mistral"), axis=1)
mixtral_mv = df.apply(lambda row: majority_vote(row, mixtral_flatten, "mixtral"), axis=1)
llama_mv = df.apply(lambda row: majority_vote(row, llama_flatten, "llama"), axis=1)
olmo_mv = df.apply(lambda row: majority_vote(row, olmo_flatten, "olmo"), axis=1)
qwen_mv = df.apply(lambda row: majority_vote(row, qwen_flatten, "qwen"), axis=1)
commandr_mv = df.apply(lambda row: majority_vote(row, commandr_flatten, "commandr"), axis=1)

y_pred_all = all_mv["all"]
y_pred_small = small_mv["small"]
y_pred_medium = medium_mv["medium"]
y_pred_large = large_mv["large"]
y_pred_mistral = mistral_mv["mistral"]
y_pred_mixtral = mixtral_mv["mixtral"]
y_pred_llama = llama_mv["llama"]
y_pred_olmo = olmo_mv["olmo"]
y_pred_qwen = qwen_mv["qwen"]
y_pred_commandr = commandr_mv["commandr"]
y_true = df["claim_cw_platinum"]
p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(y_true,
                                                                            y_pred_all,
                                                                            average='macro',
                                                                            labels=["NFS", "UFS", "CFS"])
p_macro_small, r_macro_small, f1_macro_small, _ = precision_recall_fscore_support(y_true,
                                                                                  y_pred_small,
                                                                                  average='macro',
                                                                                  labels=["NFS", "UFS", "CFS"])
p_macro_medium, r_macro_medium, f1_macro_medium, _ = precision_recall_fscore_support(y_true,
                                                                                     y_pred_medium,
                                                                                     average='macro',
                                                                                     labels=["NFS", "UFS", "CFS"])
p_macro_large, r_macro_large, f1_macro_large, _ = precision_recall_fscore_support(y_true,
                                                                                  y_pred_large,
                                                                                  average='macro',
                                                                                  labels=["NFS", "UFS", "CFS"])
p_macro_mistral, r_macro_mistral, f1_macro_mistral, _ = precision_recall_fscore_support(y_true,
                                                                                  y_pred_mistral,
                                                                                  average='macro',
                                                                                  labels=["NFS", "UFS", "CFS"])
p_macro_mixtral, r_macro_mixtral, f1_macro_mixtral, _ = precision_recall_fscore_support(y_true,
                                                                                  y_pred_mixtral,
                                                                                  average='macro',
                                                                                  labels=["NFS", "UFS", "CFS"])
p_macro_llama, r_macro_llama, f1_macro_llama, _ = precision_recall_fscore_support(y_true,
                                                                                  y_pred_llama,
                                                                                  average='macro',
                                                                                  labels=["NFS", "UFS", "CFS"])
p_macro_olmo, r_macro_olmo, f1_macro_olmo, _ = precision_recall_fscore_support(y_true,
                                                                               y_pred_olmo,
                                                                               average='macro',
                                                                               labels=["NFS", "UFS", "CFS"])
p_macro_qwen, r_macro_qwen, f1_macro_qwen, _ = precision_recall_fscore_support(y_true,
                                                                               y_pred_qwen,
                                                                               average='macro',
                                                                               labels=["NFS", "UFS", "CFS"])
p_macro_commandr, r_macro_commandr, f1_macro_commandr, _ = precision_recall_fscore_support(y_true,
                                                                                           y_pred_commandr,
                                                                                           average='macro',
                                                                                           labels=["NFS", "UFS", "CFS"])
results_f1["all_mv"] = {
    "p_macro": p_macro_all,
    "r_macro": r_macro_all,
    "f1_macro": f1_macro_all,
}
results_f1["small_mv"] = {
    "p_macro": p_macro_small,
    "r_macro": r_macro_small,
    "f1_macro": f1_macro_small,
}
results_f1["medium_mv"] = {
    "p_macro": p_macro_medium,
    "r_macro": r_macro_medium,
    "f1_macro": f1_macro_medium,
}
results_f1["large_mv"] = {
    "p_macro": p_macro_large,
    "r_macro": r_macro_large,
    "f1_macro": f1_macro_large,
}
results_f1["mistral_mv"] = {
    "p_macro": p_macro_mistral,
    "r_macro": r_macro_mistral,
    "f1_macro": f1_macro_mistral,
}
results_f1["mixtral_mv"] = {
    "p_macro": p_macro_mixtral,
    "r_macro": r_macro_mixtral,
    "f1_macro": f1_macro_mixtral,
}
results_f1["llama_mv"] = {
    "p_macro": p_macro_llama,
    "r_macro": r_macro_llama,
    "f1_macro": f1_macro_llama,
}
results_f1["olmo_mv"] = {
    "p_macro": p_macro_olmo,
    "r_macro": r_macro_olmo,
    "f1_macro": f1_macro_olmo,
}
results_f1["qwen_mv"] = {
    "p_macro": p_macro_qwen,
    "r_macro": r_macro_qwen,
    "f1_macro": f1_macro_qwen,
}
results_f1["commandr_mv"] = {
    "p_macro": p_macro_commandr,
    "r_macro": r_macro_commandr,
    "f1_macro": f1_macro_commandr,
}
# %%
pd.DataFrame(results_f1).T.round(3)
# %%
# Mean and std of the runs when compared with platinum labels using F1-score.
# %%
results_f1_disagg = {}
for r in all_flatten:
    for i in range(3): # We predicted cw labels 3 times per run.
        y_pred = df[f"{r}_m{i}"]
        y_true = df["claim_cw_platinum"]

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["NFS", "UFS", "CFS"])

        res = {
            "p_macro": p_macro,
            "r_macro": r_macro,
            "f1_macro": f1_macro,
        }
        results_f1_disagg[f"{r}_m{i}"] = res
# %%
df_f1_res_disagg = pd.DataFrame(results_f1_disagg).T
# %%
df_f1_res_disagg = df_f1_res_disagg.reset_index().rename(columns={"index": "run"})
df_f1_res_disagg["model_name"] = df_f1_res_disagg["run"].apply(lambda r: "_".join(r.split("_")[:-1]))
# %%
(
    df_f1_res_disagg[["model_name", "p_macro", "r_macro", "f1_macro"]]
    .groupby("model_name")
    .agg(["mean", "std"])
    .loc[all_flatten]
    .round(3)
)
# %%
