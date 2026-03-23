# %%[markdown]
# ## Percent Agreement
# This notebook reproduces the results of Table 2 of the paper.
# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_all_llms.csv")
# %%
claim_gold_annA = pd.read_csv("../../data/wsf_arg_plus_per_claim_gold_disagg.csv")[["concat_hate", "claim_cw_annA_gold"]]
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
claim_gold_annA_hs = claim_gold_annA[claim_gold_annA["concat_hate"] == 1]
claim_gold_annA_non_hs = claim_gold_annA[claim_gold_annA["concat_hate"] == 0]
llm_mv_hs = df[df["concat_hate"] == 1]
llm_mv_non_hs = df[df["concat_hate"] == 0]
percent_agreement_tab = {}
for r in runs:
    percent_agreement_tab[r] = {}
    percent_agreement_tab[r]["all_claims_hs"] = (claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]).sum() / len(claim_gold_annA_hs["claim_cw_annA_gold"])
    percent_agreement_tab[r]["CFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "CFS")
    percent_agreement_tab[r]["UFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "UFS")
    percent_agreement_tab[r]["NFS_hs"] = ((claim_gold_annA_hs["claim_cw_annA_gold"] == llm_mv_hs[f"{r}_claim_cw"]) & (claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_hs["claim_cw_annA_gold"] == "NFS")

    percent_agreement_tab[r]["all_claims_non_hs"] = (claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]).sum() / len(claim_gold_annA_non_hs["claim_cw_annA_gold"])
    percent_agreement_tab[r]["CFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "CFS")
    percent_agreement_tab[r]["UFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "UFS")
    percent_agreement_tab[r]["NFS_non_hs"] = ((claim_gold_annA_non_hs["claim_cw_annA_gold"] == llm_mv_non_hs[f"{r}_claim_cw"]) & (claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")).sum() / sum(claim_gold_annA_non_hs["claim_cw_annA_gold"] == "NFS")
# %%
pd.DataFrame(percent_agreement_tab).T
# %%
