# %%
import pandas as pd

df = pd.read_csv("../data/wsf_arg_extension_with_nohs_cw.csv")
# %%
cols_non_hs = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]
cols_hs = ["premise0", "premise1", "premise2", "conclusion"]
pd.crosstab(
    pd.concat([df.loc[(df["concat_hate"] == 1) & (~df[f"{c}_cw_final"].isna()), f"{c}_cw_final"] for c in cols_hs]).reset_index(drop=True),
    pd.concat([df.loc[(df["concat_hate"] == 1) & (~df[f"{c}_cw_final"].isna()), f"{c}_hate"] for c in cols_hs]).reset_index(drop=True), margins=True
)
# %%
cols_non_hs = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]
cols_hs = ["premise0", "premise1", "premise2", "conclusion"]
pd.crosstab(
    pd.concat([df.loc[(df["concat_hate"] == 0) & (~df[f"{c}_cw_final"].isna()), f"{c}_cw_final"] for c in cols_non_hs]).reset_index(drop=True),
    pd.concat([df.loc[(df["concat_hate"] == 0) & (~df[f"{c}_cw_final"].isna()), f"{c}_hate"] for c in cols_non_hs]).reset_index(drop=True), margins=True
)
# %%
pd.concat([df.loc[(df["concat_hate"] == 0) & (~df[f"{c}_cw_final"].isna()), f"{c}_cw_final"] for c in cols_non_hs]).reset_index(drop=True).value_counts()
# %%
df["premise0_cw_final"] = df[["premise0_cw_annA", "premise0_cw_annB", "premise0_cw_final"]].apply(lambda t: t["premise0_cw_annA"] if t["premise0_cw_annA"] == t["premise0_cw_annB"] else t["premise0_cw_final"], axis=1)
df["premise1_cw_final"] = df[["premise1_cw_annA", "premise1_cw_annB", "premise1_cw_final"]].apply(lambda t: t["premise1_cw_annA"] if t["premise1_cw_annA"] == t["premise1_cw_annB"] else t["premise1_cw_final"], axis=1)
df["premise2_cw_final"] = df[["premise2_cw_annA", "premise2_cw_annB", "premise2_cw_final"]].apply(lambda t: t["premise2_cw_annA"] if t["premise2_cw_annA"] == t["premise2_cw_annB"] else t["premise2_cw_final"], axis=1)
df["premise3_cw_final"] = df[["premise3_cw_annA", "premise3_cw_annB", "premise3_cw_final"]].apply(lambda t: t["premise3_cw_annA"] if t["premise3_cw_annA"] == t["premise3_cw_annB"] else t["premise3_cw_final"], axis=1)
df["premise4_cw_final"] = df[["premise4_cw_annA", "premise4_cw_annB", "premise4_cw_final"]].apply(lambda t: t["premise4_cw_annA"] if t["premise4_cw_annA"] == t["premise4_cw_annB"] else t["premise4_cw_final"], axis=1)
df["premise5_cw_final"] = df[["premise5_cw_annA", "premise5_cw_annB", "premise5_cw_final"]].apply(lambda t: t["premise5_cw_annA"] if t["premise5_cw_annA"] == t["premise5_cw_annB"] else t["premise5_cw_final"], axis=1)
df["conclusion_cw_final"] = df[["conclusion_cw_annA", "conclusion_cw_annB", "conclusion_cw_final"]].apply(lambda t: t["conclusion_cw_annA"] if t["conclusion_cw_annA"] == t["conclusion_cw_annB"] else t["conclusion_cw_final"], axis=1)
# %%
pd.concat([
df["premise0_cw_final"],
df["premise1_cw_final"],
df["premise2_cw_final"],
df["premise3_cw_final"],
df["premise4_cw_final"],
df["premise5_cw_final"]])
# %%
df_hate = df[df["concat_hate"] == 1]
df_non_hate = df[df["concat_hate"] == 0]
# %%
pd.crosstab(df_hate["premise0_hate"], df_hate["premise0_cw_final"]) + pd.crosstab(df_hate["premise1_hate"], df_hate["premise1_cw_final"]) + pd.crosstab(df_hate["premise2_hate"], df_hate["premise2_cw_final"])
# %%
pd.crosstab(df_hate["conclusion_hate"], df_hate["conclusion_cw_final"])
# %%
df["concat_hate"].value_counts()
# %%
df_hate["conclusion"]
# %%
(
    (~df_hate["premise0"].isna()).sum() +
    (~df_hate["premise1"].isna()).sum() +
    (~df_hate["premise2"].isna()).sum()
)
# %%
def count_arg(arg):
    if isinstance(arg, str):
        if arg.strip() != "":
            return 1
    return 0

df_hate.apply(lambda row: count_arg(row["premise0"]) + count_arg(row["premise1"]) + count_arg(row["premise2"]), axis=1).describe()
# %%
df_hate.apply(lambda row: count_arg(row["premise0"]) + count_arg(row["premise1"]) + count_arg(row["premise2"]) + count_arg(row["conclusion"]), axis=1).describe()
# %%
df_non_hate["use_claims_only"].value_counts()
# %%
(
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise0"].isna()).sum() +
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise1"].isna()).sum() +
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise2"].isna()).sum() +
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise3"].isna()).sum() +
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise4"].isna()).sum() +
    (~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise5"].isna()).sum()
)
# %%
(~df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "conclusion"].isna()).sum()
# %%
(
    (~df_non_hate["premise0"].isna()).sum() +
    (~df_non_hate["premise1"].isna()).sum() +
    (~df_non_hate["premise2"].isna()).sum() +
    (~df_non_hate["premise3"].isna()).sum() +
    (~df_non_hate["premise4"].isna()).sum() +
    (~df_non_hate["premise5"].isna()).sum() +
    (~df_non_hate["conclusion"].isna()).sum()
)
# %%
df_non_hate.apply(
    lambda row: (
        count_arg(row["premise0"]) +
        count_arg(row["premise1"]) +
        count_arg(row["premise2"]) +
        count_arg(row["premise3"]) +
        count_arg(row["premise4"]) +
        count_arg(row["premise5"]) +
        count_arg(row["conclusion"])), axis=1).describe()
# %%
df_non_hate[df_non_hate["use_claims_only"] != "yes"].apply(
    lambda row: (
        count_arg(row["premise0"]) +
        count_arg(row["premise1"]) +
        count_arg(row["premise2"]) +
        count_arg(row["premise3"]) +
        count_arg(row["premise4"]) +
        count_arg(row["premise5"])), axis=1).describe()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise0_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise1_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise2_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise3_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise4_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "premise5_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] != "yes", "conclusion_cw_final"].value_counts()
# %%
# %%
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise0_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise1_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise2_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise3_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise4_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "premise5_cw_final"].value_counts()
# %%
df_non_hate.loc[df_non_hate["use_claims_only"] == "yes", "conclusion_cw_final"].value_counts()

# %%
df_hate_dis = df_hate[df_hate["premise0_cw_annA"] != df_hate["premise0_cw_annB"]]

#df_hate.loc[df_hate["premise0_cw_annA"] != df_hate["premise0_cw_annB"], "premise0_cw_final"]
# %%
cols = ["premise0", "premise1", "premise2"]

all_p_counts = {}
all_p_counts["nof_agreement_with_human"] = 0
all_p_counts["nof_agreement_with_llm"] = 0
all_p_counts["nof_agreement_with_none"] = 0
all_p_counts["nof_premises_dis"] = 0
for c in cols:
    df_hate_dis = df_hate[
        (~df_hate[f"{c}"].isna()) &
        (df_hate[f"{c}_cw_annA"] != df_hate[f"{c}_cw_annB"])]
    all_p_counts["nof_premises_dis"] += len(df_hate_dis)
    all_p_counts["nof_agreement_with_human"] += (df_hate_dis[f"{c}_cw_annA"] == df_hate_dis[f"{c}_cw_final"]).sum()
    all_p_counts["nof_agreement_with_llm"] += (df_hate_dis[f"{c}_cw_annB"] == df_hate_dis[f"{c}_cw_final"]).sum()
    all_p_counts["nof_agreement_with_none"] += (
        (df_hate_dis[f"{c}_cw_annA"] != df_hate_dis[f"{c}_cw_final"]) &
        (df_hate_dis[f"{c}_cw_annB"] != df_hate_dis[f"{c}_cw_final"])
    ).sum()

all_p_counts["percent_agreement_with_human"] = all_p_counts["nof_agreement_with_human"] / all_p_counts["nof_premises_dis"]
all_p_counts["percent_agreement_with_llm"] = all_p_counts["nof_agreement_with_llm"] / all_p_counts["nof_premises_dis"]
all_p_counts["percent_agreement_with_none"] = all_p_counts["nof_agreement_with_none"] / all_p_counts["nof_premises_dis"]
# %%
pd.Series(all_p_counts)
# %%
all_c_counts = {}
df_hate_conclusion = df_hate[
    (~df_hate["conclusion"].isna()) &
    (df_hate["conclusion_cw_annA"] != df_hate["conclusion_cw_annB"])
]

all_c_counts["nof_agreement_with_human"] = (df_hate_conclusion["conclusion_cw_annA"] == df_hate_conclusion["conclusion_cw_final"]).sum()
all_c_counts["nof_agreement_with_llm"] = (df_hate_conclusion["conclusion_cw_annB"] == df_hate_conclusion["conclusion_cw_final"]).sum()
all_c_counts["nof_agreement_with_none"] = (
    (df_hate_conclusion["conclusion_cw_annA"] != df_hate_conclusion["conclusion_cw_final"]) &
    (df_hate_conclusion["conclusion_cw_annB"] != df_hate_conclusion["conclusion_cw_final"])
).sum()
all_c_counts["nof_conclusion_dis"] = len(df_hate_conclusion)

all_c_counts["percent_agreement_with_human"] = all_c_counts["nof_agreement_with_human"] / all_c_counts["nof_conclusion_dis"]
all_c_counts["percent_agreement_with_llm"] = all_c_counts["nof_agreement_with_llm"] / all_c_counts["nof_conclusion_dis"]
all_c_counts["percent_agreement_with_none"] = all_c_counts["nof_agreement_with_none"] / all_c_counts["nof_conclusion_dis"]
# %%
pd.Series(all_c_counts)
# %%
cols = ["premise0", "premise1", "premise2", "conclusion"]

all_arg_counts = {}
all_arg_counts["nof_agreement_with_human"] = 0
all_arg_counts["nof_agreement_with_llm"] = 0
all_arg_counts["nof_agreement_with_none"] = 0
all_arg_counts["nof_args_dis"] = 0
for c in cols:
    df_hate_dis = df_hate[
        (~df_hate[f"{c}"].isna()) &
        (df_hate[f"{c}_cw_annA"] != df_hate[f"{c}_cw_annB"])]
    all_arg_counts["nof_args_dis"] += len(df_hate_dis)
    all_arg_counts["nof_agreement_with_human"] += (df_hate_dis[f"{c}_cw_annA"] == df_hate_dis[f"{c}_cw_final"]).sum()
    all_arg_counts["nof_agreement_with_llm"] += (df_hate_dis[f"{c}_cw_annB"] == df_hate_dis[f"{c}_cw_final"]).sum()
    all_arg_counts["nof_agreement_with_none"] += (
        (df_hate_dis[f"{c}_cw_annA"] != df_hate_dis[f"{c}_cw_final"]) &
        (df_hate_dis[f"{c}_cw_annB"] != df_hate_dis[f"{c}_cw_final"])
    ).sum()

all_arg_counts["percent_agreement_with_human"] = all_arg_counts["nof_agreement_with_human"] / all_arg_counts["nof_args_dis"]
all_arg_counts["percent_agreement_with_llm"] = all_arg_counts["nof_agreement_with_llm"] / all_arg_counts["nof_args_dis"]
all_arg_counts["percent_agreement_with_none"] = all_arg_counts["nof_agreement_with_none"] / all_arg_counts["nof_args_dis"]
# %%
pd.Series(all_arg_counts)
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5"]

all_p_counts_nonhs = {}
all_p_counts_nonhs["nof_agreement_with_human"] = 0
all_p_counts_nonhs["nof_agreement_with_llm"] = 0
all_p_counts_nonhs["nof_agreement_with_none"] = 0
all_p_counts_nonhs["nof_args_dis"] = 0
for c in cols:
    df_non_hate_dis = df_non_hate[
        (~df_non_hate[f"{c}"].isna()) &
        (df_non_hate["use_claims_only"] != "yes") &
        (df_non_hate[f"{c}_cw_annA"] != df_non_hate[f"{c}_cw_annB"])]
    all_p_counts_nonhs["nof_args_dis"] += len(df_non_hate_dis)
    all_p_counts_nonhs["nof_agreement_with_human"] += (df_non_hate_dis[f"{c}_cw_annA"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_p_counts_nonhs["nof_agreement_with_llm"] += (df_non_hate_dis[f"{c}_cw_annB"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_p_counts_nonhs["nof_agreement_with_none"] += (
        (df_non_hate_dis[f"{c}_cw_annA"] != df_non_hate_dis[f"{c}_cw_final"]) &
        (df_non_hate_dis[f"{c}_cw_annB"] != df_non_hate_dis[f"{c}_cw_final"])
    ).sum()

all_p_counts_nonhs["percent_agreement_with_human"] = all_p_counts_nonhs["nof_agreement_with_human"] / all_p_counts_nonhs["nof_args_dis"]
all_p_counts_nonhs["percent_agreement_with_llm"] = all_p_counts_nonhs["nof_agreement_with_llm"] / all_p_counts_nonhs["nof_args_dis"]
all_p_counts_nonhs["percent_agreement_with_none"] = all_p_counts_nonhs["nof_agreement_with_none"] / all_p_counts_nonhs["nof_args_dis"]
# %%
pd.Series(all_p_counts_nonhs)
# %%
cols = ["conclusion"]

all_c_counts_nonhs = {}
all_c_counts_nonhs["nof_agreement_with_human"] = 0
all_c_counts_nonhs["nof_agreement_with_llm"] = 0
all_c_counts_nonhs["nof_agreement_with_none"] = 0
all_c_counts_nonhs["nof_args_dis"] = 0
for c in cols:
    df_non_hate_dis = df_non_hate[
        (~df_non_hate[f"{c}"].isna()) &
        (df_non_hate["use_claims_only"] != "yes") &
        (df_non_hate[f"{c}_cw_annA"] != df_non_hate[f"{c}_cw_annB"])]
    all_c_counts_nonhs["nof_args_dis"] += len(df_non_hate_dis)
    all_c_counts_nonhs["nof_agreement_with_human"] += (df_non_hate_dis[f"{c}_cw_annA"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_c_counts_nonhs["nof_agreement_with_llm"] += (df_non_hate_dis[f"{c}_cw_annB"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_c_counts_nonhs["nof_agreement_with_none"] += (
        (df_non_hate_dis[f"{c}_cw_annA"] != df_non_hate_dis[f"{c}_cw_final"]) &
        (df_non_hate_dis[f"{c}_cw_annB"] != df_non_hate_dis[f"{c}_cw_final"])
    ).sum()

all_c_counts_nonhs["percent_agreement_with_human"] = all_c_counts_nonhs["nof_agreement_with_human"] / all_c_counts_nonhs["nof_args_dis"]
all_c_counts_nonhs["percent_agreement_with_llm"] = all_c_counts_nonhs["nof_agreement_with_llm"] / all_c_counts_nonhs["nof_args_dis"]
all_c_counts_nonhs["percent_agreement_with_none"] = all_c_counts_nonhs["nof_agreement_with_none"] / all_c_counts_nonhs["nof_args_dis"]
# %%
pd.Series(all_c_counts_nonhs)
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

all_args_counts_nonhs = {}
all_args_counts_nonhs["nof_agreement_with_human"] = 0
all_args_counts_nonhs["nof_agreement_with_llm"] = 0
all_args_counts_nonhs["nof_agreement_with_none"] = 0
all_args_counts_nonhs["nof_args_dis"] = 0
for c in cols:
    df_non_hate_dis = df_non_hate[
        (~df_non_hate[f"{c}"].isna()) &
        (df_non_hate["use_claims_only"] != "yes") &
        (df_non_hate[f"{c}_cw_annA"] != df_non_hate[f"{c}_cw_annB"])]
    all_args_counts_nonhs["nof_args_dis"] += len(df_non_hate_dis)
    all_args_counts_nonhs["nof_agreement_with_human"] += (df_non_hate_dis[f"{c}_cw_annA"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_args_counts_nonhs["nof_agreement_with_llm"] += (df_non_hate_dis[f"{c}_cw_annB"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_args_counts_nonhs["nof_agreement_with_none"] += (
        (df_non_hate_dis[f"{c}_cw_annA"] != df_non_hate_dis[f"{c}_cw_final"]) &
        (df_non_hate_dis[f"{c}_cw_annB"] != df_non_hate_dis[f"{c}_cw_final"])
    ).sum()

all_args_counts_nonhs["percent_agreement_with_human"] = all_args_counts_nonhs["nof_agreement_with_human"] / all_args_counts_nonhs["nof_args_dis"]
all_args_counts_nonhs["percent_agreement_with_llm"] = all_args_counts_nonhs["nof_agreement_with_llm"] / all_args_counts_nonhs["nof_args_dis"]
all_args_counts_nonhs["percent_agreement_with_none"] = all_args_counts_nonhs["nof_agreement_with_none"] / all_args_counts_nonhs["nof_args_dis"]
# %%
pd.Series(all_args_counts_nonhs)
# %%
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

all_args_counts_w_onlyc_nonhs = {}
all_args_counts_w_onlyc_nonhs["nof_agreement_with_human"] = 0
all_args_counts_w_onlyc_nonhs["nof_agreement_with_llm"] = 0
all_args_counts_w_onlyc_nonhs["nof_agreement_with_none"] = 0
all_args_counts_w_onlyc_nonhs["nof_args_dis"] = 0
for c in cols:
    df_non_hate_dis = df_non_hate[
        (~df_non_hate[f"{c}"].isna()) &
        #(df_non_hate["use_claims_only"] != "yes") &
        (df_non_hate[f"{c}_cw_annA"] != df_non_hate[f"{c}_cw_annB"])]
    all_args_counts_w_onlyc_nonhs["nof_args_dis"] += len(df_non_hate_dis)
    all_args_counts_w_onlyc_nonhs["nof_agreement_with_human"] += (df_non_hate_dis[f"{c}_cw_annA"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_args_counts_w_onlyc_nonhs["nof_agreement_with_llm"] += (df_non_hate_dis[f"{c}_cw_annB"] == df_non_hate_dis[f"{c}_cw_final"]).sum()
    all_args_counts_w_onlyc_nonhs["nof_agreement_with_none"] += (
        (df_non_hate_dis[f"{c}_cw_annA"] != df_non_hate_dis[f"{c}_cw_final"]) &
        (df_non_hate_dis[f"{c}_cw_annB"] != df_non_hate_dis[f"{c}_cw_final"])
    ).sum()

all_args_counts_w_onlyc_nonhs["percent_agreement_with_human"] = all_args_counts_w_onlyc_nonhs["nof_agreement_with_human"] / all_args_counts_w_onlyc_nonhs["nof_args_dis"]
all_args_counts_w_onlyc_nonhs["percent_agreement_with_llm"] = all_args_counts_w_onlyc_nonhs["nof_agreement_with_llm"] / all_args_counts_w_onlyc_nonhs["nof_args_dis"]
all_args_counts_w_onlyc_nonhs["percent_agreement_with_none"] = all_args_counts_w_onlyc_nonhs["nof_agreement_with_none"] / all_args_counts_w_onlyc_nonhs["nof_args_dis"]
# %%
pd.Series(all_args_counts_w_onlyc_nonhs)
# %%
