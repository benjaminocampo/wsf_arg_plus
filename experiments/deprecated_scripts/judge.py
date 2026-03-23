# %%
import pandas as pd

df = pd.read_csv("../data/wsf_misinformation_check-worthiness_validated_by_mariana.csv")
# %%
schema = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "description": "One label per argument component, in order.",
            "items": {
                "type": "string",
                "enum": ["hateful", "not hateful"]
            },
            "minItems": 2,
            "maxItems": 5
        }
    },
    "required": ["labels"]
}
import json
json.dumps(schema)
# %%
df["premise0_final_cw"]
# %%
df["premise0_final_cw"] = df["annA_premise0"]
df["premise1_final_cw"] = df["annA_premise1"]
df["premise2_final_cw"] = df["annA_premise2"]
df["conclusion_final_cw"] = df["annA_conclusion"]

df.loc[~df["annC_premise0_cw_final"].isna(), "premise0_final_cw"] = df.loc[~df["annC_premise0_cw_final"].isna(), "annC_premise0_cw_final"]
df.loc[~df["annC_premise1_cw_final"].isna(), "premise1_final_cw"] = df.loc[~df["annC_premise1_cw_final"].isna(), "annC_premise1_cw_final"]
df.loc[~df["annC_premise2_cw_final"].isna(), "premise2_final_cw"] = df.loc[~df["annC_premise2_cw_final"].isna(), "annC_premise2_cw_final"]
df.loc[~df["annC_conclusion_cw_final"].isna(), "conclusion_final_cw"] = df.loc[~df["annC_conclusion_cw_final"].isna(), "annC_conclusion_cw_final"]
# %%
repl = {
        "NFS": "Non-Factual",
        "UFS": "Unimportant Factual",
        "CFS": "Check-worthy Factual"
    }
df["premise0_final_cw"] = df["premise0_final_cw"].replace(repl)
df["premise1_final_cw"] = df["premise1_final_cw"].replace(repl)
df["premise2_final_cw"] = df["premise2_final_cw"].replace(repl)
df["conclusion_final_cw"] = df["conclusion_final_cw"].replace(repl)
# %%
import math
arg_cols = ["premise0", "premise1", "premise2", "conclusion"]
arg_comps = []
for _, row in df.iterrows():
    ss = ""
    for c in arg_cols:
        if ((not isinstance(row[c], str)) and (math.isnan(row[c]))):
            continue
        ss = ss + f"{c} [{row[f'{c}_final_cw']}]: {row[c]}\n"
    arg_comps.append(ss)
# %%
cw_label = pd.concat([df["premise0_final_cw"], df["premise1_final_cw"], df["premise2_final_cw"], df["conclusion_final_cw"]])
cw_label.name = "cw_label"
cw_label.reset_index(drop=True)
# %%
cw_label[~cw_label.isna()]
# %%
hate_label = pd.concat([df["premise0_hate"], df["premise1_hate"], df["premise2_hate"], df["conclusion_hate"]])
hate_label.name = "hate_label"
hate_label.reset_index(drop=True)
# %%
hate_label[~cw_label.isna()]
# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import matthews_corrcoef

# Assuming you already have cw_label and hate_label
# Filter to keep only non-null in cw_label (and same positions for hate_label)
mask = ~cw_label.isna()
cw = cw_label[mask].reset_index(drop=True)
hate = hate_label[mask].reset_index(drop=True)

# --- 1. Cramér’s V + Chi-square test ---
contingency = pd.crosstab(cw, hate.replace({1: "HS", 0: "Non-HS"}))

chi2, p, dof, expected = chi2_contingency(contingency)
n = contingency.sum().sum()
phi2 = chi2 / n
r, k = contingency.shape
cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))

print("=== Cramér’s V and Chi² test ===")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p:.4e}")
print(f"Cramér’s V: {cramers_v:.4f}")

# --- 2. Matthews Correlation Coefficient (MCC) ---
mcc = matthews_corrcoef(cw, hate.replace({1: "HS", 0: "Non-HS"}))
print("\n=== Matthews Correlation Coefficient ===")
print(f"MCC: {mcc:.4f}")




# %%
df["annA_premise0"] == df["annC_premise0_cw_final"]


# df["annA_premise1"]
# df["annB_premise1"]
# df["annC_premise1_cw_final"]
# 
# df["annA_premise2"]
# df["annB_premise2"]
# df["annC_premise2_cw_final"]
# 
# df["annA_conclusion"]
# df["annB_conclusion"]
# df["annC_conclusion_cw_final"]
# %%
#pd.crosstab(df["annA_premise0"], df["annC_premise0_cw_final"])
(df["annA_premise0"] == df["annC_premise0_cw_final"]).sum()
# %%
premise0_judge_df = df[~df["annC_premise0_cw_final"].isna()]

agg1 = (premise0_judge_df["annA_premise0"] == premise0_judge_df["annC_premise0_cw_final"]).sum() / len(premise0_judge_df)
agg2 = (premise0_judge_df["annB_premise0"] == premise0_judge_df["annC_premise0_cw_final"]).sum() / len(premise0_judge_df)
# %%
agg1
# %%
agg2
# %%
1 - (agg1 + agg2)
# %%
premise1_judge_df = df[~df["annC_premise1_cw_final"].isna()]

agg1 = (premise1_judge_df["annA_premise1"] == premise1_judge_df["annC_premise1_cw_final"]).sum() / len(premise1_judge_df)
agg2 = (premise1_judge_df["annB_premise1"] == premise1_judge_df["annC_premise1_cw_final"]).sum() / len(premise1_judge_df)
# %%
agg1
# %%
agg2
# %%
1 - (agg1 + agg2)
# %%
premise2_judge_df = df[~df["annC_premise2_cw_final"].isna()]

agg1 = (premise2_judge_df["annA_premise2"] == premise2_judge_df["annC_premise2_cw_final"]).sum() / len(premise2_judge_df)
agg2 = (premise2_judge_df["annB_premise2"] == premise2_judge_df["annC_premise2_cw_final"]).sum() / len(premise2_judge_df)
# %%
agg1
# %%
agg2
# %%
1 - (agg1 + agg2)

# %%
conclusion_judge_df = df[~df["annC_conclusion_cw_final"].isna()]

agg1 = (conclusion_judge_df["annA_conclusion"] == conclusion_judge_df["annC_conclusion_cw_final"]).sum() / len(conclusion_judge_df)
agg2 = (conclusion_judge_df["annB_conclusion"] == conclusion_judge_df["annC_conclusion_cw_final"]).sum() / len(conclusion_judge_df)
# %%
agg1
# %%
agg2
# %%
1 - (agg1 + agg2)
# %%
len(premise0_judge_df)
# %%
len(premise1_judge_df)
# %%
len(conclusion_judge_df)
# %%
pd.crosstab(df["annA_premise0"], df["annC_premise0_cw_final"])
# %%
pd.crosstab(df["annB_premise0"], df["annC_premise0_cw_final"])
# %%
pd.concat([df["annB_premise0"], df["annB_premise1"], df["annB_premise2"]])
# %%
pd.concat([df["annC_premise0_cw_final"], df["annC_premise1_cw_final"], df["annC_premise2_cw_final"]])
# %%
pd.crosstab(pd.concat([df["annB_premise0"], df["annB_premise1"], df["annB_premise2"]], ignore_index=True).rename("LLM"), pd.concat([df["annC_premise0_cw_final"], df["annC_premise1_cw_final"], df["annC_premise2_cw_final"]], ignore_index=True).rename("Judge"), margins="all")
# %%
pd.crosstab(df["annB_conclusion"].rename("LLM"), df["annC_conclusion_cw_final"].rename("Judge"), margins="all")
# %%
pd.concat([df["annB_premise0"], df["annB_premise1"], df["annB_premise2"]]).rename("A")
# %%
