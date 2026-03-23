# %% [markdown]
# ## OpenAI Moderation Tool on HS messages that have CW claims
# This notebook reproduces the results of Table 5 of the paper.
# %%
import pandas as pd
df = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
openai_scores = pd.read_csv("../../data/openai_mod_tool_wsf_arg_plus.csv")

cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]
# %%
iscws = []

for _, row in df.iterrows():
    cws = [row[f"{col}_cw_platinum"] for col in cols]
    iscw = any(cw == "CFS" for cw in cws)
    iscws.append(iscw)
df["contains_cw"] = iscws

res_mod_tool = pd.concat([df, openai_scores], axis=1)
# %%
cols_scores = [c for c in openai_scores.columns if "score" in c]
# %%
# Correction: In the paper runs the test with the entire dataset.
# We correct this and we run it with only the HS partition using platinum
# We obtain the same conclusions. Statistically significant higher scores
# of harassment and hate for HS messages with at least one check-worthy claim
# than those without.
# harassment p = 0.004 (effect_size = 0.227) mean w/cw = 0.728, mean w/o cw = 0.611
# hate       p = 0.038 (effect_size = 0.154) mean w/cw = 0.410, mean w/o cw = 0.319

res_mod_tool_hs = res_mod_tool.loc[
    res_mod_tool["concat_hate"] == 1,
    ["contains_cw"] + cols_scores
]
# %%
res_mod_tool_hs.groupby("contains_cw").mean().T
# %%
from scipy.stats import mannwhitneyu

res_mannwhitneyu_less = {}
for c in cols_scores:
    res_mannwhitneyu_less[c] = {}
    stat, p = mannwhitneyu(res_mod_tool_hs.loc[~res_mod_tool_hs["contains_cw"], c].reset_index(drop=True),
                           res_mod_tool_hs.loc[res_mod_tool_hs["contains_cw"], c].reset_index(drop=True),
                           alternative="less", method="auto")
    res_mannwhitneyu_less[c]["stat"] = stat
    res_mannwhitneyu_less[c]["p"] = p
    res_mannwhitneyu_less[c]["is_significant"] = p < 0.05
pd.DataFrame(
    res_mannwhitneyu_less
)
# %%
stat_test = pd.DataFrame(
    res_mannwhitneyu_less
).T
# %%
stat_test["effect_size"] = 1 - (2 * stat_test["stat"]) / (len(res_mod_tool_hs[res_mod_tool_hs["contains_cw"]]) * len(res_mod_tool_hs[~res_mod_tool_hs["contains_cw"]])) 
# %%
stat_test
# %%
