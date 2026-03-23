# %%
import pandas as pd

df_scores = pd.read_csv("../data/openai_moderation_wsf.csv")
df = pd.read_csv("../data/wsf_arg_extension_with_nohs_cw.csv")
# %%
cols_scores = [c for c in df_scores.columns if "score" in c]
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

iscws = []
for _, row in df.iterrows():
    cws = [row[f"{col}_cw_final"] for col in cols]
    iscw = any(cw == "CFS" for cw in cws)
    iscws.append(iscw)
df["is_cw"] = iscws

res_mod_tool = pd.concat([df, df_scores], axis=1)
# %%
res_mod_tool[["is_cw"] + cols_scores].groupby("is_cw").describe().to_csv("../data/openai_moderation_wsf_agg.csv")
# %%
from scipy.stats import mannwhitneyu

res_mannwhitneyu = {}
for c in cols_scores:
    res_mannwhitneyu[c] = {}
    stat, p = mannwhitneyu(res_mod_tool.loc[~res_mod_tool["is_cw"], c].reset_index(drop=True),
                           res_mod_tool.loc[res_mod_tool["is_cw"], c].reset_index(drop=True),
                           alternative="less")
    res_mannwhitneyu[c]["stat"] = stat
    res_mannwhitneyu[c]["p"] = p
    res_mannwhitneyu[c]["is_significant"] = p < 0.05

pd.DataFrame(
    res_mannwhitneyu
).to_csv("../data/openai_moderation_wsf_sign_test_less.csv")
# %%
res_mannwhitneyu = {}
for c in cols_scores:
    res_mannwhitneyu[c] = {}
    stat, p = mannwhitneyu(res_mod_tool.loc[~res_mod_tool["is_cw"], c].reset_index(drop=True),
                           res_mod_tool.loc[res_mod_tool["is_cw"], c].reset_index(drop=True),
                           alternative="greater")
    res_mannwhitneyu[c]["stat"] = stat
    res_mannwhitneyu[c]["p"] = p
    res_mannwhitneyu[c]["is_significant"] = p < 0.05

pd.DataFrame(
    res_mannwhitneyu
).to_csv("../data/openai_moderation_wsf_sign_test_greater.csv")
# %%
res_mannwhitneyu = {}
for c in cols_scores:
    res_mannwhitneyu[c] = {}
    stat, p = mannwhitneyu(res_mod_tool.loc[~res_mod_tool["is_cw"], c].reset_index(drop=True),
                           res_mod_tool.loc[res_mod_tool["is_cw"], c].reset_index(drop=True),
                           alternative="two-sided")
    res_mannwhitneyu[c]["stat"] = stat
    res_mannwhitneyu[c]["p"] = p
    res_mannwhitneyu[c]["is_significant"] = p < 0.05

pd.DataFrame(
    res_mannwhitneyu
).to_csv("../data/openai_moderation_wsf_sign_test_two-sided.csv")
