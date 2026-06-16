# %%
import pandas as pd

df_orig = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
openai_scores = pd.read_csv("../../data/openai_mod_tool_wsf_arg_plus.csv")
# %%
df = pd.concat([df_orig, openai_scores], axis=1)
# %%
df_hs = df[df["concat_hate"] == 1].copy()
# %%
from sklearn.metrics import recall_score

y_true = df_hs["concat_hate"] # all 1's
y_pred = df_hs["openai_flagged"] # preds from the openai tool

recall_score(y_true=y_true, y_pred=y_pred).__round__(3)
# %%
(sum(y_pred == y_true) / len(df_hs) * 100).__round__(2)
# %%
sum(y_pred == y_true)
# %%
df_hs[y_pred != y_true]
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

mask = []
for _, row in df_hs.iterrows():
    cws = [row[f"{col}_cw_platinum"] for col in cols]
    contains_cw = any(cw == "CFS" for cw in cws)
    mask.append(contains_cw)
df_hs["contains_cw"] = mask
# %%
cols_scores = [c for c in openai_scores.columns if "score" in c]
(
    df_hs
    .loc[:, ["contains_cw"] + cols_scores]
    .groupby("contains_cw")
    .mean()
    .T
    .round(3)
)
# %%
cols_scores
# %%
(df_hs["openai_score_harassment"] >= 0.1).sum().item()
# %%
(df_hs["openai_score_hate"] >= 0.1).sum().item()
# %%
df_hs["openai_cat_harassment"].sum().item()
# %%
(df_hs["openai_cat_harassment"].sum() / len(df_hs)).round(3).item()
# %%
df_hs["openai_cat_hate"].sum().item()
# %%
(df_hs["openai_cat_hate"].sum() / len(df_hs)).round(3).item()
# %%
(df_hs["openai_score_hate"] >= 0.3).sum().item()
# %%
from sklearn.metrics import precision_recall_fscore_support

ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

res_hate = {}
for th in ths:
    y_pred = (df["openai_score_hate"] >= th).astype(int)
    y_true = df["concat_hate"] 
    res_hate[th] = {}
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_true,
                                                  y_pred=y_pred,
                                                  average="macro")
    res_hate[th]["p_hate"] = p
    res_hate[th]["r_hate"] = r
    res_hate[th]["f1_hate"] = f1
# %%
res_hate_df = pd.DataFrame(res_hate).T.round(3)
# %%
print(res_hate_df.to_markdown())
# %%
res_harassment = {}
for th in ths:
    y_pred = (df["openai_score_harassment"] >= th).astype(int)
    y_true = df["concat_hate"] 
    res_harassment[th] = {}
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_true,
                                                  y_pred=y_pred,
                                                  average="macro")
    res_harassment[th]["p_harassment"] = p
    res_harassment[th]["r_harassment"] = r
    res_harassment[th]["f1_harassment"] = f1
# %%
res_harassment_df = pd.DataFrame(res_harassment).T.round(3)
# %%
print(res_harassment_df.to_markdown())
# %%
pd.concat([
    res_hate_df,
    res_harassment_df
], axis=1)
# %%
print(pd.concat([
    res_hate_df,
    res_harassment_df
], axis=1).to_markdown())
# %%
