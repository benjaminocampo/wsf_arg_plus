# %%
import pandas as pd

df = pd.read_csv("/Users/nicola/Documents/projects/cs_hs_misinfo/data/raw_generations/hs_detection/mlruns/410226340187067821/0c4c4da1dc5c476fbb089c6adde12383/artifacts/mistral-22B-medium_detect_hs_with_cw_noarg_llm_pred.csv")
# %%
df["premise0_cw_final"].value_counts().sum()
# %%
df["premise0_cw_annA"].value_counts().sum()
# %%
df["premise0_cw_final"]
# %%
df_orig = pd.read_csv("../../data/wsf_arg_plus_per_message.csv")
# %%
df_orig["conclusion_cw_gold"].value_counts()
# %%
df["conclusion_cw_final"].value_counts()
# %%
all(
    df_orig[df_orig["premise0_cw_gold"].notna()]["premise0_cw_gold"] ==
    df[df["premise0_cw_final"].notna()]["premise0_cw_final"]
)
# %%
all(
    df_orig[df_orig["premise1_cw_gold"].notna()]["premise1_cw_gold"] ==
    df[df["premise1_cw_final"].notna()]["premise1_cw_final"]
)
# %%
all(
    df_orig[df_orig["premise2_cw_gold"].notna()]["premise2_cw_gold"] ==
    df[df["premise2_cw_final"].notna()]["premise2_cw_final"]
)
# %%
all(
    df_orig[df_orig["premise3_cw_gold"].notna()]["premise3_cw_gold"] ==
    df[df["premise3_cw_final"].notna()]["premise3_cw_final"]
)
# %%
all(
    df_orig[df_orig["premise4_cw_gold"].notna()]["premise4_cw_gold"] ==
    df[df["premise4_cw_final"].notna()]["premise4_cw_final"]
)
# %%
all(
    df_orig[df_orig["premise5_cw_gold"].notna()]["premise5_cw_gold"] ==
    df[df["premise5_cw_final"].notna()]["premise5_cw_final"]
)
# %%
all(
    df_orig[df_orig["conclusion_cw_gold"].notna()]["conclusion_cw_gold"] ==
    df[df["conclusion_cw_final"].notna()]["conclusion_cw_final"]
)
# %%
