# %%
import pandas as pd

df = pd.read_csv("../../data/wsf_arg_plus_per_claim_submission.csv")
# %%
llm_loop_needed_judge = df.loc[df["claim_needed_judge_llm_loop"] == 1, "claim_idx"]
full_human_needed_judge = df.loc[df["claim_needed_judge_gold"] == 1, "claim_idx"]
# %%
len(llm_loop_needed_judge)
# %%
len(full_human_needed_judge)
# %%
sum(claim in llm_loop_needed_judge for claim in full_human_needed_judge)
# %%
df["claim_agreement_count_gold"].value_counts()
# %%
df.columns
# %%
