# %%
import pandas as pd

replacements = {
    "Check-worthy Factual": "CFS",
    "Non-Factual": "NFS",
    "Unimportant Factual": "UFS"
}

df = pd.read_csv("mistral-7B-small_check_worthy_gens.csv")

cols = ["premise0", "premise1", "premise2", "conclusion"]


for c in cols:
    y_pred = df[f"{c}__prompt_check_worthy"].replace(replacements)
    y_label = df[f"check_worthy_{c}"]

    print((y_label == y_pred).sum() / len(y_label))
# %%