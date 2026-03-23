# %%
import pandas as pd

df = pd.read_csv("../data/wsf_arg_extension_with_nohs_cw.csv")
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

df = df.fillna("")

concat_texts = []
for _, row in df.iterrows():
    concat_text = ""
    for col in cols:
        if row["use_claims_only"] == "yes":
            if row[col].strip(".") == "":
                continue
            concat_text += f'{row[col].strip(".")}. '
        else:
            if row[col].strip(".") == "":
                continue
            if col == "conclusion":
                concat_text += f'Therefore, {row[col].strip(".")}.'
            else:
                concat_text += f'{row[col].strip(".")}. '
    concat_texts.append(concat_text)

df["new_concat"] = concat_texts
# %%
cols = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]

iscws = []
for _, row in df.iterrows():
    cws = [row[f"{col}_cw_final"] for col in cols]
    iscw = any(cw == "CFS" for cw in cws)
    iscws.append(iscw)
df["is_cw"] = iscws
# %%
df[df["is_cw"]].to_csv("wsf_contains_cw.csv", index=False)
# %%
df[~df["is_cw"]].to_csv("wsf_no_contains_cw.csv", index=False)
# %%
df_mtconan = pd.read_csv("../data/mtconan_llm_pred.csv")
# %%
df_mtconan[df_mtconan["cw"] == "Check-worthy Factual"].to_csv("mtconcan_contains_cw.csv", index=False)
# %%
df_mtconan[df_mtconan["cw"] != "Check-worthy Factual"].to_csv("mtconcan_no_contains_cw.csv", index=False)
# %%
import os

# Output directory
output_dir = "mtconan_cw_profiling_ud"
output_file = "mtconan_cw_profiling_ud.txt"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over DataFrame rows
for _, row in df_mtconan[df_mtconan["cw"] == "Check-worthy Factual"].iterrows():
    file_path = os.path.join(output_dir, f"{row['INDEX']}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(row['HATE_SPEECH']))

with open(output_file, "w", encoding="utf-8") as f:
    for message in df_mtconan.loc[df_mtconan["cw"] == "Check-worthy Factual", "HATE_SPEECH"]:
        f.write(f"{message}\n")
# %%
import os

# Output directory
output_dir = "mtconan_nocw_profiling_ud"
output_file = "mtconan_nocw_profiling_ud.txt"
# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over DataFrame rows
for _, row in df_mtconan[df_mtconan["cw"] != "Check-worthy Factual"].iterrows():
    file_path = os.path.join(output_dir, f"{row['INDEX']}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(row['HATE_SPEECH']))

with open(output_file, "w", encoding="utf-8") as f:
    for message in df_mtconan.loc[df_mtconan["cw"] != "Check-worthy Factual", "HATE_SPEECH"]:
        f.write(f"{message}\n")
# %%
import os

# Output directory
output_dir = "wsf_cw_profiling_ud"
output_file = "wsf_cw_profiling_ud.txt"
# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over DataFrame rows
for _, row in df[df["is_cw"]].iterrows():
    file_path = os.path.join(output_dir, f"{row['idx']}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(row['new_concat']))

with open(output_file, "w", encoding="utf-8") as f:
    for message in df.loc[df["is_cw"], "new_concat"]:
        f.write(f"{message}\n")
# %%
import os

# Output directory
output_dir = "wsf_nocw_profiling_ud_sample"
output_file = "wsf_nocw_profiling_ud.txt"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over DataFrame rows
for _, row in df[~df["is_cw"]].iterrows():
    file_path = os.path.join(output_dir, f"{row['idx']}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(row['new_concat']))

with open(output_file, "w", encoding="utf-8") as f:
    for message in df.loc[~df["is_cw"], "new_concat"]:
        f.write(f"{message}\n")
# %%
