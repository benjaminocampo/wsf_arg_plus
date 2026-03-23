# %%
import pandas as pd
from glob import glob

hs_runs = {
	"Mistral-7B": {
		"zero": ["7cff17b88ad94faa9464b2947940a2d0",
            "95f14e05f4de44b5bc8de54beeb797e8",
            "13e8d5a7a90f4df7955bd560ce954cb9"],
		"one": ["995d1484d0464cb2897b8ea1701ce757",
            "c81f652ad938401cac8dc09b2d4af57e",
            "40e97314513046ec80d3c564ad524ae1"]
	},
	"Llama-8B": {
		"zero": ["4d0f4bdc75d74ee6a1f69beaa5103388",
            "42d9ae35ca4f4d1db16008bdbfc7a9f7",
            "daee506eb3694b5788a3ad4dc79be07f"],
		"one":	["bed4838077b24457a96f0761d7beccf7",
            "2c21dfff23874cae87404fd5abd7e313",
            "cd696bd8a1eb420e9aaecfa5162195c0"]
	},
	"Olmo2-7B": {
		"zero": ["3229a4dfda934fda830d0b2effb0298f",
            "50e85531013f40d7abf041c6ff338351",
            "a66c91e48b0840ed9182bd75fdebbb67"],
		"one":	["7fa67fe6a9cc4930a52dcc41e028a13a",
            "38630f13f4ca43928f7839f95b0e5ae5",
            "72d26a8e05d549269e416358c2477436"]
	},
	"Qwen2.5-7B": {
		"zero": ["2686e66957324dcebb68e7e14209461d",
            "bf33baec21b542a2a17bdced8275fa8b",
            "de923c3de24f4413a20a6eb3b6ba2a76"],
		"one":	["a158fc90640a48fd87adcf2c32af96cd",
            "b22d57c18a634050bef8bf13c2df816c",
            "409bd7acf1b74e07908c54320ef7e0bf"]
	},
	"Command-r-7B": {
		"zero": ["2cb5a96289a54a4a84c4934163cf226b",
            "7e952d9fd707440ca0c36e127482cc56",
            "50c78f11aa894e7786df46da3b05a890"],
		"one":	["f300054ab578477e9543926aecdae565",
            "874ee18a6d284a419800f2743b972707",
            "672f5a7d65e645b39070b4a4aeeb2a35"]
	},
	"Mixtral-8x7B": {
		"zero": ["42718aa5c1734282b1d5f86be12abed3",
            "1f4eb94569434298818f282f6da09276",
            "4fabb7804e0b43dd868193f063afaf7e"],
		"one":	["e24034b5591e4d3795247f4cbfc6c475",
            "b134d8e36ae94fe2b47b686284636966",
            "8c82da6137bb4f65985ddde267a6a489"]
	},
	"Mistral-22B": {
		"zero": ["d8c0afbd8c6b461496d8e323aea7d846",
            "d3c2a2ee237648b58324fc4cfc87346a",
            "dd527efae34c42ab82196cc6f0da9595"],
		"one":	["7251653543d147a490a840ebce03fd21",
            "0c5129b92b6b4fbd97d6b5ed03d5b0e4",
            "7b86bb9eae5747ef94c5521e345d0ba5"] 
	},
	"Olmo2-32B": {
		"zero": ["22493e1f462142fc8b671e105e706036",
            "31589a761eff46f29f25be22e615ea84",
            "1480eec713a24144b46a553c826ce260"],
		"one":	["9804734bc67745e9bfab6a3abbda4c66",
            "74aa782b94554dadbac41b88e2a55e13",
            "c2615b4b6bc84004855afa0397bc4581"]
	},
	"Mixtral-8x22B": {
		"zero": ["ea51bb6edb6c4e79aa215663adb22c13",
            "255fd733ab3641e19f95c1823ef5923c",
            "99a9a940cbe746b78a4cc8a2a765ec4e"],
		"one":	["6b9fc2f3b96943c4925fef062a7ac144",
            "99f019dae80742e69a971434369103d0",
            "22e2f340f0644e46b52f9dac99485efc"]
	},
	"Llama-70B": {
		"zero": ["ce0707c1e3a0426fbe226774ceac01f7",
            "c9dcabb0dbde4746b2dfe72c56d24245",
            "32a9d8af106147628ba21c37712301fa"],
		"one":	["b90de0ee45b340c69b609a1809206038",
            "31051c3dfecc49a88080c77b470b4365",
            "684838b6f87a436a8cdcae4587a6e414"]
	},
	"Qwen2.5-72B": {
		"zero": ["e6ca32e4ac724205aeca7c9f7d5ac015",
            "851646123c0a4150938988be249ba5a3",
            "866d4a837c7e4b0ca6abf6a4a539b7b3"],
		"one":	["1f457c7d1aca447fb8c292e0ce7861fc",
            "014277e19af4488f97128047d4d7b936",
            "c86dfbc2f7fd4891b6f89ddbbb66b3aa"]
	},
	"Command-r-104B": {
		"zero": ["99977df7b2044889b1c188ea7941ae5f",
            "4b38fb694add4ec59378324ecadd4b49",
            "ec0e39e0a48a49ee957c0d44ba1f9274"],
		"one":	["d1f57836592341d7bf1142aa7b767fca",
            "e89477f022c24d4e855ba2ce01b63f9e",
            "95aacd605afb42df81d3885218bbd925"]
	}
}

non_hs_runs = {
	"Mistral-7B": {
		"zero": ["52843c37457e43fda9433977d4be0aad",
			"1761317441ab49f1bd27131f4ec7ecc0",
			"f1709570f06441e2b324201e5bc8ac7a"],
		"one": ["6d23cb1b57dd49fd8be73edaef5e9e9a",
			"0f048b66daab4e0f85475cfcf55aab76",
			"bc45cc6bcf9046968c20b31575a0209c"]
	},
	"Llama-8B": {
		"zero": ["d077e0371d234b739c4d9c36a96bc3f1",
			"a0ddbf214b2a4da6997d6f86b334bc34",
			"79aaffbd6ee34e2f9d4db110893d4ebe"],
		"one": ["7226c5ce68824f17a62d677fae7fbd49",
			"47e299909a0040d2b6b3b7259930c998",
			"bc353855396c414397844c62af7e9981"]
	},
	"Olmo2-7B": {
		"zero": ["f3e1892982354cacb2dcdbd6958791b8",
			"2263c262a1bd4a1186080eb2c07c9521",
			"7f3d4128e5af4e86971a95a10dc4d70f"],
		"one": ["a189c854394548789f5c4ab338f1927b",
			"f6ca684af40546cd8badbb4ff45b4c15",
			"4a7555dc723b40ca87419b79cdee719b"]
	},
	"Qwen2.5-7B": {
		"zero": ["09edf0b1108446b2b2f15d1be8bdb421",
			"7f2490b3269643e79f4255577afe54ee",
			"5a52066bea284fb99ebf39a0091bd08c"],
		"one": ["1ad970717d5b4f30a0c10a8014d1b8da",
			"d806e79ee1c04c15b9a693836c0d601f",
			"365b2c14f82848899650e37af91599cb"]
	},
	"Command-r-7B": {
		"zero": ["9410a2309b324d039b017fc7d955f1d3",
			"3fc49ad927a84153a2ba76c332aa4e11",
			"473e8cdcbbd047e0b4135ea55cb78318"],
		"one": ["a5f16a3340de4f9fa9fa024f18f836e9",
			"b469ad1f606a484d9175a1598c79b3d9",
			"a023e774964a4bfead42557d8992761d"]
	},
	"Mixtral-8x7B": {
		"zero": ["d1af32e81af1433684d09f951566acc0",
			"a08e1f1bebd24745a4dd3a2a93004231",
			"44e2ed98a9e44296919cca58920d3035"],
		"one": ["665d9ef820324a3d93af9375486cb5a0",
			"7dc8a62fa4cc4c08956f9dfdf009e0ac",
			"04fd949601fa42b8a025e2294a7e7250"]
	},
	"Mistral-22B": {
		"zero": ["4056a378d89c49dd91c346be97381038",
			"a8aae877705846a3bd3638d2948c7d82",
			"5ed1fcc3929f46da847b9c9ceb2e089a"],
		"one": ["aa095bd6d49948949b4fc6ce8ca4a0e0",
			"73e123d7989c431388d7ffab7237b2fd",
			"625b86bed29c4255a4f13888d7f045d7"]
	},
	"Olmo2-32B": {
		"zero": ["c402df23101942be9aaa23c7ebc51466",
			"5ad50208838146ee83290b8a58d103d7",
			"2a4a3df6f73f4868aff0acdae8be32dd"],
		"one": ["efcf4941b0e24f389218b4abed1f8ed9",
			"48e4e961a8c84ec19a019033fabb44f5",
			"285d42b170334953a36b42456da89eed"]
	},
	"Mixtral-8x22B": {
		"zero": ["56d2d8ba382d411e8b228091894c832a",
			"cd67182fa4a643efbd0d416c1ed78c2f",
			"7b796544e7d945fdb1c72a9dec6027aa"],
		"one": ["e6c26f3d09e24a76b0836fb7e6f694d0",
			"d735c17111864444a6c823894ed1a47f",
			"d62f994ca1f54176aa9f0d6f9416970c"]
	},
	"Llama-70B": {
		"zero": ["34010b33b75b4a64b05128f088db7558",
			"cd2743e9dec74f18bee970666c692ea7",
			"d2b9040a60b5413890b24d8f46f0f057"],
		"one": ["037d82986f8b420faa5c4780e0662990",
			"0ab4466d34f145bb8a00cec4b44a7764",
			"b4b34ef8c87045419034988f7cd09689"]
	},
	"Qwen2.5-72B": {
		"zero": ["57451b3c3db74047be5b5a5493a9f1a3",
			"ad0e9fe5528b49b1b42e7c1695ecd554",
			"e5fe413627964701a1bc996e0abd4828"],
		"one": ["be79d55021c74e1fba6388165cfd26d9",
			"82f2efc3189444a4bc9eb867ce2df738",
			"2d948e0f0190458299437ad53c2d33bb"]
	},
	"Command-r-104B": {
		"zero": ["66d760f7dd374a1da7f1659e27b7a1ae",
			"3fc49ad927a84153a2ba76c332aa4e11",
			"baeb92cc525e48349d3e3133b717b361"],
		"one": ["9aa406a453534019a97b9819cef9d1c0",
			"d67ea97e58254a1192d92c941e94bc3d",
			"3ed9a4370b0440c793d303518639ebfb"]
	}
}
# %%
replacements = {
    "Check-worthy Factual": "CFS",
    "Non-Factual": "NFS",
    "Unimportant Factual": "UFS",
}
cols_hs = ["premise0", "premise1", "premise2", "conclusion"]
cols_nonhs = ["premise0", "premise1", "premise2", "premise3", "premise4", "premise5", "conclusion"]
# %%
hs_runs_df = {}
for model_name, runs in hs_runs.items():
    hs_runs_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        hs_runs_df[model_name][shot_type] = []
        for id in run_ids:
            matches = glob(f"./mlruns_hs/**/{id}/artifacts/**.csv")
            hs_runs_df[model_name][shot_type].append((id, matches[0], pd.read_csv(matches[0])))
# %%
non_hs_runs_df = {}
for model_name, runs in non_hs_runs.items():
    non_hs_runs_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        non_hs_runs_df[model_name][shot_type] = []
        for id in run_ids:
            matches = glob(f"./mlruns_nonhs/**/{id}/artifacts/**.csv")
            non_hs_runs_df[model_name][shot_type].append((id, matches[0], pd.read_csv(matches[0])))
# %%
import os
# Saving hs data
os.makedirs("cw_gens", exist_ok=True)
os.makedirs("cw_gens/hs", exist_ok=True)
for model_name, runs in hs_runs_df.items():
    os.makedirs(f"cw_gens/hs/{model_name}", exist_ok=True)
    for shot_type, run_ids in runs.items():
        os.makedirs(f"cw_gens/hs/{model_name}/{shot_type}", exist_ok=True)
        for run_id_3upla in run_ids:
            id, path, df = run_id_3upla
            os.makedirs(f"cw_gens/hs/{model_name}/{shot_type}/{id}", exist_ok=True)
            csv_filename = path.split("/")[-1]
            for c in cols_hs:
                df[f"{c}_llm_pred"] = df[f"{c}_llm_pred"].replace(replacements)
                df = df.rename(columns={f"{c}_llm_pred": f"{c}_cw_annB"})
                df = df.rename(columns={f"check_worthy_{c}": f"{c}_cw_annA"})
                df = df.drop(columns=[f"truth-o-meter_{c}"])
                df = df.drop(columns=[f"comments_{c}"])
            df.to_csv(f"cw_gens/hs/{model_name}/{shot_type}/{id}/{csv_filename}", index=False)

# %%
import os
# Saving non_hs data
os.makedirs("cw_gens", exist_ok=True)
os.makedirs("cw_gens/non_hs", exist_ok=True)
for model_name, runs in non_hs_runs_df.items():
    os.makedirs(f"cw_gens/non_hs/{model_name}", exist_ok=True)
    for shot_type, run_ids in runs.items():
        os.makedirs(f"cw_gens/non_hs/{model_name}/{shot_type}", exist_ok=True)
        for run_id_3upla in run_ids:
            id, path, df = run_id_3upla
            os.makedirs(f"cw_gens/non_hs/{model_name}/{shot_type}/{id}", exist_ok=True)
            csv_filename = path.split("/")[-1]
            for c in cols_nonhs:
                df[f"{c}_cw_annB"] = df[f"{c}_cw_annB"].replace(replacements)
            df.to_csv(f"cw_gens/non_hs/{model_name}/{shot_type}/{id}/{csv_filename}", index=False)
# %%
# Reading new dfs
hs_runs_df = {}
for model_name, runs in hs_runs.items():
    hs_runs_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        hs_runs_df[model_name][shot_type] = []
        for id in run_ids:
            matches = glob(f"./cw_gens/**/**/**/{id}/**.csv")
            hs_runs_df[model_name][shot_type].append((id, matches[0], pd.read_csv(matches[0])))


# Reading new dfs
non_hs_runs_df = {}
for model_name, runs in non_hs_runs.items():
    non_hs_runs_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        non_hs_runs_df[model_name][shot_type] = []
        for id in run_ids:
            matches = glob(f"./cw_gens/**/**/**/{id}/**.csv")
            non_hs_runs_df[model_name][shot_type].append((id, matches[0], pd.read_csv(matches[0])))
# %%
# Majority Voting
from collections import Counter

def majority_vote(row, col):
    values = [row[f"{col}_m0"], row[f"{col}_m1"], row[f"{col}_m2"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({col: most_common_value, f"{col}_agreement_count": count})
    else:
        return pd.Series({col: "All unequal", f"{col}_agreement_count": 1})

replacements = {
    "Check-worthy Factual": "CFS",
    "Non-Factual": "NFS",
    "Unimportant Factual": "UFS",
}
cols = ["premise0", "premise1", "premise2", "conclusion"]

non_hs_mj_vot_df = {}
for model_name, runs in non_hs_runs_df.items():
    non_hs_mj_vot_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        df0, df1, df2 = run_ids[0][2], run_ids[1][2], run_ids[2][2]
        mj_vot_df = df0.copy()
        mj_vot_df = mj_vot_df.drop(
            columns=[f"{c}_cw_annB"for c in cols_nonhs]
        )
        for c in cols_nonhs:
            mj_vot_df[f"{c}_cw_annB_m0"] = df0[f"{c}_cw_annB"]
            mj_vot_df[f"{c}_cw_annB_m1"] = df1[f"{c}_cw_annB"]
            mj_vot_df[f"{c}_cw_annB_m2"] = df2[f"{c}_cw_annB"]
            mj_vot_df[[f"{c}_cw_annB", f"{c}_llm_pred_agreement_count"]] = mj_vot_df.apply(
                lambda row: majority_vote(row, col=f"{c}_cw_annB"), axis=1
            )
        #mj_vot_df.to_csv(f"./cw_gens/non_hs/{model_name}/{shot_type}/mj_vot.csv", index=False)
        non_hs_mj_vot_df[model_name][shot_type] = mj_vot_df
# %%
hs_mj_vot_df = {}
for model_name, runs in hs_runs_df.items():
    hs_mj_vot_df[model_name] = {}
    for shot_type, run_ids in runs.items():
        df0, df1, df2 = run_ids[0][2], run_ids[1][2], run_ids[2][2]
        mj_vot_df = df0.copy()
        mj_vot_df = mj_vot_df.drop(
            columns=[f"{c}_cw_annB"for c in cols_hs]
        )
        for c in cols_hs:
            mj_vot_df[f"{c}_cw_annB_m0"] = df0[f"{c}_cw_annB"]
            mj_vot_df[f"{c}_cw_annB_m1"] = df1[f"{c}_cw_annB"]
            mj_vot_df[f"{c}_cw_annB_m2"] = df2[f"{c}_cw_annB"]
            mj_vot_df[[f"{c}_cw_annB", f"{c}_llm_pred_agreement_count"]] = mj_vot_df.apply(
                lambda row: majority_vote(row, col=f"{c}_cw_annB"), axis=1
            )
        #mj_vot_df.to_csv(f"./cw_gens/hs/{model_name}/{shot_type}/mj_vot.csv", index=False)
        hs_mj_vot_df[model_name][shot_type] = mj_vot_df
# %%
hs_mj_vot_res = {}
for model_name, runs in hs_mj_vot_df.items():
    hs_mj_vot_res[model_name] = {}
    for shot_type, mj_vot_df in runs.items():
        hs_mj_vot_res[model_name][shot_type] = pd.concat([mj_vot_df.loc[(~mj_vot_df[c].isna()), f"{c}_llm_pred_agreement_count"].value_counts() for c in cols_hs ], axis=1)
# %%
pd.DataFrame(hs_mj_vot_res).T.to_csv("mj_voting_hs.csv")
# %%
df_long_hs = pd.DataFrame(hs_mj_vot_res).T.reset_index().rename(columns={"index": "model"}).melt("model", var_name="col", value_name="val")
df_long_hs[[3, 2, 1]] = df_long_hs["val"].apply(lambda t: t.sum(axis=1))
# %%
df_long_hs.to_csv("mj_voting_hs_all.csv")
# %%
non_hs_mj_vot_res = {}
for model_name, runs in non_hs_mj_vot_df.items():
    non_hs_mj_vot_res[model_name] = {}
    for shot_type, mj_vot_df in runs.items():
        non_hs_mj_vot_res[model_name][shot_type] = pd.concat([mj_vot_df.loc[mj_vot_df["use_claims_only"] != "yes", f"{c}_llm_pred_agreement_count"].value_counts() for c in cols_nonhs ], axis=1)
# %%
pd.DataFrame(non_hs_mj_vot_res).T.to_csv("mj_voting_non_hs_argdata.csv")
# %%


# %%
non_hs_mj_vot_res_all = {}
for model_name, runs in non_hs_mj_vot_df.items():
    non_hs_mj_vot_res_all[model_name] = {}
    for shot_type, mj_vot_df in runs.items():
        non_hs_mj_vot_res_all[model_name][shot_type] = pd.concat([mj_vot_df.loc[(~mj_vot_df[c].isna()) & (mj_vot_df["concat_hate"] == 0), f"{c}_llm_pred_agreement_count"].value_counts() for c in cols_nonhs], axis=1)
# %%
pd.DataFrame(non_hs_mj_vot_res_all).T.to_csv("mj_voting_non_hs_all.csv")

# %%

df_long = pd.DataFrame(non_hs_mj_vot_res_all).T.reset_index().rename(columns={"index": "model"}).melt("model", var_name="col", value_name="val")
df_long[[3, 2, 1]] = df_long["val"].apply(lambda t: t.sum(axis=1))
df_long.to_csv("mj_voting_non_hs_all.csv")
#df_long[["shot", "score"]] = df_long["col"].str.rsplit("_", expand=True)
#
#out = (
#    df_long
#    .groupby(["model", "shot", "score"])["val"]
#    .sum()
#    .unstack("score", fill_value=0)
#)
# %%
