# %%
import pandas as pd
df_claim = pd.read_csv("../../data/wsf_arg_plus_per_claim.csv")
# %%
df_claim
# %%
# Save to a single txt file
output_file = 'all_claims.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for text in df_claim['claim']:
        f.write(text + '\n')
# %%
pf_ud = pd.read_csv("../../data/profiling_ud_claims.csv", sep="\t")
# %%
df_pf_ud = pd.concat([df_claim, pf_ud], axis=1)
# %%
df_pf_ud.columns
# %%
# Always one sentence
df_pf_ud["n_sentences"].describe().round(3)
# %%
df_pf_ud.columns.tolist()
# %%
groups = {
    # Raw Text Properties
    "n_sentences": ["n_sentences"],
    "n_tokens": ["n_tokens"],
    "tokens_per_sent": ["tokens_per_sent"],
    "char_per_tok": ["char_per_tok"],
    "lexical_density": ["lexical_density"],
    # Morphosyntactc Information
    "upos_dist": ["upos_dist_ADJ", # part-of-speech categories defined in the Universal POS tags
                  "upos_dist_ADP",
                  "upos_dist_ADV",
                  "upos_dist_AUX",
                  "upos_dist_CCONJ",
                  "upos_dist_DET",
                  "upos_dist_INTJ",
                  "upos_dist_NOUN",
                  "upos_dist_NUM",
                  "upos_dist_PART",
                  "upos_dist_PRON",
                  "upos_dist_PROPN",
                  "upos_dist_PUNCT",
                  "upos_dist_SCONJ",
                  "upos_dist_SYM",
                  "upos_dist_VERB",
                  "upos_dist_X"],
    # Inflectional Morphology
    "verbs_tense_dist": ["verbs_tense_dist_Past",
                         "verbs_tense_dist_Pres"],
    "verbs_mood_dist": ["verbs_mood_dist_Imp",
                        "verbs_mood_dist_Ind"],
    "verbs_form_dist": ["verbs_mood_dist_Imp",
                        "verbs_mood_dist_Ind",
                        "verbs_form_dist_Fin",
                        "verbs_form_dist_Ger",
                        "verbs_form_dist_Inf",
                        "verbs_form_dist_Part"],
    "verbs_num_pers_dist_+": ["verbs_num_pers_dist_+",
                              "verbs_num_pers_dist_Sing+3"],
    # Auxiliary usage
    "aux_tense_dist": ["aux_tense_dist_Past",
                       "aux_tense_dist_Pres"],
    "aux_mood_dist": ["aux_mood_dist_Imp",
                      "aux_mood_dist_Ind"],
    "aux_form_dist": ["aux_form_dist_Fin",
                      "aux_form_dist_Ger",
                      "aux_form_dist_Inf",
                      "aux_form_dist_Part"],
    "aux_num_pers_dist_+": ["aux_num_pers_dist_+",
                            "aux_num_pers_dist_Sing+1",
                            "aux_num_pers_dist_Sing+3"],
    # Syntactic Features
    "verbal_head_per_sent": ["verbal_head_per_sent"],
    "verbal_root_perc": ["verbal_root_perc"],
    "avg_verb_edges": ["avg_verb_edges"],
    "verb_edges_dist": ['verb_edges_dist_0',
                        'verb_edges_dist_1',
                        'verb_edges_dist_2',
                        'verb_edges_dist_3',
                        'verb_edges_dist_4',
                        'verb_edges_dist_5',
                        'verb_edges_dist_6'],
    # Global and Local Parsed Tree Structure
    "avg_max_depth": ["avg_max_depth"],
    "avg_token_per_clause": ["avg_token_per_clause"],
    "avg_max_links_len": ["avg_max_links_len"],
    "avg_links_len": ["avg_links_len"],
    "max_links_len": ["max_links_len"],
    "avg_prepositional_chain_len": ["avg_prepositional_chain_len"],
    "n_prepositional_chains": ["n_prepositional_chains"],
    "prep_dist": ["prep_dist_1", "prep_dist_2", "prep_dist_3"],
    # Order of elements
    "obj_pre": ["obj_pre"],
    "obj_post": ["obj_post"],
    "subj_pre": ["subj_pre"],
    "sub_post": ["subj_post"],
    # Syntactic Relations
    "dep_dist": ["dep_dist_acl",
                 "dep_dist_acl:relcl",
                 "dep_dist_advcl",
                 "dep_dist_advmod",
                 "dep_dist_amod",
                 "dep_dist_appos",
                 "dep_dist_aux",
                 "dep_dist_aux:pass",
                 "dep_dist_case",
                 "dep_dist_cc",
                 "dep_dist_cc:preconj",
                 "dep_dist_ccomp",
                 "dep_dist_compound",
                 "dep_dist_compound:prt",
                 "dep_dist_conj",
                 "dep_dist_cop",
                 "dep_dist_csubj",
                 "dep_dist_det",
                 "dep_dist_det:predet",
                 "dep_dist_discourse",
                 "dep_dist_expl",
                 "dep_dist_fixed",
                 "dep_dist_flat",
                 "dep_dist_goeswith",
                 "dep_dist_iobj",
                 "dep_dist_list",
                 "dep_dist_mark",
                 "dep_dist_nmod",
                 "dep_dist_nmod:npmod",
                 "dep_dist_nmod:poss",
                 "dep_dist_nmod:tmod",
                 "dep_dist_nsubj",
                 "dep_dist_nsubj:pass",
                 "dep_dist_nummod",
                 "dep_dist_obj",
                 "dep_dist_obl",
                 "dep_dist_obl:npmod",
                 "dep_dist_obl:tmod",
                 "dep_dist_parataxis",
                 "dep_dist_punct",
                 "dep_dist_root",
                 "dep_dist_vocative",
                 "dep_dist_xcomp"],
    # Use of Subordination
    "principal_proposition_dist": ["principal_proposition_dist"],
    "subordinate_proposition_dist": ["subordinate_proposition_dist"],
    "subordinate_post": ["subordinate_post"],
    "subordinate_pre": ["subordinate_pre"],
    "avg_subordinate_chain_len": ["avg_subordinate_chain_len"],
    "subordinate_dist": ["subordinate_dist_1",
                         "subordinate_dist_2",
                         "subordinate_dist_3",
                         "subordinate_dist_4",
                         "subordinate_dist_5"]
}
# %%
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Small constant to avoid log(0)
EPS = 1e-10

def normalized_entropy(row):
    row = np.asarray(row, dtype=float)

    if len(row) <= 1:
        return np.nan

    h = entropy(row + EPS)
    h_max = np.log(len(row))

    return h / h_max if h_max > 0 else np.nan


# POS diversity
upos_cols = [c for c in df_pf_ud.columns if c.startswith("upos_dist_")]

df_pf_ud["upos_entropy"] = df_pf_ud[upos_cols].apply(
    normalized_entropy,
    axis=1
)

# Dependency diversity
dep_cols = [c for c in df_pf_ud.columns if c.startswith("dep_dist_")]

df_pf_ud["dep_entropy"] = df_pf_ud[dep_cols].apply(
    normalized_entropy,
    axis=1
)

# Verb-edge diversity
verb_edge_cols = [c for c in df_pf_ud.columns if c.startswith("verb_edges_dist_")]

df_pf_ud["verb_edge_entropy"] = df_pf_ud[verb_edge_cols].apply(
    normalized_entropy,
    axis=1
)

# Subordination diversity
subord_cols = [c for c in df_pf_ud.columns if c.startswith("subordinate_dist_")]

df_pf_ud["subordinate_entropy"] = df_pf_ud[subord_cols].apply(
    normalized_entropy,
    axis=1
)

# Prepositional-chain diversity
prep_cols = [c for c in df_pf_ud.columns if c.startswith("prep_dist_")]

df_pf_ud["prep_entropy"] = df_pf_ud[prep_cols].apply(
    normalized_entropy,
    axis=1
)

# Verb morphology complexity
verb_morph_cols = [
    c for c in df_pf_ud.columns
    if c.startswith("verbs_tense_dist_")
    or c.startswith("verbs_mood_dist_")
    or c.startswith("verbs_form_dist_")
]

df_pf_ud["verb_morph_entropy"] = df_pf_ud[verb_morph_cols].apply(
    normalized_entropy,
    axis=1
)

# Auxiliary morphology complexity
aux_morph_cols = [
    c for c in df_pf_ud.columns
    if c.startswith("aux_tense_dist_")
    or c.startswith("aux_mood_dist_")
    or c.startswith("aux_form_dist_")
]

df_pf_ud["aux_morph_entropy"] = df_pf_ud[aux_morph_cols].apply(
    normalized_entropy,
    axis=1
)

# Word-order flexibility
word_order_cols = [
    c for c in [
        "obj_pre",
        "obj_post",
        "subj_pre",
        "subj_post"
    ]
    if c in df_pf_ud.columns
]

if len(word_order_cols) > 1:
    word_order_dist = (
        df_pf_ud[word_order_cols]
        .div(df_pf_ud[word_order_cols].sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0)
    )

    df_pf_ud["word_order_entropy"] = word_order_dist.apply(
        normalized_entropy,
        axis=1
    )

# Final aggregated feature set

aggregated_features = [
    "n_tokens",
    "lexical_density",
    "char_per_tok",
    "upos_entropy",
    "dep_entropy",
    "verb_morph_entropy",
    "aux_morph_entropy",
    "avg_max_depth",
    "avg_links_len",
    "avg_token_per_clause",
    "subordinate_entropy",
    "word_order_entropy"
]

aggregated_features = [
    c for c in aggregated_features if c in df_pf_ud.columns
]

profiling_ud_aggregated = df_pf_ud[aggregated_features].copy()

print(profiling_ud_aggregated.head())
# %%
profiling_ud_aggregated_cw = pd.concat([df_claim[["claim_cw_platinum"]], profiling_ud_aggregated], axis=1)
# %%
(
    profiling_ud_aggregated_cw
    #.drop(columns=["Filename"])
    .groupby("claim_cw_platinum")
    .agg(["mean", "std"])
    .round(3)
    #.to_csv("profiling_ud_aggregated.csv")
)
# %%
# %%
agg_df = (
    df_pf_ud[["claim_cw_platinum"] + list(pf_ud.columns)]
    .drop(columns=["Filename"])
    .groupby("claim_cw_platinum")
    .agg(["mean", "std"])
    .round(3)
)

# Build clean result (no duplicates)
result = pd.DataFrame(index=agg_df.index)

for col in agg_df.columns.levels[0]:
    result[col] = (
        agg_df[(col, "mean")].map("{:.3f}".format)
        + " ± "
        + agg_df[(col, "std")].map("{:.3f}".format)
    )

print(result)

# Keep only the combined columns
#result = result[agg_df.columns.levels[0]]
# %%
result.to_clipboard()
# %%
