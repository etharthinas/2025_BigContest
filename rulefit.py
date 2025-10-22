import pandas as pd
import numpy as np

# from sklearn.model_selection import StratifiedKFold
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# from sklearn.preprocessing import StandardScaler
from imodels import Rule, RuleFitClassifier

# from scipy.special import expit
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from imodels.util.rule import get_feature_dict, replace_feature_name
from imodels.util.arguments import check_fit_arguments
from collections import Counter


# === STEP 1. PREPARE DATA ===
def prepare_data(df):
    # replace missing BZN values with None (for clarity)
    df["bzn"] = df["bzn"].replace({np.nan: None})

    # pivot to wide form, one row per mct
    pivot_df = df.pivot(
        index=["mct", "zcd", "bzn", "closed"],
        columns="attribute",
        values=[
            "mean_all",
            "mean_zcd",
            "mean_bzn",
            "slope_all",
            "slope_zcd",
            "slope_bzn",
        ],
    )
    pivot_df.columns = ["_".join(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    return pivot_df


# === STEP 2. RULEFIT TRAINING FUNCTION ===
def train_rulefit(X, y, random_state=42, max_rules=50, cutoff = 10, include_linear=False):

    rf = RuleFitClassifier(
        tree_size=2, sample_fract=1.0, max_rules=max_rules, random_state=random_state, include_linear=include_linear
    )
    X, y, feature_names = check_fit_arguments(rf, X, y, feature_names=None)
    rf.n_features_ = X.shape[1]
    rf.feature_dict_ = get_feature_dict(X.shape[1], feature_names)
    rf.feature_placeholders = np.array(list(rf.feature_dict_.keys()))
    rf.feature_names = np.array(list(rf.feature_dict_.values()))
    extracted_rules = rf._extract_rules(X, y)
    rf.rules_without_feature_names_, rf.coef, rf.intercept = rf._score_rules(
        X, y, extracted_rules
    )
    rf.rules_ = [
        replace_feature_name(rule, rf.feature_dict_)
        for rule in rf.rules_without_feature_names_
    ]

    invert_dict = dict(zip(rf.feature_dict_.values(), rf.feature_dict_.keys()))

    all_rules = rf._get_rules()
    all_rules = all_rules.loc[(abs(all_rules.coef) > 0.05)]
    all_rules = (
        all_rules.sort_values(by=["importance"])
        .loc[all_rules.type == "rule"]
        .head(cutoff)
    )
    # key_features = all_rules.sort_values(by = ['importance'], ascending=False).head(rf.max_rules)['rule'].apply(lambda x: x.split(' ')[0])
    # key_counter = Counter(key_features)
    # key_features_ = [k for k, _ in key_counter.most_common(attr)]
    # breakpoint()

    key_rules = (
        all_rules.loc[all_rules.type == "rule"]["rule"]
        .apply(lambda x: x.replace(x.split(" ")[0], invert_dict[x.split(" ")[0]]))
    )

    # new_rules = [rule for rule in extracted_rules if rule.split(" ")[0] in key_features_]
    rf.rules_without_feature_names_, rf.coef, rf.intercept = rf._score_rules(
        X, y, key_rules
    )
    rf.rules_ = [
        replace_feature_name(rule, rf.feature_dict_)
        for rule in rf.rules_without_feature_names_
    ]
    rf.complexity_ = rf._get_complexity()
    return rf


# === STEP 3. RULE EXTRACTION (OPTIONAL INTERPRETABILITY) ===
def extract_rules(rf, rule_type, category=None):
    rules = rf._get_rules()
    rules = rules[(rules.coef != 0) & (rules.support > 0)]
    rules["type"] = rule_type
    rules["category"] = category
    return rules


# === STEP 4. PIPELINE MAIN ===
def rulefit_pipeline(df):
    pivot_df = prepare_data(df)
    X_all = pivot_df.drop(columns=["closed", "mct", "zcd", "bzn"])
    y_all = pivot_df["closed"]

    closed_df = pivot_df[pivot_df["closed"] == 1]

    static_cols = ['mct', 'zcd', 'bzn', 'closed']
    # Select all columns *except* the static ones
    perturb_cols = closed_df.columns.drop(static_cols)
    static_df = closed_df[static_cols]
    dynamic_df = closed_df[perturb_cols]

    # 3. Set parameters for noise generation
    num_copies = 20
    mean = 0.0
    std_dev = 0.00
    noise_shape = dynamic_df.shape # (num_rows, num_perturb_cols)

    # 4. Create the list of 10 mock DataFrames
    mock_data_list = []
    for _ in range(num_copies):
        # Generate the random noise array
        # N(0, 0.1) -> loc=mean, scale=std_dev
        noise = np.random.normal(loc=mean, scale=std_dev, size=noise_shape)
        
        # Add noise to the dynamic part
        # We add the numpy array directly to the DataFrame
        perturbed_dynamic_df = dynamic_df + noise
        
        # Re-combine the static columns with the new perturbed columns
        new_mock_df = pd.concat([static_df, perturbed_dynamic_df], axis=1)
        
        # Add the newly created DataFrame to our list
        mock_data_list.append(new_mock_df)

    combined_df = pd.concat(mock_data_list)
    new_pivot_df = pd.concat([pivot_df, combined_df], ignore_index=True)

    new_X_all = new_pivot_df.drop(columns=["closed", "mct", "zcd", "bzn"])
    new_y_all = new_pivot_df["closed"]
    # breakpoint()

    all_rules = []
    preds_general = np.zeros(len(pivot_df))
    preds_zcd = np.zeros(len(pivot_df))
    preds_bzn = np.zeros(len(pivot_df))

    # --- General Model ---
    general_feats = [
        c for c in X_all.columns if c.startswith(("mean_all", "slope_all"))
    ]
    rf_gen = train_rulefit(new_X_all[general_feats], new_y_all, max_rules=50, include_linear=True, cutoff=23)
    preds_general = rf_gen.predict_proba(X_all[general_feats])[:, 1]
    all_rules.append(extract_rules(rf_gen, "general"))

    # --- ZCD Models ---
    zcd_rules_list = []
    for ((zcd_name, sub), (_, sub_org)) in tqdm(zip(new_pivot_df.groupby("zcd"), pivot_df.groupby("zcd")), desc="ZCD RuleFits"):
        # breakpoint()
        feats = [c for c in sub.columns if c.startswith(("mean_zcd", "slope_zcd"))]
        feats_org = [c for c in sub_org.columns if c.startswith(("mean_zcd", "slope_zcd"))]

        if sub["closed"].sum() < 5 or len(sub) - sub["closed"].sum() < 5 or len(sub) < 20:
            continue
        rf_z = train_rulefit(sub[feats], sub["closed"], max_rules=50, cutoff=8)
        
        preds_z = rf_z.predict_proba(sub_org[feats_org])[:, 1]
        preds_zcd[sub_org.index] = preds_z
        zcd_rules_list.append(extract_rules(rf_z, "ZCD", zcd_name))
    if zcd_rules_list:
        all_rules.append(pd.concat(zcd_rules_list, ignore_index=True))

    # --- BZN Models ---
    bzn_rules_list = []
    for ((bzn_name, sub), (_, sub_org)) in tqdm(
        zip(new_pivot_df.dropna(subset=["bzn"]).groupby("bzn"), pivot_df.dropna(subset=["bzn"]).groupby("bzn")), desc="BZN RuleFits"
    ):
        feats = [c for c in sub.columns if c.startswith(("mean_bzn", "slope_bzn"))]
        feats_org = [c for c in sub_org.columns if c.startswith(("mean_bzn", "slope_bzn"))]

        if sub["closed"].sum() < 5 or len(sub) - sub["closed"].sum() < 5 or len(sub) < 20:
            continue
        rf_b = train_rulefit(sub[feats], sub["closed"], max_rules=50, cutoff=8)


        preds_b = rf_b.predict_proba(sub_org[feats_org])[:, 1]
        preds_bzn[sub_org.index] = preds_b
        bzn_rules_list.append(extract_rules(rf_b, "BZN", bzn_name))
    if bzn_rules_list:
        all_rules.append(pd.concat(bzn_rules_list, ignore_index=True))

    # --- Combine all rules into one DataFrame ---
    rulebook = pd.concat(all_rules, ignore_index=True)

    # --- Stack predictions into meta-model ---
    meta_df = pd.DataFrame(
        {"p_general": preds_general, "p_zcd": preds_zcd, "p_bzn": preds_bzn}
    ).fillna(0.0)
    meta_y = y_all

    meta_model = LogisticRegression(class_weight="balanced")
    meta_model.fit(meta_df, meta_y)
    final_probs = meta_model.predict_proba(meta_df)[:, 1]

    pivot_df["p_general"] = preds_general
    pivot_df["p_zcd"] = preds_zcd
    pivot_df["p_bzn"] = preds_bzn
    pivot_df["p_final"] = final_probs

    return {
        "rulebook": rulebook,
        "meta_model": meta_model,
        "results": pivot_df[
            ["mct", "zcd", "bzn", "closed", "p_general", "p_zcd", "p_bzn", "p_final"]
        ],
    }


# === STEP 5. EXECUTE PIPELINE ===
# results = rulefit_pipeline(df)

if __name__ == "__main__":
    from sklearn.metrics import f1_score, confusion_matrix

    df = pd.read_csv("data/normalized_all_mcts.csv")
    results = rulefit_pipeline(df)
    print(
        "F1 : ",
        f1_score(results["results"]["closed"], results["results"]["p_final"] > 0.5),
    )
    print(
        "Confusion Matrix:\n",
        confusion_matrix(
            results["results"]["closed"], results["results"]["p_final"] > 0.5
        ),
    )
    breakpoint()
