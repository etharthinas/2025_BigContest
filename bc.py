import pandas as pd
import numpy as np
# from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
from imodels import RuleFitClassifier
# from scipy.special import expit
from tqdm import tqdm

# === STEP 1. PREPARE DATA ===
def prepare_data(df):
    # replace missing BZN values with None (for clarity)
    df['bzn'] = df['bzn'].replace({np.nan: None})

    # pivot to wide form, one row per mct
    pivot_df = df.pivot(
        index=['mct', 'zcd', 'bzn', 'closed'],
        columns='attribute',
        values=['mean_all','mean_zcd','mean_bzn','slope_all','slope_zcd','slope_bzn']
    )
    pivot_df.columns = ['_'.join(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    return pivot_df

# === STEP 2. RULEFIT TRAINING FUNCTION ===
def train_rulefit(X, y, random_state=42):
    rf = RuleFitClassifier(tree_size=2, sample_fract=1.0, max_rules=200, random_state=random_state)
    rf.fit(X, y)
    return rf

# === STEP 3. RULE EXTRACTION (OPTIONAL INTERPRETABILITY) ===
def extract_rules(rf, rule_type, category=None):
    rules = rf._get_rules()
    rules = rules[(rules.coef != 0) & (rules.support > 0)]
    rules['type'] = rule_type
    rules['category'] = category
    return rules

# === STEP 4. PIPELINE MAIN ===
def rulefit_pipeline(df):
    pivot_df = prepare_data(df)
    X_all = pivot_df.drop(columns=['closed', 'mct', 'zcd', 'bzn'])
    y_all = pivot_df['closed']

    all_rules = []
    preds_general = np.zeros(len(pivot_df))
    preds_zcd = np.zeros(len(pivot_df))
    preds_bzn = np.zeros(len(pivot_df))

    # --- General Model ---
    general_feats = [c for c in X_all.columns if c.startswith(('mean_all', 'slope_all'))]
    rf_gen = train_rulefit(X_all[general_feats], y_all)
    preds_general = rf_gen.predict_proba(X_all[general_feats])[:, 1]
    all_rules.append(extract_rules(rf_gen, 'general'))

    # --- ZCD Models ---
    zcd_rules_list = []
    for zcd_name, sub in tqdm(pivot_df.groupby('zcd'), desc='ZCD RuleFits'):
        feats = [c for c in sub.columns if c.startswith(('mean_zcd', 'slope_zcd'))]
        if sub['closed'].sum() < 3 or len(sub) < 20:
            continue
        rf_z = train_rulefit(sub[feats], sub['closed'])
        preds_z = rf_z.predict_proba(sub[feats])[:, 1]
        preds_zcd[sub.index] = preds_z
        zcd_rules_list.append(extract_rules(rf_z, 'ZCD', zcd_name))
    if zcd_rules_list:
        all_rules.append(pd.concat(zcd_rules_list, ignore_index=True))

    # --- BZN Models ---
    bzn_rules_list = []
    for bzn_name, sub in tqdm(pivot_df.dropna(subset=['bzn']).groupby('bzn'), desc='BZN RuleFits'):
        feats = [c for c in sub.columns if c.startswith(('mean_bzn', 'slope_bzn'))]
        if sub['closed'].sum() < 3 or len(sub) < 20:
            continue
        rf_b = train_rulefit(sub[feats], sub['closed'])
        preds_b = rf_b.predict_proba(sub[feats])[:, 1]
        preds_bzn[sub.index] = preds_b
        bzn_rules_list.append(extract_rules(rf_b, 'BZN', bzn_name))
    if bzn_rules_list:
        all_rules.append(pd.concat(bzn_rules_list, ignore_index=True))

    # --- Combine all rules into one DataFrame ---
    rulebook = pd.concat(all_rules, ignore_index=True)

    # --- Stack predictions into meta-model ---
    meta_df = pd.DataFrame({
        'p_general': preds_general,
        'p_zcd': preds_zcd,
        'p_bzn': preds_bzn
    }).fillna(0.0)
    meta_y = y_all

    meta_model = LogisticRegression(class_weight='balanced')
    meta_model.fit(meta_df, meta_y)
    final_probs = meta_model.predict_proba(meta_df)[:, 1]

    pivot_df['p_general'] = preds_general
    pivot_df['p_zcd'] = preds_zcd
    pivot_df['p_bzn'] = preds_bzn
    pivot_df['p_final'] = final_probs

    return {
        'rulebook': rulebook,
        'meta_model': meta_model,
        'results': pivot_df[['mct','zcd','bzn','closed','p_general','p_zcd','p_bzn','p_final']]
    }

# === STEP 5. EXECUTE PIPELINE ===
# results = rulefit_pipeline(df)

if __name__ == "__main__":
    df = pd.read_csv("data/normalized_all_mcts.csv")
    results = rulefit_pipeline(df)
    breakpoint()