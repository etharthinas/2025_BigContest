import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import resample
from imodels import RuleFitClassifier
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

TOP_RULE_LIMIT = 500
POSITIVE_MULTIPLIER = (
    3  # Duplicate closed=1 rows by this factor before training each RuleFit
)


def _oversample_positive(df: pd.DataFrame) -> pd.DataFrame:
    if POSITIVE_MULTIPLIER <= 1:
        return df
    positives = df[df["closed"] == 1]
    if positives.empty:
        return df
    extras = resample(
        positives,
        replace=True,
        n_samples=len(positives) * (POSITIVE_MULTIPLIER - 1),
        random_state=42,
    )
    return pd.concat([df, extras], ignore_index=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bzn"] = df["bzn"].replace({np.nan: None})
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
            "r2_all",
            "r2_zcd",
            "r2_bzn",
        ],
    )
    pivot_df.columns = ["_".join(col) for col in pivot_df.columns]
    return pivot_df.reset_index()


def train_rulefit(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> RuleFitClassifier:
    model = RuleFitClassifier(
        tree_size=2,
        sample_fract=1.0,
        max_rules=200,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def extract_rules(
    rf: RuleFitClassifier, rule_type: str, category: Optional[str] = None
) -> pd.DataFrame:
    rules = rf._get_rules()
    rules = rules[(rules.coef != 0) & (rules.support > 0)].copy()
    if "rule_type" in rules.columns:
        rules = rules[~rules["rule_type"].fillna("").str.lower().eq("linear")]
    if "rule" in rules.columns:
        rules = rules[
            rules["rule"].apply(
                lambda expr: (
                    any(token in expr for token in ("<=", ">=", "<", ">", "=="))
                    if isinstance(expr, str)
                    else False
                )
            )
        ]
    rules["type"] = rule_type
    rules["category"] = category
    return rules


def rulefit_pipeline(df: pd.DataFrame) -> dict:
    pivot_df = prepare_data(df)
    X_all = pivot_df.drop(columns=["closed", "mct", "zcd", "bzn"])
    y_all = pivot_df["closed"]

    all_rules = []
    preds_general = np.zeros(len(pivot_df))
    preds_zcd = np.zeros(len(pivot_df))
    preds_bzn = np.zeros(len(pivot_df))

    # General RuleFit with oversampling
    general_df = pivot_df[
        ["closed"]
        + [col for col in pivot_df.columns if col.startswith(("mean_all", "slope_all"))]
    ]
    general_df = _oversample_positive(general_df)
    general_feats = [
        c for c in general_df.columns if c.startswith(("mean_all", "slope_all"))
    ]

    rf_general = train_rulefit(general_df[general_feats], general_df["closed"])
    preds_general = rf_general.predict_proba(X_all[general_feats])[:, 1]
    all_rules.append(extract_rules(rf_general, "general"))

    # ZCD RuleFits
    zcd_rules = []
    for zcd, sub in tqdm(pivot_df.groupby("zcd"), desc="ZCD RuleFits"):
        feats = [c for c in sub.columns if c.startswith(("mean_zcd", "slope_zcd"))]
        if len(feats) == 0 or sub["closed"].sum() < 5 or len(sub) < 20:
            continue
        zcd_train = _oversample_positive(sub[["closed"] + feats])
        rf_z = train_rulefit(zcd_train[feats], zcd_train["closed"])
        preds = rf_z.predict_proba(sub[feats])[:, 1]
        preds_zcd[sub.index] = preds
        zcd_rules.append(extract_rules(rf_z, "ZCD", zcd))
    if zcd_rules:
        all_rules.append(pd.concat(zcd_rules, ignore_index=True))

    # BZN RuleFits
    bzn_rules = []
    pivot_bzn = pivot_df.dropna(subset=["bzn"])
    for bzn, sub in tqdm(pivot_bzn.groupby("bzn"), desc="BZN RuleFits"):
        feats = [c for c in sub.columns if c.startswith(("mean_bzn", "slope_bzn"))]
        if len(feats) == 0 or sub["closed"].sum() < 5 or len(sub) < 20:
            continue
        bzn_train = _oversample_positive(sub[["closed"] + feats])
        rf_b = train_rulefit(bzn_train[feats], bzn_train["closed"])
        preds = rf_b.predict_proba(sub[feats])[:, 1]
        preds_bzn[sub.index] = preds
        bzn_rules.append(extract_rules(rf_b, "BZN", bzn))
    if bzn_rules:
        all_rules.append(pd.concat(bzn_rules, ignore_index=True))

    rulebook = pd.concat(all_rules, ignore_index=True)

    meta_df = pd.DataFrame(
        {
            "p_general": preds_general,
            "p_zcd": preds_zcd,
            "p_bzn": preds_bzn,
        }
    ).fillna(0.0)

    meta_model = LogisticRegression(class_weight="balanced")
    meta_model.fit(meta_df, y_all)
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
        "pivot_df": pivot_df,
        "meta_metrics": {
            "f1": f1_score(y_all, (final_probs >= 0.5).astype(int)),
            "confusion_matrix": confusion_matrix(
                y_all, (final_probs >= 0.5).astype(int)
            ),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = pd.read_csv("data/normalized_all_mcts.csv")
    results = rulefit_pipeline(df)

    LOGGER.info("Meta-model F1: %.4f", results["meta_metrics"]["f1"])
    LOGGER.info(
        "Meta-model confusion matrix:\n%s", results["meta_metrics"]["confusion_matrix"]
    )

    rulebook_path = "rulebook_weighted.csv"
    pivot_path = "pivot_df_weighted.csv"

    results["rulebook"].to_csv(rulebook_path, index=False)
    # results['pivot_df'].to_csv(pivot_path, index=False)

    LOGGER.info("Weighted rulebook written to %s", rulebook_path)
    LOGGER.info("Weighted pivot dataframe written to %s", pivot_path)
    breakpoint()
