from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "closed"

SLOPE_TO_R2 = {
    "slope_all": "r2_all",
    "slope_bzn": "r2_bzn",
    "slope_zcd": "r2_zcd",
}


@dataclass
class Rule:
    attribute: str
    bzn: Optional[str]
    zcd: Optional[str]
    feature: str
    direction: str
    threshold: float
    f1: float
    precision: float
    recall: float
    support: int
    group_size: int
    positives: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "attribute": self.attribute,
            "bzn": self.bzn,
            "zcd": self.zcd,
            "feature": self.feature,
            "direction": self.direction,
            "threshold": self.threshold,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "support": self.support,
            "group_size": self.group_size,
            "positives": self.positives,
            "rule": self.expression(),
        }

    def expression(self) -> str:
        parts: List[str] = [f"attribute == {self.attribute!r}"]
        if self.bzn:
            parts.append(f"bzn == {self.bzn!r}")
        if self.zcd:
            parts.append(f"zcd == {self.zcd!r}")
        conditions = " and ".join(parts)
        comparison = f"{self.feature} {self.direction} {self.threshold:.4f}"
        return f"if {conditions}: predict closed=1 when {comparison}"


def load_normalized(normalized_path: str) -> pd.DataFrame:
    normalized_df = pd.read_csv(normalized_path)
    if "closed" not in normalized_df:
        raise ValueError(
            "The normalized data must include a 'closed' column. "
            "Re-run preprocessing to append the target."
        )

    df = normalized_df.dropna(subset=["closed"]).copy()
    df["closed"] = df["closed"].astype(int)
    return df


def _scan_direction(values: np.ndarray, labels: np.ndarray, min_support: int) -> Optional[Dict[str, float]]:
    order = np.argsort(values)
    values_sorted = values[order]
    labels_sorted = labels[order]

    total_pos = labels_sorted.sum()
    if total_pos == 0:
        return None

    unique_values, first_indices = np.unique(values_sorted, return_index=True)
    cumsum_pos = np.cumsum(labels_sorted)
    n = len(values_sorted)

    best: Optional[Dict[str, float]] = None
    for value, idx in zip(unique_values, first_indices):
        tp = total_pos - (cumsum_pos[idx - 1] if idx > 0 else 0)
        pred_pos = n - idx
        if pred_pos < min_support or tp == 0:
            continue

        precision = tp / pred_pos
        recall = tp / total_pos
        if precision == 0 or recall == 0:
            continue

        f1 = 2 * precision * recall / (precision + recall)
        candidate = {
            "threshold": float(value),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": int(pred_pos),
        }
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    return best


def best_threshold(values: np.ndarray, labels: np.ndarray, min_support: int) -> Optional[Dict[str, object]]:
    mask = ~np.isnan(values)
    filtered_values = values[mask]
    filtered_labels = labels[mask]

    if filtered_values.size == 0:
        return None

    positives = filtered_labels.sum()
    if positives == 0 or positives == filtered_labels.size:
        return None

    ge_candidate = _scan_direction(filtered_values, filtered_labels, min_support)
    le_candidate = _scan_direction(-filtered_values, filtered_labels, min_support)

    candidates: List[Dict[str, object]] = []
    if ge_candidate:
        ge_candidate.update({"direction": ">="})
        candidates.append(ge_candidate)
    if le_candidate:
        le_candidate.update({"direction": "<=", "threshold": -le_candidate["threshold"]})
        candidates.append(le_candidate)

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["f1"])


def numeric_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col != target]


def generate_rules(
    df: pd.DataFrame,
    groupings: Optional[Sequence[Sequence[str]]] = None,
    min_group_size: int = 25,
    min_support: int = 5,
    min_r2_for_slope: float = 0.3,
) -> List[Rule]:
    if groupings is None:
        groupings = [
            ("attribute",),
            ("attribute", "bzn"),
            ("attribute", "zcd"),
        ]

    features = numeric_features(df, TARGET_COLUMN)
    rules: List[Rule] = []

    for grouping in groupings:
        grouping_list = list(grouping)
        for key_values, group_df in df.groupby(grouping_list, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_map = dict(zip(grouping_list, key_values))

            skip_group = False
            for key, value in key_map.items():
                if key == "attribute":
                    continue
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    skip_group = True
                    break
            if skip_group:
                continue

            group_size = len(group_df)
            positives = int(group_df[TARGET_COLUMN].sum())
            if group_size < min_group_size or positives == 0:
                continue

            for feature in features:
                feature_df = group_df
                if feature in SLOPE_TO_R2:
                    r2_col = SLOPE_TO_R2[feature]
                    if r2_col not in group_df:
                        continue
                    feature_df = feature_df[feature_df[r2_col] >= min_r2_for_slope]
                feature_values = feature_df[feature].to_numpy(dtype=float, copy=False)
                target_values = feature_df[TARGET_COLUMN].to_numpy(dtype=float, copy=False)

                if feature_values.size < max(min_support, 2):
                    continue

                candidate = best_threshold(feature_values, target_values, min_support)
                if not candidate:
                    continue

                rule = Rule(
                    attribute=key_map.get("attribute", ""),
                    bzn=key_map.get("bzn"),
                    zcd=key_map.get("zcd"),
                    feature=feature,
                    direction=candidate["direction"],
                    threshold=float(candidate["threshold"]),
                    f1=float(candidate["f1"]),
                    precision=float(candidate["precision"]),
                    recall=float(candidate["recall"]),
                    support=int(candidate["support"]),
                    group_size=group_size,
                    positives=positives,
                )
                rules.append(rule)

    return sorted(rules, key=lambda r: r.f1, reverse=True)


def rules_to_dataframe(rules: Iterable[Rule]) -> pd.DataFrame:
    return pd.DataFrame([rule.as_dict() for rule in rules])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate threshold rules that maximise F1 for predicting closures.")
    parser.add_argument("--normalized-path", default="data/normalized_all_mcts.csv", help="Path to the normalized data CSV.")
    parser.add_argument("--output", default="rules.csv", help="Path to save the scored rules as CSV.")
    parser.add_argument("--min-group-size", type=int, default=10, help="Skip groups with fewer rows than this threshold.")
    parser.add_argument("--min-support", type=int, default=5, help="Skip rules that mark fewer rows as positive than this support.")
    parser.add_argument("--min-r2", type=float, default=0.5, help="Minimum r2 required for slope-based features.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of top rules to print when no output path is provided.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    df = load_normalized(args.normalized_path)
    rules = generate_rules(
        df,
        min_group_size=args.min_group_size,
        min_support=args.min_support,
        min_r2_for_slope=args.min_r2,
    )

    if not rules:
        print("No rules generated. Try lowering the group/support thresholds.")
        return

    rules_df = rules_to_dataframe(rules)

    if args.output:
        rules_df.to_csv(args.output, index=False)
    else:
        top_k = min(args.top_k, len(rules_df))
        print(rules_df.head(top_k).to_string(index=False))


if __name__ == "__main__":
    main()
