import pandas as pd
import numpy as np
from preprocess import preprocess, DATA_PATH, dead_mcts_only


class MCT:
    def __init__(self, name, DATA_PATH=DATA_PATH):
        self.name = name
        self.merged_df = preprocess(DATA_PATH)
        self.bzn_cd_nm = self.merged_df[self.merged_df["ENCODED_MCT"] == name][
            "HPSN_MCT_BZN_CD_NM"
        ].values[0]
        self.zcd_nm = self.merged_df[self.merged_df["ENCODED_MCT"] == name][
            "HPSN_MCT_ZCD_NM"
        ].values[0]

    def get_time_attribute(self, attribute):
        return (
            self.merged_df[self.merged_df["ENCODED_MCT"] == self.name][
                [attribute, "TA_YM"]
            ]
            .groupby("TA_YM")
            .mean()
        )

    def normalize(self, attribute, by=["bzn", "zcd", "all"]) -> pd.DataFrame:
        if by == "bzn":
            same = self.merged_df[
                self.merged_df["HPSN_MCT_BZN_CD_NM"] == self.bzn_cd_nm
            ][[attribute, "TA_YM"]]
        elif by == "zcd":
            same = self.merged_df[self.merged_df["HPSN_MCT_ZCD_NM"] == self.zcd_nm][
                [attribute, "TA_YM"]
            ]
        elif by == "all":
            same = self.merged_df[[attribute, "TA_YM"]]
        else:
            raise ValueError()

        avg = same.groupby("TA_YM").mean()
        std = same.groupby("TA_YM").std()
        normalized = ((self.get_time_attribute(attribute) - avg) / std).fillna(
            0
        )  # if std is NaN
        return normalized

    def generate_comparison_tuple(self, attribute) -> dict:
        normalized_all = self.normalize(attribute, by="all")
        normalized_bzn = self.normalize(attribute, by="bzn")
        normalized_zcd = self.normalize(attribute, by="zcd")

        mean_all = normalized_all.mean(axis=0)
        mean_bzn = normalized_bzn.mean(axis=0)
        mean_zcd = normalized_zcd.mean(axis=0)

        std_all = normalized_all.std(axis=0)
        std_bzn = normalized_bzn.std(axis=0)
        std_zcd = normalized_zcd.std(axis=0)

        slope_all, r2_all = self.slope(normalized_all[attribute])
        slope_bzn, r2_bzn = self.slope(normalized_bzn[attribute])
        slope_zcd, r2_zcd = self.slope(normalized_zcd[attribute])

        return {
            "mean_all": mean_all.values[0],
            "mean_bzn": mean_bzn.values[0],
            "mean_zcd": mean_zcd.values[0],
            "std_all": std_all.values[0],
            "std_bzn": std_bzn.values[0],
            "std_zcd": std_zcd.values[0],
            "slope_all": slope_all,
            "slope_bzn": slope_bzn,
            "slope_zcd": slope_zcd,
            "r2_all": r2_all,
            "r2_bzn": r2_bzn,
            "r2_zcd": r2_zcd,
            "mct": self.name,
            "attribute": attribute,
            "bzn": self.bzn_cd_nm,
            "zcd": self.zcd_nm,
        }

    def slope(self, df: pd.DataFrame) -> tuple[float, float]:
        x = np.arange(len(df))
        y = df.values.flatten()

        if len(y) < 2 or np.all(y == y[0]):
            return 0.0, 0.0  # slope=0, R²=0 if no variation

        m = np.polyfit(x, y, 1)[0]

        # compute R² safely
        y_pred = m * x + np.polyfit(x, y, 1)[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return m, r2
