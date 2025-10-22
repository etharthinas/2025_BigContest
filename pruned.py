# 파일 이름: bc_final_pruned_evaluation_v2.py
import logging
import re
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from imodels import RuleFitClassifier
from tqdm import tqdm

# --- 로깅 설정 ---
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- ATTRIBUTE_KOR_MAP ---
ATTRIBUTE_KOR_MAP = {
    "mean_all_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "mean_all_M12_FME_30_RAT": "여성 30대 고객 비중",
    "mean_all_M12_FME_40_RAT": "여성 40대 고객 비중",
    "mean_all_M12_FME_50_RAT": "여성 50대 고객 비중",
    "mean_all_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "mean_all_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "mean_all_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "mean_all_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "mean_all_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "mean_all_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "mean_all_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "mean_all_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "mean_all_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "mean_all_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "mean_all_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "mean_all_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "mean_all_RC_M1_AV_NP_AT": "객단가 구간",
    "mean_all_RC_M1_SAA": "매출금액 구간",
    "mean_all_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "mean_all_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "mean_all_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "mean_all_RC_M1_TO_UE_CT": "매출건수 구간",
    "mean_all_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "slope_all_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "slope_all_M12_FME_30_RAT": "여성 30대 고객 비중",
    "slope_all_M12_FME_40_RAT": "여성 40대 고객 비중",
    "slope_all_M12_FME_50_RAT": "여성 50대 고객 비중",
    "slope_all_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "slope_all_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "slope_all_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "slope_all_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "slope_all_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "slope_all_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "slope_all_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "slope_all_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "slope_all_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "slope_all_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "slope_all_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "slope_all_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "slope_all_RC_M1_AV_NP_AT": "객단가 구간",
    "slope_all_RC_M1_SAA": "매출금액 구간",
    "slope_all_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "slope_all_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "slope_all_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "slope_all_RC_M1_TO_UE_CT": "매출건수 구간",
    "slope_all_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "mean_zcd_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "mean_zcd_M12_FME_30_RAT": "여성 30대 고객 비중",
    "mean_zcd_M12_FME_40_RAT": "여성 40대 고객 비중",
    "mean_zcd_M12_FME_50_RAT": "여성 50대 고객 비중",
    "mean_zcd_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "mean_zcd_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "mean_zcd_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "mean_zcd_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "mean_zcd_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "mean_zcd_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "mean_zcd_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "mean_zcd_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "mean_zcd_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "mean_zcd_RC_M1_AV_NP_AT": "객단가 구간",
    "mean_zcd_RC_M1_SAA": "매출금액 구간",
    "mean_zcd_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "mean_zcd_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "mean_zcd_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "mean_zcd_RC_M1_TO_UE_CT": "매출건수 구간",
    "slope_zcd_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "slope_zcd_M12_FME_30_RAT": "여성 30대 고객 비중",
    "slope_zcd_M12_FME_40_RAT": "여성 40대 고객 비중",
    "slope_zcd_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "slope_zcd_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "slope_zcd_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "slope_zcd_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "slope_zcd_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "slope_zcd_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "slope_zcd_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "slope_zcd_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "slope_zcd_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "slope_zcd_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "slope_zcd_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "slope_zcd_RC_M1_AV_NP_AT": "객단가 구간",
    "slope_zcd_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "slope_zcd_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "slope_zcd_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "slope_zcd_RC_M1_TO_UE_CT": "매출건수 구간",
    "slope_zcd_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "slope_zcd_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "slope_zcd_RC_M1_SAA": "매출금액 구간",
    "mean_zcd_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "mean_zcd_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "mean_zcd_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "slope_zcd_M12_FME_50_RAT": "여성 50대 고객 비중",
    "mean_zcd_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "mean_bzn_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "mean_bzn_M12_FME_30_RAT": "여성 30대 고객 비중",
    "mean_bzn_M12_FME_40_RAT": "여성 40대 고객 비중",
    "mean_bzn_M12_FME_50_RAT": "여성 50대 고객 비중",
    "mean_bzn_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "mean_bzn_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "mean_bzn_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "mean_bzn_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "mean_bzn_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "mean_bzn_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "mean_bzn_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "mean_bzn_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "mean_bzn_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "mean_bzn_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "mean_bzn_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "mean_bzn_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "mean_bzn_RC_M1_AV_NP_AT": "객단가 구간",
    "mean_bzn_RC_M1_SAA": "매출금액 구간",
    "mean_bzn_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "mean_bzn_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "mean_bzn_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "mean_bzn_RC_M1_TO_UE_CT": "매출건수 구간",
    "slope_bzn_M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "slope_bzn_M12_FME_30_RAT": "여성 30대 고객 비중",
    "slope_bzn_M12_FME_40_RAT": "여성 40대 고객 비중",
    "slope_bzn_M12_FME_50_RAT": "여성 50대 고객 비중",
    "slope_bzn_M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "slope_bzn_M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "slope_bzn_M12_MAL_30_RAT": "남성 30대 고객 비중",
    "slope_bzn_M12_MAL_40_RAT": "남성 40대 고객 비중",
    "slope_bzn_M12_MAL_50_RAT": "남성 50대 고객 비중",
    "slope_bzn_M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "slope_bzn_M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "slope_bzn_M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "slope_bzn_M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "slope_bzn_M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "slope_bzn_MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "slope_bzn_MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "slope_bzn_RC_M1_AV_NP_AT": "객단가 구간",
    "slope_bzn_RC_M1_SAA": "매출금액 구간",
    "slope_bzn_RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율",
    "slope_bzn_RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "slope_bzn_RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "slope_bzn_RC_M1_TO_UE_CT": "매출건수 구간",
    "slope_bzn_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "mean_bzn_RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
}


# --- 데이터 준비 ---
def prepare_data(df):
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
        ],
    )
    new_cols = []
    for col_tuple in pivot_df.columns:
        if all(isinstance(item, str) for item in col_tuple):
            new_cols.append("_".join(col_tuple).strip())
        else:
            LOGGER.warning(f"Unexpected column name format: {col_tuple}. Skipping.")
    pivot_df.columns = new_cols
    pivot_df = pivot_df.reset_index()
    return pivot_df


# --- RuleFit 모델 학습 및 룰 추출 ---
def train_rulefit_and_extract(
    X, y, random_state=42
) -> Tuple[RuleFitClassifier, pd.DataFrame]:
    # RuleFitClassifier 인스턴스 생성 시 에러 방지를 위해 필요한 파라미터만 전달
    rf = RuleFitClassifier(
        # tree_size=4, # 원본 파라미터 유지 필요 시 활성화
        # sample_fract=0.75, # 원본 파라미터 유지 필요 시 활성화
        # memory_par=0.01, # 원본 파라미터 유지 필요 시 활성화
        # tree_generator=None, # 원본 파라미터 유지 필요 시 활성화
        # lin_trim_quantile=0.025, # 원본 파라미터 유지 필요 시 활성화
        # lin_standardise=True, # 원본 파라미터 유지 필요 시 활성화
        # exp_rand_tree_size=True, # 원본 파라미터 유지 필요 시 활성화
        # include_linear=True, # 원본 파라미터 유지 필요 시 활성화
        # cv=3, # 원본 파라미터 유지 필요 시 활성화
        random_state=random_state
    )

    try:
        rf.fit(X, y)  # 가중치 없이 학습
    except ValueError as ve:
        if "Invalid alpha and max_rules passed" in str(ve):
            LOGGER.error(f"RuleFit 학습 중 내부 오류 (Invalid alpha/max_rules): {ve}")
            return rf, pd.DataFrame()
        else:
            LOGGER.error(f"RuleFit 학습 중 예상치 못한 ValueError: {ve}", exc_info=True)
            raise  # 다른 ValueError는 예외를 발생시켜 디버깅
    except Exception as e:
        LOGGER.error(f"RuleFit 학습 중 오류 발생: {e}", exc_info=True)
        return rf, pd.DataFrame()

    try:
        # [수정됨] _get_rules() 사용
        rules_df = rf._get_rules()
        rules_df = rules_df[rules_df.coef != 0].sort_values("support", ascending=False)
    except AttributeError:
        try:
            # _get_rules가 없을 경우의 대체 시도 (라이브러리 버전에 따라 다를 수 있음)
            rules_df = pd.DataFrame(rf.rules_)  # 예시적인 대체 속성
            rules_df = rules_df[rules_df.coef != 0].sort_values(
                "support", ascending=False
            )
            LOGGER.warning("rf._get_rules()를 찾지 못해 rf.rules_를 사용했습니다.")
        except AttributeError:
            LOGGER.error(
                f"룰 추출 실패: _get_rules, rules_ 모두 찾을 수 없음.", exc_info=True
            )
            return rf, pd.DataFrame()
    except Exception as e:
        LOGGER.error(f"룰 추출 중 오류 발생: {e}", exc_info=True)
        return rf, pd.DataFrame()

    # 추출된 rules_df가 비어있지 않은지 확인
    if rules_df.empty:
        LOGGER.warning("추출된 규칙이 없습니다.")

    return rf, rules_df


# --- 룰북 정제 로직 ---
# (이전 코드와 동일하게 유지)
def prune_rulebook(rulebook: pd.DataFrame) -> pd.DataFrame:
    if rulebook.empty:
        LOGGER.warning("정제할 룰북이 비어 있습니다.")
        return rulebook
    initial_count = len(rulebook)
    LOGGER.info(f"--- 룰북 정제 시작 ---")
    LOGGER.info(f"정제 전 룰 개수: {initial_count}개")
    df_pruned = rulebook[rulebook["coef"].abs() >= 0.0005].copy()
    count_after_step1 = len(df_pruned)
    removed_step1 = initial_count - count_after_step1
    LOGGER.info(
        f"[1단계 Coef=0 제거] 완료: {removed_step1}개 제거, {count_after_step1}개 남음"
    )
    if df_pruned.empty:
        LOGGER.warning("1단계 정제 후 남은 룰이 없습니다.")
        return df_pruned

    # [수정됨] 정규 표현식 수정: 더 많은 공백 허용 및 소수점 처리 개선
    rule_parser_re = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*([<>=!]+)\s*(-?\d*\.?\d+)\s*$")

    def parse_rule(rule_str):
        match = rule_parser_re.match(str(rule_str))
        if match:
            groups = match.groups()
            if len(groups) == 3:
                var, op, thresh_str = groups
                try:
                    return var, op, float(thresh_str)
                except ValueError:
                    return str(rule_str).strip(), None, None
            else:
                return str(rule_str).strip(), None, None
        else:
            return str(rule_str).strip(), None, None  # 선형 규칙

    parsed = df_pruned["rule"].apply(parse_rule)
    df_pruned["var_name"] = parsed.apply(
        lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None
    )
    df_pruned["op"] = parsed.apply(
        lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None
    )
    df_pruned["thresh"] = parsed.apply(
        lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else None
    )

    high_is_good_vars = [
        "RC_M1_SAA",
        "RC_M1_AV_NP_AT",
        "DLV_SAA_RAT",
        "M1_SME_RY_SAA_RAT",
        "M12_SME_RY_SAA_PCE_RT",
        "M12_SME_BZN_SAA_PCE_RT",
        "RC_M1_TO_UE_CT",
        "RC_M1_UE_CUS_CN",
        "M1_SME_RY_CNT_RAT",
        "MCT_UE_CLN_REU_RAT",
        "MCT_OPE_MS_CN",
    ]
    low_is_good_vars = ["APV_CE_RAT", "M12_SME_RY_ME_MCT_RAT", "M12_SME_BZN_ME_MCT_RAT"]
    customer_ratio_vars = [
        "M12_MAL_1020_RAT",
        "M12_MAL_30_RAT",
        "M12_MAL_40_RAT",
        "M12_MAL_50_RAT",
        "M12_MAL_60_RAT",
        "M12_FME_1020_RAT",
        "M12_FME_30_RAT",
        "M12_FME_40_RAT",
        "M12_FME_50_RAT",
        "M12_FME_60_RAT",
    ]

    def is_rational_v2(row):
        # [수정됨] var_name이 None이거나 비어있으면 합리적인 것으로 간주
        if pd.isna(row["var_name"]) or not row["var_name"]:
            return True
        coef, var_name, op = row["coef"], row["var_name"], row["op"]
        # [수정됨] var_base 추출 로직 강화
        var_base_match = re.match(r"^(?:mean|slope)_(?:all|zcd|bzn)_(.*)", var_name)
        var_base = var_base_match.group(1) if var_base_match else var_name
        is_slope = var_name.startswith("slope_")
        if op is None:
            return True
        if var_base in high_is_good_vars:
            if coef > 0 and op in (">", ">="):
                return False
            if coef < 0 and op in ("<", "<="):
                return False
        if var_base in low_is_good_vars:
            if coef > 0 and op in ("<", "<="):
                return False
            if coef < 0 and op in (">", ">="):
                return False
        if var_base in customer_ratio_vars and is_slope:
            if coef > 0 and op in (">", ">="):
                return False
            if coef < 0 and op in ("<", "<="):
                return False
        return True

    df_pruned["is_rational"] = df_pruned.apply(is_rational_v2, axis=1)
    cols_to_drop = [
        col
        for col in ["var_name", "op", "thresh", "is_rational"]
        if col in df_pruned.columns
    ]
    df_final = df_pruned[df_pruned["is_rational"]].drop(columns=cols_to_drop)
    count_after_step2 = len(df_final)
    removed_step2 = count_after_step1 - count_after_step2
    LOGGER.info(
        f"[2단계 합리성 검사] 완료: {removed_step2}개 제거, {count_after_step2}개 남음"
    )
    LOGGER.info(f"--- 룰북 정제 완료 ---")
    return df_final


# --- 정제된 룰북으로 예측 확률 계산 ---
# (이전 코드와 동일하게 유지)
def _sigmoid(values: np.ndarray) -> np.ndarray:
    # [수정됨] np.exp의 overflow 방지
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def _evaluate_rule_expression(rule_expr: str, feature_df: pd.DataFrame) -> np.ndarray:
    if not rule_expr or rule_expr in {"1", "True"}:
        return np.ones(len(feature_df), dtype=bool)
    normalized_expr = rule_expr.replace(" & ", " and ")
    tokens = [
        token.strip() for token in normalized_expr.split(" and ") if token.strip()
    ]
    mask = np.ones(len(feature_df), dtype=bool)
    operators = [
        ("<=", np.less_equal),
        (">=", np.greater_equal),
        ("==", np.equal),
        ("<", np.less),
        (">", np.greater),
    ]
    feature_cols = set(feature_df.columns)
    for token in tokens:
        matched = False
        for symbol, op in operators:
            if symbol in token:
                try:
                    parts = token.split(symbol, 1)
                    if len(parts) == 2:
                        feature_name, threshold_text = parts
                        feature_name = feature_name.strip()
                        threshold = float(threshold_text.strip())
                        if feature_name not in feature_cols:
                            mask &= False
                            matched = True
                            break
                        # [수정됨] 안전한 연산을 위해 nan 처리
                        mask &= op(
                            feature_df[feature_name].fillna(0).to_numpy(), threshold
                        )
                        matched = True
                        break
                    else:
                        mask &= False
                        matched = True
                        break
                except ValueError:
                    mask &= False
                    matched = True
                    break
                except Exception as e:
                    LOGGER.error(
                        f"Rule eval error token='{token}', rule='{rule_expr}': {e}"
                    )
                    mask &= False
                    matched = True
                    break  # 상세 에러 로깅
        if not matched:
            mask &= False
    return mask


def _build_rule_feature_matrix(
    rules: pd.DataFrame, feature_df: pd.DataFrame
) -> np.ndarray:
    n_samples, n_rules = len(feature_df), len(rules)
    matrix = np.zeros((n_samples, n_rules), dtype=float)
    feature_cols = set(feature_df.columns)
    has_type_col = "type" in rules.columns
    for col_idx, (_, row) in enumerate(rules.iterrows()):
        rule_expr = row.get("rule", "").strip()
        rule_type = row.get("type", "").lower() if has_type_col else ""
        is_linear = False
        if rule_expr:
            is_linear = (rule_type == "linear") or not any(
                op in rule_expr for op in ("<=", ">=", "==", "<", ">")
            )
        if is_linear:
            feature_name = rule_expr
            if feature_name in feature_cols:
                # [수정됨] 안전한 연산을 위해 nan 처리
                matrix[:, col_idx] = (
                    feature_df[feature_name].fillna(0).to_numpy(dtype=float)
                )
            continue
        if rule_expr:
            matrix[:, col_idx] = _evaluate_rule_expression(
                rule_expr, feature_df
            ).astype(float)
    return matrix


def predict_proba_from_pruned_rules(
    pruned_rulebook: pd.DataFrame, X: pd.DataFrame
) -> np.ndarray:
    if pruned_rulebook.empty:
        return np.zeros(len(X))
    if X.empty:
        return np.zeros(0)
    rule_features = set()
    # [수정됨] 'rule' 컬럼이 없을 경우 대비
    if "rule" in pruned_rulebook.columns:
        for rule_str in pruned_rulebook["rule"].astype(str):  # astype(str) 추가
            match = re.match(r"^\s*([a-zA-Z0-9_]+)", rule_str)
            if match:
                rule_features.add(match.group(1))
    missing_features = rule_features - set(X.columns)
    rule_matrix = _build_rule_feature_matrix(pruned_rulebook, X)
    coefs = pruned_rulebook["coef"].to_numpy(dtype=float)
    if rule_matrix.shape[1] != len(coefs):
        LOGGER.error(
            f"Matrix shape ({rule_matrix.shape}) vs coef length({len(coefs)}) mismatch."
        )
        return np.zeros(len(X))
    scores = rule_matrix @ coefs
    probabilities = _sigmoid(scores)
    return probabilities


# === 메인 파이프라인 ===
def rulefit_pipeline_pruned(df):
    try:
        pivot_df = prepare_data(df)
        cols_to_drop = ["closed", "mct", "zcd", "bzn"]
        cols_exist = [col for col in cols_to_drop if col in pivot_df.columns]
        X_all = pivot_df.drop(columns=cols_exist)
        if "closed" not in pivot_df.columns:
            LOGGER.error("Target 'closed' not in pivot_df.")
            return None
        y_all = pivot_df["closed"]
    except Exception as e:
        LOGGER.error(f"Data prep error: {e}", exc_info=True)
        return None

    all_original_rules = []
    LOGGER.info("--- 1단계: 모델 학습 및 룰 추출 시작 ---")
    general_feats = [
        c for c in X_all.columns if c.startswith(("mean_all", "slope_all"))
    ]
    if not general_feats:
        LOGGER.warning("General 모델 학습 피처 없음.")
        rules_gen = pd.DataFrame()
    else:
        _, rules_gen = train_rulefit_and_extract(X_all[general_feats], y_all)
    if isinstance(rules_gen, pd.DataFrame) and not rules_gen.empty:
        rules_gen["model_type"] = "general"
        all_original_rules.append(rules_gen)

    zcd_rules_list = []
    for zcd_name, sub in tqdm(pivot_df.groupby("zcd"), desc="ZCD RuleFits"):
        sub_cols_to_drop = [col for col in cols_to_drop if col in sub.columns]
        # [수정됨] sub_X_all 생성 시 안전하게 처리
        if not sub.empty:
            sub_X_all = sub.drop(columns=sub_cols_to_drop, errors="ignore")
        else:
            continue  # 빈 그룹 스킵

        feats = [
            c for c in sub_X_all.columns if c.startswith(("mean_zcd", "slope_zcd"))
        ]
        if (
            "closed" not in sub.columns
            or sub["closed"].sum() < 2
            or len(sub) < 20
            or not feats
        ):
            continue
        _, rules_z = train_rulefit_and_extract(sub_X_all[feats], sub["closed"])
        if isinstance(rules_z, pd.DataFrame) and not rules_z.empty:
            rules_z["model_type"] = "zcd"
            rules_z["category"] = zcd_name
            zcd_rules_list.append(rules_z)
    if zcd_rules_list:
        all_original_rules.append(pd.concat(zcd_rules_list, ignore_index=True))

    bzn_rules_list = []
    for bzn_name, sub in tqdm(
        pivot_df.dropna(subset=["bzn"]).groupby("bzn"), desc="BZN RuleFits"
    ):
        sub_cols_to_drop = [col for col in cols_to_drop if col in sub.columns]
        if not sub.empty:
            sub_X_all = sub.drop(columns=sub_cols_to_drop, errors="ignore")
        else:
            continue

        feats = [
            c for c in sub_X_all.columns if c.startswith(("mean_bzn", "slope_bzn"))
        ]
        if (
            "closed" not in sub.columns
            or sub["closed"].sum() < 2
            or len(sub) < 20
            or not feats
        ):
            continue
        _, rules_b = train_rulefit_and_extract(sub_X_all[feats], sub["closed"])
        if isinstance(rules_b, pd.DataFrame) and not rules_b.empty:
            rules_b["model_type"] = "bzn"
            rules_b["category"] = bzn_name
            bzn_rules_list.append(rules_b)
    if bzn_rules_list:
        all_original_rules.append(pd.concat(bzn_rules_list, ignore_index=True))

    if not all_original_rules:
        LOGGER.error("추출된 규칙이 없어 룰북을 정제할 수 없습니다.")
        return None
    combined_rulebook = pd.concat(all_original_rules, ignore_index=True).reset_index(
        drop=True
    )
    pruned_rulebook = prune_rulebook(combined_rulebook)

    pruned_rulebook_path = "rulebook_pruned_final.csv"
    pruned_rulebook.to_csv(
        pruned_rulebook_path, index=False, float_format="%.3f", encoding="utf-8-sig"
    )
    LOGGER.info(
        f"최종 정제된 룰북 ({len(pruned_rulebook)}개)을 '{pruned_rulebook_path}'로 저장했습니다."
    )

    LOGGER.info("--- 3단계: 정제된 룰북으로 확률 재계산 시작 ---")
    preds_general_pruned = np.zeros(len(pivot_df))
    preds_zcd_pruned = np.zeros(len(pivot_df))
    preds_bzn_pruned = np.zeros(len(pivot_df))
    rules_gen_pruned = pruned_rulebook[pruned_rulebook["model_type"] == "general"]
    if not rules_gen_pruned.empty and general_feats:
        preds_general_pruned = predict_proba_from_pruned_rules(
            rules_gen_pruned, X_all[general_feats]
        )

    rules_zcd_pruned = pruned_rulebook[pruned_rulebook["model_type"] == "zcd"]
    if not rules_zcd_pruned.empty:
        for zcd_name, sub_indices in pivot_df.groupby("zcd").groups.items():
            sub_rules = rules_zcd_pruned[rules_zcd_pruned["category"] == zcd_name]
            if not sub_rules.empty:
                zcd_cols = [
                    c for c in X_all.columns if c.startswith(("mean_zcd", "slope_zcd"))
                ]
                if zcd_cols:
                    sub_X = X_all.loc[sub_indices, zcd_cols]
                    if not sub_X.empty:
                        preds_zcd_pruned[sub_indices] = predict_proba_from_pruned_rules(
                            sub_rules, sub_X
                        )

    rules_bzn_pruned = pruned_rulebook[pruned_rulebook["model_type"] == "bzn"]
    if not rules_bzn_pruned.empty:
        bzn_groups = pivot_df.dropna(subset=["bzn"]).groupby("bzn")
        for bzn_name, sub_indices in bzn_groups.groups.items():
            sub_rules = rules_bzn_pruned[rules_bzn_pruned["category"] == bzn_name]
            if not sub_rules.empty:
                bzn_cols = [
                    c for c in X_all.columns if c.startswith(("mean_bzn", "slope_bzn"))
                ]
                if bzn_cols:
                    sub_X = X_all.loc[sub_indices, bzn_cols]
                    if not sub_X.empty:
                        preds_bzn_pruned[sub_indices] = predict_proba_from_pruned_rules(
                            sub_rules, sub_X
                        )

    LOGGER.info("--- 4단계: 메타 모델 학습 및 평가 시작 ---")
    meta_df_pruned = pd.DataFrame(
        {
            "p_general": preds_general_pruned,
            "p_zcd": preds_zcd_pruned,
            "p_bzn": preds_bzn_pruned,
        }
    ).fillna(0.0)
    meta_y = y_all
    meta_model_pruned = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta_model_pruned.fit(meta_df_pruned, meta_y)
    final_probs_pruned = meta_model_pruned.predict_proba(meta_df_pruned)[:, 1]
    final_preds_pruned = meta_model_pruned.predict(meta_df_pruned)

    f1_pruned = f1_score(meta_y, final_preds_pruned)
    cm_pruned = confusion_matrix(meta_y, final_preds_pruned)
    LOGGER.info(f"--- 최종 성능 (정제된 룰북 + 메타 모델) ---")
    LOGGER.info(f"Meta-model F1: {f1_pruned:.4f}")
    LOGGER.info(f"Meta-model confusion matrix:\n{cm_pruned}")
    LOGGER.info("---------------------------------------------")

    # pivot_df에 결과 컬럼 추가 전 존재 여부 확인 후 추가
    pivot_df["p_general_pruned"] = preds_general_pruned
    pivot_df["p_zcd_pruned"] = preds_zcd_pruned
    pivot_df["p_bzn_pruned"] = preds_bzn_pruned
    pivot_df["p_final_pruned"] = final_probs_pruned

    # 'results' 키에 포함될 컬럼 정의 및 확인
    results_cols = [
        "mct",
        "zcd",
        "bzn",
        "closed",
        "p_general_pruned",
        "p_zcd_pruned",
        "p_bzn_pruned",
        "p_final_pruned",
    ]
    available_results_cols = [c for c in results_cols if c in pivot_df.columns]

    return {
        "pruned_rulebook": pruned_rulebook,
        "meta_model": meta_model_pruned,
        "results": pivot_df[available_results_cols],  # 사용 가능한 컬럼만 선택
        "pivot_df": pivot_df,
    }


# === 스크립트 실행 ===
if __name__ == "__main__":
    try:
        input_csv_path = "normalized_all_mcts.csv"
        df_input = pd.read_csv(input_csv_path)
        LOGGER.info(f"입력 파일 '{input_csv_path}' 로드 완료.")
        final_results = rulefit_pipeline_pruned(df_input)
        if final_results:
            pivot_output_path = "pivot_df_final_pruned.csv"
            cols_to_save = [
                "mct",
                "zcd",
                "bzn",
                "closed",
                "p_general_pruned",
                "p_zcd_pruned",
                "p_bzn_pruned",
                "p_final_pruned",
            ]
            final_pivot_df = final_results["pivot_df"]
            cols_exist_in_final = [
                c for c in cols_to_save if c in final_pivot_df.columns
            ]
            if cols_exist_in_final:
                # [수정됨] float_format=".4f" -> "%.4f"
                final_pivot_df[cols_exist_in_final].to_csv(
                    pivot_output_path,
                    index=False,
                    encoding="utf-8-sig",
                    float_format="%.4f",
                )
                LOGGER.info(
                    f"최종 결과 포함된 피벗 테이블을 '{pivot_output_path}'로 저장했습니다."
                )
            else:
                LOGGER.warning("저장할 결과 컬럼이 없어 파일 저장을 스킵합니다.")
        else:
            LOGGER.error("최종 결과 생성 실패로 파일 저장을 스킵합니다.")
    except FileNotFoundError:
        LOGGER.error(f"오류: 입력 파일 '{input_csv_path}'를 찾을 수 없습니다.")
    except Exception as e:
        LOGGER.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
