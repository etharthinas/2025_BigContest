import pandas as pd
DATA_PATH = "./data"

def preprocess(DATA_PATH: str):
    filenames = ["big_data_set1_f.csv", "big_data_set2_f.csv", "big_data_set3_f.csv"]
    dfs = [pd.read_csv(f"{DATA_PATH}/{filename}", encoding='cp949') for filename in filenames]
    mct_info, revenue_info, customer_info = dfs

    merged_df = pd.merge(revenue_info, customer_info, on=['ENCODED_MCT', 'TA_YM'], how='outer')
    merged_df = pd.merge(merged_df, mct_info, on = ['ENCODED_MCT'], how = 'outer')

    merged_df["TA_YM"] = pd.to_datetime(merged_df["TA_YM"], format='%Y%m')
    merged_df["MCT_OPE_MS_CN"] = merged_df["MCT_OPE_MS_CN"].str[0].astype(int)
    merged_df["RC_M1_SAA"] = merged_df["RC_M1_SAA"].str[0].astype(int)
    merged_df["RC_M1_TO_UE_CT"] = merged_df["RC_M1_TO_UE_CT"].str[0].astype(int)
    merged_df["APV_CE_RAT"] = merged_df["APV_CE_RAT"].fillna('').astype(str).str[0]
    merged_df["RC_M1_UE_CUS_CN"] = merged_df["RC_M1_UE_CUS_CN"].str[0].astype(int)
    merged_df["RC_M1_AV_NP_AT"] = merged_df["RC_M1_AV_NP_AT"].str[0].astype(int)
    merged_df["ARE_D"] = pd.to_datetime(merged_df["ARE_D"], format='%Y%m%d')
    merged_df["MCT_ME_D"] = pd.to_datetime(merged_df["MCT_ME_D"].fillna(''), format='%Y%m%d.0', errors='coerce')

    merged_df["closed"] = merged_df["MCT_ME_D"].notna().astype(int)

    return merged_df

def dead_mcts_only(DATA_PATH = DATA_PATH):
    merged_df = preprocess(DATA_PATH)
    dead_mcts = merged_df[merged_df["MCT_ME_D"].notna()]["ENCODED_MCT"].unique().tolist()
    return dead_mcts

def labeled_all(DATA_PATH = DATA_PATH):
    merged_df = preprocess(DATA_PATH)
    labeled_all = merged_df["ENCODED_MCT"].unique().tolist()
    return labeled_all