from mct import MCT
import pandas as pd
from tqdm import tqdm
from preprocess import dead_mcts_only

def normalize_dead_mcts():
    dead_mcts = dead_mcts_only()
    data_df = list()

    columns = [
        "RC_M1_SAA",
        "RC_M1_AV_NP_AT",
        "RC_M1_UE_CUS_CN",
        "RC_M1_TO_UE_CT",
        "M1_SME_RY_SAA_RAT",
        "M12_SME_RY_SAA_PCE_RT",
        "M1_SME_RY_CNT_RAT",
        "M12_SME_BZN_SAA_PCE_RT",
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
        "MCT_UE_CLN_REU_RAT",
        "MCT_UE_CLN_NEW_RAT",
        "RC_M1_SHC_RSD_UE_CLN_RAT",
        "RC_M1_SHC_WP_UE_CLN_RAT",
        "RC_M1_SHC_FLP_UE_CLN_RAT"
    ]

    for mct_name in tqdm(dead_mcts):
        for attribute in columns:
            normalized_data = MCT(mct_name).generate_comparison_tuple(attribute)
            normalized_data["ENCODED_MCT"] = mct_name
            normalized_data["attribute"] = attribute
            data_df.append(normalized_data)
    df = pd.DataFrame(data_df)
    df.to_csv("normalized_dead_mcts.csv", index=False)


if __name__ == "__main__":
    normalize_dead_mcts()