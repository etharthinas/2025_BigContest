import pandas as pd
from collections import Counter

# rules = pd.read_csv("smaller_rulebook.csv")
# attr = rules.sort_values(by="importance")[:500].rule.apply(lambda x: x.split(" ")[0])
# x = Counter(attr)
# breakpoint()


df = pd.read_csv("data/normalized_all_mcts.csv")
df2 = (
    df[["std_all", "std_bzn", "std_zcd", "r2_all", "r2_bzn", "r2_zcd"]]
    .groupby(df.attribute)
    .mean()
)

df = pd.read_csv("38.csv")
breakpoint()

