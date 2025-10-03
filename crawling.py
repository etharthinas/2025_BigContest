import pandas as pd

df1 = pd.read_csv("data/address.csv")
df2 = pd.read_csv("second.csv")

merged = df1.merge(
    df2[["ENCODED_MCT", "url"]],
    on="ENCODED_MCT",
    how="left",
    suffixes=("", "_new")
)

# Overwrite url with df2's url where available
merged["url"] = merged["url_new"].combine_first(merged["url"])

# Drop the helper column
merged = merged.drop(columns=["url_new"])
breakpoint()