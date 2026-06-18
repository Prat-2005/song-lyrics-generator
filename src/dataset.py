import pandas as pd

files = [
    "../data/Beyonce.csv",
    "../data/DuaLipa.csv",
    "../data/Eminem.csv",
    "../data/JustinBieber.csv",
    "../data/TaylorSwift.csv",
    "../data/Drake.csv",
    "../data/NickiMinaj.csv",
    "../data/ColdPlay.csv",
    "../data/BillieEilish.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file, index_col=0)

    # Basic cleaning
    df = df.dropna(subset=["Lyric"])
    df = df.drop_duplicates(subset=["Lyric"])

    # Randomly pick 500 songs (or all if fewer than 500)
    sample_size = min(500, len(df))
    sampled_df = df.sample(
        n=sample_size,
        random_state=42,
        replace=False
    )

    dfs.append(sampled_df)

# Combine all artists
mixed_df = pd.concat(dfs, ignore_index=True)

# Shuffle the combined dataset
mixed_df = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
mixed_df["Lyric"] = (
    mixed_df["Lyric"]
    .astype(str)
    .str.replace("\u2028", " <NEWLINE> ", regex=False)
    .str.replace("\u2029", " <NEWLINE> ", regex=False)
)

mixed_df.to_csv("../data/mixed_lyrics.csv", index=False)