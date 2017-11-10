import pandas as pd
from utilities import file_io


if __name__ == "__main__":
    chunks = file_io.read_lastfm_user_art_file("data/userid-timestamp-artid-artname-traid-traname.tsv")

    # demonstration of sequential reading
    df1 = chunks.read(5)
    print(df1)
    df2 = chunks.read(5)
    print(df2)

    # demonstration of combining data frames
    df_combined = pd.concat((df1, df2))

    unique_artist_songs = [tuple(row.values) for _, row in df_combined.loc[:, ["artname", "traname"]].iterrows()]
    print(unique_artist_songs)

