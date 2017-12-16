import pandas as pd
import config


def read_lastfm_user_art_file(file_name):
    df = pd.read_csv(file_name, sep='\t',
                     names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname'],
                     chunksize=config.file_io_config["chunk_size"],
                     dtype=object)
    return df


def create_track_id2(df):
    """(experimenting) create unique id by (artname, traname) pair"""
    #df["trackid2"] = df.apply(lambda row: hash(row["artname"] + row["traname"]), axis=1).astype('int64')
    df["trackid2"] = df.apply(lambda row: row["artname"] + row["traname"], axis=1)
    return df


def get_all_valid_songs(word2vec_file):
    df = pd.read_csv(word2vec_file)
    #print(df["Unnamed: 0"])
    #df["song"] = df.apply(lambda row: row["Unnamed: 0"].decode('utf-8') if type(row["Unnamed: 0"]) is not str else row["Unnamed: 0"], axis=1)
    #df["song"] = df["Unnamed: 0"].applymap(lambda row: row.decode('utf-8') if type(row) is not str else row)
    byte_strings = df["Unnamed: 0"].values
    songs = [string[2:-1] for string in byte_strings]
    #df["song"] = songs
    return songs
