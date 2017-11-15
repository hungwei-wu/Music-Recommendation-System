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
