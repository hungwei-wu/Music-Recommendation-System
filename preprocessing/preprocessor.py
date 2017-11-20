from collections import Counter
import scipy.sparse as sp


class Preprocessor(object):
    def __init__(self, chunks, vectorizer):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.user_song_dict = {}

    def read_songs(self, n_record):
        df = self.chunks.read(n_record)
        return df[['artname','traname']]

    def read_user_songs(self, n_records):
        df = self.chunks.read(n_records)

        df = self._create_track_id2(df)
        users = df.groupby('userid')['trackid2']
        for user_id, grouped_value in users:
            transformed_vec = self.vectorizer.transform([dict(Counter(grouped_value))])
            self.user_song_dict[user_id] = self.user_song_dict.get(user_id, 0) + transformed_vec

    def get_user_song_matrix(self):
        return sp.vstack(self.user_song_dict.values())

    def _create_track_id2(self, df):
        """(experimenting) create unique id by (artname, traname) pair"""
        #df["trackid2"] = df.apply(lambda row: hash(row["artname"] + row["traname"]), axis=1).astype('int64')
        df["trackid2"] = df.apply(lambda row: row["artname"] + row["traname"], axis=1)
        return df

    def reset_file_reader(self, chunks):
        #chunks.close()
        self.chunks = chunks



