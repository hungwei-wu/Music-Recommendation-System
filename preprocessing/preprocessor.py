from collections import Counter
import scipy.sparse as sp
from collections import defaultdict


class Preprocessor(object):
    def __init__(self, chunks, vectorizer):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.user_song_dict = defaultdict(Counter)

    def read_songs(self, n_record):
        df = self.chunks.read(n_record)
        return df[['artname', 'traname']]

    def read_user_songs(self, n_records):
        df = self.chunks.read(n_records)

        df = self._create_track_id2(df)
        users = df.groupby('userid')['trackid2']
        for user_id, grouped_value in users:
            self.user_song_dict[user_id].update(Counter(grouped_value))

    def get_user_song_matrix(self):
        # change here to waive effort to fit
        X = self.vectorizer.fit_transform(self.user_song_dict.values())
        return X

    def _create_track_id2(self, df):
        """(experimenting) create unique id by (artname, traname) pair"""
        #df["trackid2"] = df.apply(lambda row: hash(row["artname"] + row["traname"]), axis=1).astype('int64')
        df["trackid2"] = df.apply(lambda row: row["artname"] + row["traname"], axis=1)
        return df

    def reset_file_reader(self, chunks):
        #chunks.close()
        self.chunks = chunks





