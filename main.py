import pandas as pd
import numpy as np
from utilities import file_io
from collections import Counter
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp
import time


if __name__ == "__main__":
    #chunks = file_io.read_lastfm_user_art_file("data/userid-timestamp-artid-artname-traid-traname.tsv")
    #chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/tmp.tsv")
    vectorizer = FeatureHasher()
    user_df = {}

    start_time = time.time()
    for i, chunk in enumerate(chunks):
        #if i >= 100000:
        #    break
        #print(chunk)
        df1 = file_io.create_track_id2(chunk)
        users = df1.groupby('userid')['trackid2']
        for user_id, grouped_value in users:
            #print(dict(Counter(grouped_value)))
            transformed_vec = vectorizer.transform([dict(Counter(grouped_value))])
            user_df[user_id] = user_df.get(user_id, 0) + transformed_vec
    print("parsing file using {0}".format(time.time() - start_time))
    #print(user_df)
    X = sp.vstack(user_df.values())
    #print(X)

    start_time = time.time()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, list(range(X.shape[0])))
    print(clf.predict(user_df["user_000003"]))
    #sp.bsr_matrix([[87516, 107016, 259247]], shape=(1, 1048576))
    #X = zeros()
    print("training and predict using {0}".format(time.time() - start_time))



    # df1.to_csv('test.csv', encoding='utf-8')
    # print(df1)
    # df2 = chunks.read(5)
    # print(df2)
    #
    # # demonstration of combining data frames
    # df_combined = pd.concat((df1, df2))
    #
    # unique_artist_songs = [tuple(row.values) for _, row in df_combined.loc[:, ["artname", "traname"]].iterrows()]
    # print(unique_artist_songs)

