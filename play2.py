from utilities import file_io
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor
from sklearn.metrics.pairwise import pairwise_distances
from model import recommendation
import numpy as np


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    # #chunks = file_io.read_lastfm_user_art_file("data/userid-timestamp-artid-artname-traid-traname.tsv")
    # chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    #
    # # read songs
    #
    #
    # #songs = pre.read_songs(10)
    # #print(songs)
    vectorizer = FeatureHasher(n_features=10, non_negative=True)
    # reset file reader
    #chunks = file_io.read_lastfm_user_art_file("data/test_very_short.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    pre = Preprocessor(chunks, vectorizer)
    pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(500)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()
    #print(X.todense())
    dist = pairwise_distances(X)
    pred = recommendation.predict(X, dist)
    #print(pred)

    print(recommendation.predict_by_factorize(X))
    recommendation.recommend_all(X, pred)
    # start_time = time.time()
    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf.fit(X, list(range(X.shape[0])))
    # print(clf.predict(pre.user_song_dict["user_000001"]))
    #
    # print("training and predict using {0}".format(time.time() - start_time))