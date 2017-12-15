from utilities import file_io
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor

from model import recommendation
import numpy as np


if __name__ == "__main__":
    start_time = time.time()
    np.set_printoptions(threshold=np.nan)

    vectorizer = DictVectorizer()
    # reset file reader
    #chunks = file_io.read_lastfm_user_art_file("data/test_very_short.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    pre = Preprocessor(chunks, vectorizer)
    pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(300)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()
    #print(X.todense())

    pred = recommendation.predict(X)

    #pred = recommendation.predict_by_factorize(X)
    recommended = recommendation.recommend_all(X, pred)
    songs = pre.get_songs_by_indices(recommended, 1)
    print(songs)
    print("program finish in {0:2f} sec".format(time.time() - start_time))
