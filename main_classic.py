from utilities import file_io
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor

from model import recommendation
import numpy as np
from metrics import evaluation


if __name__ == "__main__":
    start_time = time.time()
    np.set_printoptions(threshold=np.nan)

    vectorizer = DictVectorizer()
    # reset file reader
    #chunks = file_io.read_lastfm_user_art_file("data/test_very_short.tsv")
    #chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/halfid_20%_train.tsv")

    valid_songs = file_io.get_all_valid_songs('data/song_word2vec_whole_truncate_50000_new.csv')
    #valid_songs = ['Womanizer']
    pre = Preprocessor(chunks, vectorizer, valid_songs)
    pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(3000000)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()

    pred = recommendation.predict_by_factorize(X)

    #pred = recommendation.predict_by_factorize(X)
    recommended = recommendation.recommend_all(X, pred)
    songs = pre.get_songs_by_indices(recommended, 3)
    print(songs)
    print("prediction in {0:2f} sec".format(time.time() - start_time))
    evaluation.hit(songs, 3, 'data/halfid_20%_test.tsv', 'user_000001', 'uni')
    print("program finish in {0:2f} sec".format(time.time() - start_time))
