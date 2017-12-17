from utilities import file_io
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor

from model import recommendation
import numpy as np
from metrics import evaluation
from utilities import cluster_cf


if __name__ == "__main__":
    start_time = time.time()
    np.set_printoptions(threshold=np.nan)

    vectorizer = DictVectorizer()
    # reset file reader
    chunks = file_io.read_lastfm_user_art_file("data/halfid_20%_train.tsv")

    valid_songs = []    # don't filter with valid songs
    valid_songs = file_io.get_all_valid_songs('data/song_word2vec_whole_truncate_60000_new.csv')

    pre = Preprocessor(chunks, vectorizer, valid_songs)
    pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(3000000)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()
    print("non zeros: {0}".format(X.count_nonzero()))
    print("pre-processed in {0:2f} sec".format(time.time() - start_time))

    cluster_cf.cluster_usr(X, k=5)
    print("non zeros: {0}".format(X.count_nonzero()))
    pred = recommendation.predict_by_user(X)

    #pred = recommendation.predict_by_factorize(X)
    recommended = recommendation.recommend_all(X, pred)
    songs = pre.get_songs_by_indices(recommended, 3)
    print("user 1 top recommended songs: {0}".format(songs[0]))
    print("predict in {0:2f} sec".format(time.time() - start_time))

    evaluation.hit(songs, 3, 'data/halfid_20%_test.tsv', 'user_000001', 'uni')

    evaluation.RS_coverage_variation(songs)
    evaluation.diversity(songs, 'data/song_word2vec_whole_truncate_60000_new.csv')
    evaluation.hit(songs, 3, 'data/halfid_20%_train.tsv', 'user_000001', 'nov')
    songs = pre.get_songs_by_indices(recommended, 80)
    evaluation.catlalog_coverage(40000, songs)

    print("program finish in {0:2f} sec".format(time.time() - start_time))
