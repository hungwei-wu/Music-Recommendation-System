from utilities import file_io
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor


if __name__ == "__main__":
    #chunks = file_io.read_lastfm_user_art_file("data/userid-timestamp-artid-artname-traid-traname.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")

    # read songs
    vectorizer = FeatureHasher()
    pre = Preprocessor(chunks, vectorizer)
    songs = pre.read_songs(10)
    print(songs)

    # reset file reader
    chunks = file_io.read_lastfm_user_art_file("data/tmp.tsv")
    pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(1000)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()

    start_time = time.time()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, list(range(X.shape[0])))
    print(clf.predict(pre.user_song_dict["user_000001"]))

    print("training and predict using {0}".format(time.time() - start_time))

