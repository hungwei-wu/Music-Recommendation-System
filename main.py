from utilities import file_io
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
import time
from preprocessing.preprocessor import Preprocessor
from LyricsProcessor import LyricsProcessor

if __name__ == "__main__":
    #chunks = file_io.read_lastfm_user_art_file("data/userid-timestamp-artid-artname-traid-traname.tsv")
    chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")

    # read songs
    vectorizer = FeatureHasher()
    pre = Preprocessor(chunks, vectorizer)
    songs = pre.read_songs(20)
    print(songs)

    # reset file reader
    #chunks = file_io.read_lastfm_user_art_file("data/tmp.tsv")
    #pre.reset_file_reader(chunks)

    # read user song mapping
    pre.read_user_songs(1000)
    # convert to user-song matrix
    X = pre.get_user_song_matrix()
    print (X.shape)

    start_time = time.time()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, list(range(X.shape[0])))
    print(clf.predict(pre.user_song_dict["user_000001"]))

    print("training and predict using {0}".format(time.time() - start_time))

    
    #song_content = [ (artist,song) for (artist,song) in zip(list(songs['artname']), list(songs['traname']))]
    # temporary song_list
    song_content = [('Underworld', 'Boy, Boy, Boy'),
                    ('Underworld', 'Crocodile'),
                    ('Led Zeppelin','Stairway to heaven'),
                    ('Imagine Dragons','Thunder'),
                    ('Sam Smith', 'Too Good At Goodbyes'),
                    ('Ed Sheeran','Perfect'),
                    ('Demi Lovato','Sorry Not Sorry'), 
                    ('Pink','What About Us')
                    ]
    l_pre = LyricsProcessor(song_content)
    l_pre.tfidf_transform()
    l_pre.word2vec()
    test1=l_pre.get_w2v_from_songname('Boy, Boy, Boy')
    test2=l_pre.get_w2v_from_songname('Crocodile')
    l_pre.write_song_word2vec()
    