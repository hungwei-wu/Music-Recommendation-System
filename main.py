from utilities import file_io
import time
from preprocessing.preprocessor import Preprocessor
from lyricsprocessing.LyricsProcessor import LyricsProcessor
from sklearn.feature_extraction import FeatureHasher
if __name__ == "__main__":
    chunks = file_io.read_lastfm_user_art_file("data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv")
    #chunks = file_io.read_lastfm_user_art_file("data/test_shorter.tsv")
    vectorizer = FeatureHasher()
    pre = Preprocessor(chunks, vectorizer)
    """
    # Check chunks size : tot_len = 19098862
    len_perchunk = 1000000
    tot_len = 0;
    while len_perchunk == 1000000:
        songs = pre.read_songs(1000000)
        len_perchunk =len(songs.index)
        tot_len += len_perchunk
    """
    # Q: if song repeat ??
    #songs = songs[15:21]
    batch_num = 100*20
    batch_size = 10000
    csv_song_list = []
    seen_list = {}
    for i in range(batch_num):
        print("================"+str(i)+" Batch =======================")
        # read songs
        songs = pre.read_songs(batch_size)
        print(songs)
        
        # global repeat
        song_content = [ (artist,song.split('(')[0]) for (artist,song) in zip(list(songs['artname']), list(songs['traname'])) if song not in seen_list.keys() ]
        for (artist,song) in song_content:
            seen_list[song]=''
        # temporary song_list

        l_pre = LyricsProcessor(song_content)
        l_pre.tfidf_transform()
        nf= l_pre.not_found
        l_pre.word2vec()
        #test1=l_pre.get_w2v_from_songname('Someday You Will Be Loved')
        l_pre.write_song_word2vec()
        csv_song_list.extend(list(l_pre.write_to_csv.keys()))
        #(song_not_found, song_written , written_num) = l_pre.written_info_batch()
        # check the last chunck
        if len(songs.index) < batch_size:
            print ("Total num :"+ str(i * batch_size + len(songs.index)) )
            print ("finish the last batch")
            break
    