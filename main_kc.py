from scipy.sparse import load_npz
import numpy as np
import time
import pandas as pd
import sys


from utilities import cluster_cf
from model import recommendation
from metrics import evaluation

if __name__ == "__main__":
    
    if sys.version_info[0]==2:
        sys.exit("Use python3")
    # Load matrix (sparse)
    # Which one?
    sp = load_npz('./utilities/sp/train/uid_tname/sparse_matrix.npz')
    sp_tra = np.load('./utilities/sp/train/uid_tname/sp_info_tra.npy')
    # sp = load_npz('./utilities/sp/uid_tname/sparse_matrix.npz')
    print ("uid_tname fill rate: {}".format(sp.count_nonzero()/(sp.shape[0]*sp.shape[1])))
    
    word2vec_df = pd.read_csv("data/song_word2vec/song_word2vec_whole_truncate_60000_new.csv")
    
    reduce_mat,encode_mat,tra = cluster_cf.user_encode(sp, sp_tra, word2vec_df)
    print ("sp matrix size:",sp.shape)
    print ("encoded matrix size:",reduce_mat.shape)
    
    k_lst = [1,25,50,75,100]
    fr_lst = [.05,.25,.75,1]
    for k in k_lst: 
        for fr in fr_lst:
            print ("k:{},fr:{}".format(k,fr))
            usr_labels, _ = cluster_cf.user_kmeans(encode_mat,k)
            fill_mat = cluster_cf.fill_matrix(reduce_mat,usr_labels,fill_rate=fr)
            #print ("before filling counts:",reduce_mat.count_nonzero())
            #print ("after filling counts: ",fill_mat.count_nonzero())

            # Call recommendation
            #print (reduce_mat.shape,fill_mat.shape)
            recommend = cluster_cf.recommend_all(reduce_mat,fill_mat)
            songs = cluster_cf.get_songs_by_indices(recommend,tra)
            evaluation.hit(songs, 3, 'data/halfid_20%_test.tsv', 'user_000001', 'uni')
            print ("---")
