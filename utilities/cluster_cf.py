import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import time
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


def get_lyrics_dict(sp_tra,df):
    #df = pd.read_csv("data/song_word2vec/song_word2vec_tfidfweight.csv")
    #tra2vec_df = df.rename(columns={ df.columns[0] : 'traname'}).set_index("traname")
    df = df.rename(columns={ df.columns[0] : 'traname'})
    df["traname"] = df["traname"].apply(lambda x: x[2:-1])
    tra2vec_df = df.set_index("traname")
    
    tra2vec_dict = {}
    for traname in tra2vec_df.index:
        vec = tra2vec_df.loc[traname].as_matrix()
        if vec.ndim != 1:
            print (traname,vec.shape[0])
            vec = vec[0]
        mat = np.array(vec,dtype=np.float32).reshape((5,300))
        tra2vec_dict[traname] = np.mean(mat,axis=0) # Use mean to stand song (1,300)
    #print ("song collected:",len(tra2vec_dict.items()))
    return tra2vec_dict

def sp_shrink(sp,sp_tra,word2vec_dict):
    reduce_tra_idx = [ i for i,x in enumerate(sp_tra) if x in word2vec_dict.keys() ]
    reduce_tra = sp_tra[reduce_tra_idx]
    reduce_mat = sp[:,reduce_tra_idx]
    #print ("song reduced:",reduce_mat.shape[1])
    return reduce_mat,reduce_tra

def user_encode(sp, sp_tra,word2vec_df):
    # Retrieve lyrics vec
    # Return user-vec matrix
    tra2vec_dict = get_lyrics_dict(sp_tra,word2vec_df)
    rate_mat,tra = sp_shrink(sp,sp_tra,tra2vec_dict)
    
    N_user, N_item = rate_mat.shape
    encode_mat = np.zeros((N_user,300))
    
    for i in range(N_user):
        weight_vec = np.zeros(300)
        weight_sum = 1 # need add back
        for loc in rate_mat[i].nonzero()[1]:
            score = rate_mat[i,loc]
            weight_vec += score * tra2vec_dict[ tra[loc] ]
            weight_sum += score
        encode_mat[i] = weight_vec / weight_sum
    return rate_mat,encode_mat

def user_kmeans(rate_mat, k=1):
    # Clustering using of users
    # Fill matrix with the centers

    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(rate_mat)
    usr_labels, usr_cluster_c = kmeans.labels_, kmeans.cluster_centers_
    print("---k-means (k={}) took {} sec---".format(k, time.time() - start_time))
    return usr_labels, usr_cluster_c

def fill_matrix(rate_mat,usr_labels,min_rate=0.5, add_rate=0.01):
    def get_zero_loc(usr_rate):
        rate = usr_rate.count_nonzero() / N_item
        #print ("fill rate:",rate)
        if rate > min_rate: # full enough
            return []
        idx = np.random.choice(N_item, int(np.ceil(N_item*add_rate)), replace=False)
        vec = np.zeros((1, N_item),dtype=bool)  # sp no vector
        vec[0,idx] = 1

        sp_vec = usr_rate.toarray().astype(bool)
        loc_vec = np.logical_xor(np.logical_or(sp_vec, vec), sp_vec) # avoid filling nonzero loc
        loc = np.where(loc_vec != 0)
        if len(loc[0]) != 0:
            return loc # (loc[x],loc[y])
        return []

    N_user, N_item = rate_mat.shape

    # Construct cluster_c using labels (sparse)
    cls_num, cnt_cls = np.unique(usr_labels, return_counts=True)
    n_cls = len(cls_num)
    usr_cluster_c = lil_matrix((n_cls,N_item))
    for i, i_cls in enumerate(usr_labels):
        usr_cluster_c[i_cls] += rate_mat[i]
    for i in cls_num:
        usr_cluster_c[i_cls] = usr_cluster_c[i_cls] / cnt_cls[i]

    # Fill matrix, use lil_matrix to change will be faster!
    rate_mat = rate_mat.tolil() 
    start_time = time.time()
    for i, i_cls in enumerate(usr_labels):
        user_rate = rate_mat[i] # (1,N_item)
        revise_loc = get_zero_loc(user_rate) # Fill out the rate_mat if cap < threshold
        #print (len(revise_loc))
        if len(revise_loc) != 0:
            rate_mat[i,revise_loc[1]] = usr_cluster_c[i_cls, revise_loc[1]]
    print("---Filling took {} sec---".format(time.time() - start_time))
    return rate_mat

if __name__ == "__main__":

    # Dummy input: Init with 0/1 (Listening counts)
    N_usr = 100
    N_item = 100

    rate_mat = np.random.choice([0, 1], size=(N_usr, N_item), p=[1. / 3, 2. / 3]).astype(np.float16)
    print ("origin: {} / {}".format(np.count_nonzero(rate_mat), N_usr * N_item))

    k = 2  # kmeans
    fill_matrix(rate_mat, k)
    print ("fill: {} / {}".format(np.count_nonzero(rate_mat), N_usr * N_item))
