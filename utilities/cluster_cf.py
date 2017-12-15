import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import time



def load_matrix():
    # Produce user-item matrix, fill with counts
    pass

def cluster_usr(rate_mat, k=1, min_rate=0.1, add_rate=0.01):
    # Clustering using of users
    # Fill matrix with the centers

    N_user, N_item = rate_mat.shape
    print ("Matrix: {}x{}" .format(N_user, N_item))
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(rate_mat)
    user_labels, user_cluster_c = kmeans.labels_, kmeans.cluster_centers_
    print("k-means (k={}) took {} sec".format(k, time.time() - start_time))

    def get_zero_loc(usr_rate):
        rate = np.sum(usr_rate,axis=1)[0,0] / N_item
        # print ("fill rate:",rate)
        if rate > min_rate:
            return [] # full enough
        # index = np.random.choice(np.where(usr_rate[0,:] == 0)[0], int(N_item * 0.1), replace=False)

        idx = np.random.choice(N_item, int(N_item * add_rate), replace=False)
        vec = np.zeros((1, N_item),dtype=bool)  # sp no vector
        vec[0,idx] = 1

        sp_vec = usr_rate.toarray().astype(bool)
        loc_vec = np.logical_xor(np.logical_or(sp_vec, vec), sp_vec) # avoid filling nonzero loc
        return np.where(loc_vec != 0)
        # return idx

    start_time = time.time()
    for i, i_cls in enumerate(user_labels):
        user_rate = rate_mat[i] # (1,N_item)
        revise_loc = get_zero_loc(user_rate) # Fill out the rate_mat if cap < threshold
        if len(revise_loc) != 0:
            rate_mat[i,revise_loc] = user_cluster_c[i_cls, revise_loc]
        # for loc in revise_loc:
        #     if rate_mat[i, loc] == 0:
        #         rate_mat[i, loc] = user_cluster_c[i_cls, loc]
    print("filling took {} sec".format(time.time() - start_time))

def cluster_item(rate_mat, k=1, min_rate=0.1, add_rate=0.01):
    # Retrieve lyrics
    # Vector sets, cluster

    N_user, N_item = rate_mat.shape
    print ("Matrix: {}x{}" .format(N_user, N_item))
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(rate_mat.T)
    user_labels, user_cluster_c = kmeans.labels_, kmeans.cluster_centers_
    print("k-means (k={}) took {} sec".format(k, time.time() - start_time))


    pass


if __name__ == "__main__":

    # Dummy input: Init with 0/1 (Listening counts)
    N_usr = 100
    N_item = 100

    rate_mat = np.random.choice([0, 1], size=(N_usr, N_item), p=[1. / 3, 2. / 3]).astype(np.float16)
    print ("origin: {} / {}".format(np.count_nonzero(rate_mat), N_usr * N_item))

    k = 2  # kmeans
    fill_matrix(rate_mat, k)
    print ("fill: {} / {}".format(np.count_nonzero(rate_mat), N_usr * N_item))