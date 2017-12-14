import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans



def load_matrix():
    # Produce user-item matrix, fill with counts
    pass


def fill_matrix(rate_mat, k=2):
    # Fill matrix using cluster centers of users
    N_item = rate_mat.shape[1]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(rate_mat)
    user_labels = kmeans.labels_
    user_cluster_c = kmeans.cluster_centers_

    def get_zero_loc(usr_rate, N_item):
        print (usr_rate.shape,np.sum(usr_rate,axis=1)[0,0],usr_rate.todense())
        ss = np.sum(usr_rate,axis=1)[0,0]

        if ss / N_item > 0.6:
            return []
        # index = np.random.choice(np.where(usr_rate[0,:] == 0)[0], int(N_item * 0.1), replace=False)
        N_item=1000
        idx = np.random.choice(N_item, int(N_item * 0.1), replace=False)
        return np.array(idx)

    for i, i_cls in enumerate(user_labels):
        user_rate = rate_mat[i, :]
        revise_loc = get_zero_loc(user_rate,N_item) # Fill out the rate_mat if cap < threshold
        print (revise_loc)
        if len(revise_loc) != 0:
            print (i_cls, revise_loc,user_cluster_c.shape)
            print (user_rate[revise_loc])
            # user_rate[revise_loc] = user_cluster_c[i_cls, revise_loc]

def item_cluster():
    # Retrieve lyrics' vector sets, cluster
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