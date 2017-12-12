import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# Dummy input
N_usr = 100
N_item = 100
k = 2

# Init with 0/1 (Listening counts)
rate_matrix = np.random.choice([0, 1], size=(N_usr, N_item), p=[1. / 3, 2. / 3]).astype(np.float16)

# User clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(rate_matrix)
user_labels = kmeans.labels_
user_cluster_c = kmeans.cluster_centers_

def get_zero_loc(usr_rate,N_item):
    if sum(usr_rate) / N_item > 0.5:
        return None
    index = np.random.choice(np.where(usr_rate == 0)[0], N_item * 0.1, replace=False)
    return (index,)

print(rate_matrix)

# Fill out the rate_matrix if cap < threshold
for i, i_cls in enumerate(user_labels):
    user_rate = rate_matrix[i, :]
    revise_loc = get_zero_loc(user_rate,N_item)
    if revise_loc:
        user_rate[revise_loc] = user_cluster_c[i_cls, revise_loc]

print(np.count_nonzero(rate_matrix))
