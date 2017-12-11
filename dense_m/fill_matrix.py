import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# Dummy input
N_usr = 20
N_item = 100

# Init with 0/1
rate_matrix = np.random.choice([0, 1], size=(N_usr, N_item), p=[
                               1. / 3, 2. / 3]).astype(np.float32)

# User clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(rate_matrix)
labels = kmeans.labels_
cluster_c = kmeans.cluster_centers_


def get_revise_loc(usr_rate): return np.where(usr_rate == 0)


print(rate_matrix)

# Fill out the rate_matrix
for i, i_cls in enumerate(labels):
    user_rate = rate_matrix[i, :]
    revise_loc = get_revise_loc(user_rate)
    user_rate[revise_loc] = cluster_c[i_cls, revise_loc]

print(rate_matrix)

#  fill top-k center attr to per user (Random?)
#k = 100
#for i, per_user in enumerate(rate_matrix):

