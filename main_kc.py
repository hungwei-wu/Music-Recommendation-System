from utilities import cluster_cf
from scipy.sparse import load_npz
import numpy as np
import time


if __name__ == "__main__":

    # Load matrix (sparse)
    # Which one?
    sp = load_npz('./utilities/sp/uid_tname/sparse_matrix.npz')
    sp_tra = np.load('./utilities/sp/uid_tname/sp_info_tra.npy')
    # sp = load_npz('./utilities/sp/uid_tname/sparse_matrix.npz')
    print ("uid_tid nonzero:{}".format(sp.count_nonzero()))

    # User-Cluster fill
    # start_time = time.time()
    # cluster_cf.cluster_usr(sp,k=5,min_rate=0.1, add_rate=0.01)
    # print("training and predict using {0} sec".format(time.time() - start_time))
    # print(sp.count_nonzero())

    cluster_cf.cluster_item(sp, sp_tra, k=5, min_rate=0.1, add_rate=0.01)
