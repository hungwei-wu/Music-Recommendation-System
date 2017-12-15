from utilities import cluster_cf
from scipy.sparse import load_npz
import time


if __name__ == "__main__":

    # Load matrix (sparse)
    # Which one?
    sp1 = load_npz('./utilities/sp/uid_tid/sparse_matrix.npz')
    sp2 = load_npz('./utilities/sp/uid_tname/sparse_matrix.npz')
    print ("uid_tid nonzero:{},uid_tname nonzero:{}",sp1.count_nonzero(),sp2.count_nonzero())

    # User-Cluster fill
    # start_time = time.time()
    # cluster_cf.cluster_usr(sp1,k=1,min_rate=0.1, add_rate=0.01)
    # print("training and predict using {0} sec".format(time.time() - start_time))
    print(sp1.count_nonzero())
    cluster_cf.cluster_item(sp1, k=2, min_rate=0.1, add_rate=0.01)

    # User-Cluster fill
