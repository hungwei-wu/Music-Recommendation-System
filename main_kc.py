from utilities import cluster_cf
from scipy.sparse import load_npz

if __name__ == "__main__":

    sp = load_npz('./data/sparse_matrix_raw.npz')
    print (sp)
    fill_matrix.fill_matrix(sp)


