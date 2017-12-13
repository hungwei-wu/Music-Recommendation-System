import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


def predict(user_item, similarity):
    user_item = user_item.todense()
    mean_x = np.mean(user_item, axis=1)
    mean_diff = user_item - mean_x
    pred = mean_x + similarity.dot(mean_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred


def predict_by_factorize(user_item):
    # svd = TruncatedSVD(n_components=3)
    #
    # svd.fit(user_item)
    # print(svd.singular_values_)
    user_item_mean = csr_matrix.mean(user_item, axis=1)
    user_item_normalized = user_item - user_item_mean
    U, sigma, V = svds(user_item_normalized, k=2)
    sigma = np.diag(sigma)
    pred = U.dot(sigma).dot(V) + user_item_mean.reshape(-1, 1)
    return pred


def recommend_all(user_item, pred):
    user_item = user_item.todense()
    unseen = user_item == 0
    
    print(np.inner(user_item, unseen))

