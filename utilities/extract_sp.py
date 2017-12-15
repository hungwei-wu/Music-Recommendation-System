import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

# Use python3
file_name = "../data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
df = pd.read_csv(file_name, sep='\t',header=None, quoting=3,
                 names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname'],
                 dtype=object)

cnt_df = pd.crosstab(df.userid,df.traid)
del df
print ("release df")

np.save('sp_info_usr.npy', cnt_df.index ) # Save info
np.save('sp_info_tra.npy', cnt_df.columns ) # Save info
print ("save sp_info")

save_npz('sparse_matrix.npz',csr_matrix(cnt_df.values)) # Save sp
print ("save sparse_matrix.npz")
