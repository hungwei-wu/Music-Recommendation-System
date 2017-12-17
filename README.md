### Music Recommendation System
 
#### *Machine Learning Final Project*


#### Data

Prepare the 1000K dataset from Last.fm, and also prepare word2vec song list.
Word2vec song lists can be generated through LyricProcessor.py. 
#### Run

Run "python main_classic.py" to run traditional 
user-item collaborative algorithm. 

Run "python main_kc.py" to run proposed hybrid method.

#### Methods

- Traditional CF:

    Construct user-item matrix and compute similarity between users.
And use their similarity and de-meaned user vector as weight recommended vector.

- Proposed Hybrid Method:

    Compute distance between each lyrics. The distance are calculated by top-5 TF-IDF words,
    and transformed to vector by word2vec.
    Then we would use clustering on users, and fill out the missing values according to the 
    center of the cluster each user belongs to.



