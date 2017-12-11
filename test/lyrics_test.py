#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pyemd 
# pip install --upgrade gensim
import operator
import gensim
import lyricwikia
from sklearn.feature_extraction import DictVectorizer
from gensim.models import Word2Vec
import pandas as pd
"""
w2v = False
# Get batch of lyrics
song_list = [('Led Zeppelin','Stairway to heaven'),('Imagine Dragons','Thunder'),
             ('Sam Smith', 'Too Good At Goodbyes'),('Ed Sheeran','Perfect'),
             ('Demi Lovato','Sorry Not Sorry'), ('Pink','What About Us')]
tokenizer = RegexpTokenizer(r'[a-zA-Z]{1,}')
corpus = [ ]
stop_list = set(stopwords.words('english'))
name_lyrics = dict ()


for (artist,song) in song_list:
    lyrics = lyricwikia.get_lyrics(artist,song)
    lyrics_sp= lyrics.split("\n")
    doc = "";
    for line in lyrics_sp:
        if line:
            text_not_filtered = tokenizer.tokenize(line)
            filtered_text = [word.lower() for word in text_not_filtered \
                             if word.lower() not in stop_list]
            doc += " ".join(filtered_text) 
    name_lyrics[song] = len(corpus)        
    corpus.append(doc)


# Text feature extraction
vectorizer_count = CountVectorizer()
X = vectorizer_count.fit_transform(corpus)
# vectorizer_count.get_feature_names()

# Bi-gram: keep ordering info
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
X_bi = bigram_vectorizer.fit_transform(corpus)

# bigram_vectorizer.get_feature_names()

# Tf–idf term weighting
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X_bi)

# Combine of CountVectorizer & TfidfTransformer
tfidf_vectorizer = TfidfVectorizer()
TfIdf_vec = tfidf_vectorizer.fit_transform(corpus)

# Hashing trick(need not to "fit") + Tf-idf
hash_vectorizer = HashingVectorizer(stop_words='english')
Hash_vec = hash_vectorizer.transform(corpus)
transformer_hash = TfidfTransformer(smooth_idf=True)
tfidf_hash = transformer_hash.fit_transform(Hash_vec)

if w2v == True:
    # word2vec Trial
    # Load Google's pre-trained Word2Vec model.
    pretrain_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
    test_similarity=pretrain_model.similarity('woman', 'man')
    
    sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    sentence_president = 'The president greets the press in Chicago'.lower().split()
    # Remove their stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    sentence_obama = [w for w in sentence_obama if w not in stopwords]
    sentence_president = [w for w in sentence_president if w not in stopwords]
    # Compute WMD.
    # pip install pyemd
    distance = pretrain_model.wmdistance(sentence_obama, sentence_president)
    print ('distance = %.4f' % distance)
"""

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
print(model)
print(model['sentence'])

corpus_df = pd.DataFrame(columns=['Song_name','Lyrics'])

song_content = [('Underworld','Boy, Boy, Boy' ),('Underworld','Crocodile'),('Underworld','Boy, Boy, Boy' ),('Underworld','Crocodile')]
song_lyrics_dict = {}
for (artist,song) in song_content:
    #print (artist,song) 
    lyrics = lyricwikia.get_lyrics(artist,song)
    doc = " ".join(lyrics.split("\n"))
    corpus_df = corpus_df.append({'Song_name':song, 'Lyrics':doc},ignore_index=True)
    
"""   
hash_vectorizer = HashingVectorizer(stop_words='english',analyzer='word')    
Hash_vec = hash_vectorizer.transform(list(corpus_df['Lyrics'])

transformer_hash = TfidfTransformer(smooth_idf=True)
tfidf_hash = transformer_hash.fit_transform(Hash_vec)
"""
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
TfIdf_vec = tfidf_vectorizer.fit_transform(list(corpus_df['Lyrics']))
print(TfIdf_vec.todense())    


sorted_x = sorted(tfidf_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
Tf_idf =  TfIdf_vec.todense()
song_word2vec = pd.DataFrame(columns=['Lyrics'])
#pretrain_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
n_word_features = 5
for i in range(len(Tf_idf)) :
    row = np.squeeze(np.asarray(Tf_idf[i]))
    sort_index = row.argsort()[::-1][:n_word_features]
    sentence = [ pretrain_model[sorted_x[idx][0]] for idx in sort_index]
    tmp = pd.DataFrame({'Lyrics':[np.asarray(sentence)]},index = [list(corpus_df['Song_name'])[i]] )
    song_word2vec=pd.concat([song_word2vec,tmp])

# test
#song_word2vec.loc['Boy, Boy, Boy']