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

import gensim
import lyricwikia

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
