#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:09 2017

@author: michellelin
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# pip install --upgrade gensim
import operator
import gensim
import lyricwikia
import pandas as pd

class LyricsProcessor(object):
    def __init__(self, songs_info):
        self.songs_info = songs_info
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.TfIdf_vec = []
        self.song_word2vec = pd.DataFrame(columns=['Lyrics'])
        self.corpus_df = pd.DataFrame(columns=['Song_name','Lyrics']) 
        
    def tfidf_transform(self):
        for (artist,song) in self.songs_info:
            #print (artist,song) 
            lyrics = lyricwikia.get_lyrics(artist,song)
            doc = " ".join(lyrics.split("\n"))
            self.corpus_df = self.corpus_df.append({'Song_name':song, 'Lyrics':doc},ignore_index=True)
        self.TfIdf_vec = self.tfidf_vectorizer.fit_transform(list(self.corpus_df['Lyrics']))
        
    def word2vec(self, n_word_features = 5):
        sorted_x = sorted(self.tfidf_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
        Tf_idf =  self.TfIdf_vec.todense()
        pretrain_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(Tf_idf)) :
            row = np.squeeze(np.asarray(Tf_idf[i]))
            sort_index = row.argsort()[::-1][:n_word_features]
            sentence = [ pretrain_model[sorted_x[idx][0]] for idx in sort_index]
            tmp = pd.DataFrame({'Lyrics':[np.asarray(sentence)]},index = [list(self.corpus_df['Song_name'])[i]] )
            self.song_word2vec=pd.concat([self.song_word2vec,tmp])
            
    def get_w2v_from_songname(self,index):
        return self.song_word2vec.loc[index]
    def write_song_word2vec(self,out_path='song_word2vec.csv'):
        self.song_word2vec.to_csv(out_path)

