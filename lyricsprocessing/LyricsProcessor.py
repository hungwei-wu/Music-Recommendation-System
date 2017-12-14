#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:09 2017

@author: michellelin
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# pip install --upgrade gensim
import operator
import gensim
import lyricwikia
import pandas as pd
import sys
from googletrans import Translator

class LyricsProcessor(object):
    def __init__(self, songs_info):
        self.songs_info = songs_info
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.TfIdf_vec = []
        self.song_word2vec = pd.DataFrame()
        self.corpus_df = pd.DataFrame(columns=['Song_name']) 
        self.not_found = {}
        self.write_to_csv = {}
        
    def tfidf_transform(self):
        for (artist,song) in self.songs_info:
            #print (artist,song) 
            #repeat in the same batch
            #song=song.encode("utf-8")
            if song in self.write_to_csv.keys():
                print (song+ " already exists")
                continue
            try:
                lyrics = lyricwikia.get_lyrics(artist,song)
                doc = " ".join(lyrics.split("\n"))
                self.corpus_df = self.corpus_df.append({'Song_name':song.encode("utf-8"), 'Lyrics':doc}, ignore_index=True)
            except:  
                print (song+ " not found")
                self.not_found[song] = artist
                continue
            self.write_to_csv[song] = artist
        self.TfIdf_vec = self.tfidf_vectorizer.fit_transform(list(self.corpus_df['Lyrics']))
        
    def word2vec(self, n_word_features = 5, TfIdf_weight = False):
        # sort words by count
        sorted_words = sorted(self.tfidf_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
        Tf_idf =  self.TfIdf_vec.todense()
        pretrain_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
        for i in range(len(Tf_idf)) :
            # for each song, take index for top n words in Tf-Idf
            row = np.squeeze(np.asarray(Tf_idf[i]))
            sort_index = row.argsort()[::-1][:n_word_features]
            
            """ 
            handle problem of translation
            translator = Translator()
            translator.translate(sorted_words[idx][0])
            """
            try:
                if not TfIdf_weight:
                    sentence = [pretrain_model[sorted_words[idx][0]] for idx in sort_index]
                else:
                    sentence = [pretrain_model[sorted_words[idx][0]] * row[idx] for idx in sort_index]
                flat_sentence = [item for sublist in sentence for item in sublist]
                tmp = pd.DataFrame(np.reshape(np.asarray(flat_sentence),(1,len(flat_sentence))),index = [list(self.corpus_df['Song_name'])[i]] )
                self.song_word2vec=pd.concat([self.song_word2vec,tmp])
            except: 
                next  
            
    def get_w2v_from_songname(self,index):
        return self.song_word2vec.loc[index]
    
    def write_song_word2vec(self,out_path='data/song_word2vec_whole.csv'):
        if not os.path.isfile(out_path):
            print ("Create csv")
            self.song_word2vec.to_csv(out_path)
        else:
            print ("Append csv")
            with open(out_path, 'a') as f:
                self.song_word2vec.to_csv(f, header=False)
                
    def written_info_batch(self):
        return (self.not_found, self.write_to_csv ,len(self.write_to_csv.keys()))
