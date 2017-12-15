import numpy as np
from collections import Counter
import sys
import csv
csv.field_size_limit(sys.maxint)


def catlalog_coverage(total_music_size, recommend_music): # catalog coverage, defined as length of recommend music / total number of music
    return len(recommend_music) / float(len(total_music_size))


def RS_coverage_variation(recommend_music):  # To all user, repeat song times.  Defined as repeat times/total music length
    total_size = len(recommend_music)
    count = Counter(recommend_music)
    print len(count)
    return len(count) / float(total_size)


def hit(recommend_music,num_in_user,file,first_user,mode): # mode = 'uni' return unitary validation (use test set file), mode = 'nov' return novelty use tran + test set
    user_list = []
    hit_rate = []
    user_index = 0
    hit_count = 0.
    nov_rate = []
    cur_user = first_user
    listen_numbers = []
    with open(file, 'rb') as tsvin:
        Input = csv.reader(tsvin, delimiter='\t')
        for row in Input:
            if (cur_user != row[0]):
                cur_user = row[0]
                for i in range(0,num_in_user):
                    listen_count = 0
                    for song in user_list:
                        if recommend_music[user_index][i] == song:
                            if(listen_count == 0):
                                hit_count = hit_count + 1
                            else:
                                listen_count = listen_count+1
                    listen_numbers.append(listen_count)
                hit_rate.append(hit_count/len(recommend_music[user_index]))
                nov_rate.append( (len(user_list)-hit_count)/len(user_list) )
                hit_count = 0.
                user_index = user_index + 1
                user_list = []
                user_list.append(row[5])
            else:
                user_list.append(row[5])
    print(hit_rate)
    print(nov_rate)
    print(listen_numbers)
    hit_prob = sum(hit_rate) / len(hit_rate)
    if(mode == 'uni'):
        return hit_prob,listen_numbers
    elif(mode == 'nov'):
        return sum(nov_rate)/len(nov_rate),listen_numbers

