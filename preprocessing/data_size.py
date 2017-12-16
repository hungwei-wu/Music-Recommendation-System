import sys
import csv
csv.field_size_limit(sys.maxint)
from itertools import repeat

def select_data (data_num,percentage):
    new_data_num = []
    for i in data_num:
        new_data_num.append(int(i*percentage))
    return new_data_num

def data_select (chunk,fold,user_select,file): #fold = can generate percentage of the data for test and for training
    user_log_num = []
    count = 1
    cur_user = 0
    index_count = 0
    index = 0
    with open(file,'rb') as tsvin:
        Input = csv.reader(tsvin, delimiter = '\t')
        for row in Input:
            if (cur_user != row[0]):
                cur_user = row[0]
                user_log_num.append(count)
                count = 1
            else:
                count = count+1
        user_log_num.append(count)
    user_log_num.pop(0)
    if (chunk != 5):
        write_test  = 0
        write_train = 1
    else:
        write_test = 1
        write_train = 0
    train_count = 0
    test_count  = 0
    changed = 0
    with open(file,'rb') as tsvin, open('data/train_data.tsv', 'wb') as tsvout1, open('data/test_data.tsv', 'wb') as tsvout2:
        Input = csv.reader(tsvin, delimiter = '\t')
        train = csv.writer(tsvout1, delimiter = '\t')
        test = csv.writer(tsvout2, delimiter = '\t')
        for row in Input:
            if (write_train):
                train.writerows([row])
                index_count = index_count+1
                train_count = train_count + 1
                if(train_count >= (user_log_num[index]-chunk*user_log_num[index]/fold) and changed == 0):
                    write_test = 1
                    write_train = 0
            elif (write_test):
                test.writerows([row])
                index_count = index_count+1
                test_count = test_count+1
                if(test_count == int(user_log_num[index]/fold)):
                    write_train = 1
                    write_test = 0
                    changed = 1
            if(index_count == user_log_num[index]):
                index_count = 0
                index = index + 1
                changed = 0
                train_count = 0
                test_count = 0
                write_train = 1
                write_test = 0
                if(index == user_select):
                    break


data_select(5,5,500,'data/userid-timestamp-artid-artname-traid-traname.tsv')

