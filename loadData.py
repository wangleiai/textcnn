import os
import json
import random
from keras.preprocessing.sequence import  pad_sequences

def getVocab():
    with open("data/vocab.json", 'r') as load_f:
        word_dict = json.load(load_f)
    return word_dict

def word_to_id(sentences, word_dict):
    new = []
    for idx, sen in enumerate(sentences):
        x = []
        for i in  range(len(sen)):
            if sen[i] in word_dict.keys():
                x.append(word_dict[sen[i]])
        new.append(x)
    return new


def get_batch_data(file_path, batch_size, word_max_length):

    word_dict = getVocab()
    with open(file_path, mode='r', encoding='utf-8') as f:
        data = f.readlines()
    def batch():
        # random_num = random.sample(range(0, len(data) - 1), batch_size)
        while True:
            random_num = random.sample(range(0, len(data)-1), batch_size)
            x = []
            y = []
            for i in range(batch_size):
                x1, y1 = data[random_num[i]].split('\t')
                x.append(x1)
                y.append(int(y1.replace('\n', '')))

            x = word_to_id(x, word_dict)
            x = pad_sequences(x, maxlen=word_max_length, padding='post', value=4999) # value默认魏0，但我的pad在字典里是4999

            yield x, y
    return batch

def get_data_nums(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return len(f.readlines())

if __name__ == '__main__':
    # batch_data = get_batch_data('data/val.txt', batch_size=5, word_max_length=500)
    # for x, y in  batch_data():
    #     print(x, y)
    pass

