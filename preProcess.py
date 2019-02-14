import os
from random import shuffle
import collections
import json

def load_data(file_path):
    cates = os.listdir(file_path)
    data = []
    label = []
    for idx, files in enumerate(cates):
        for file in os.listdir(file_path+'/'+files):
            with open(file_path+'/'+files+'/' + file, mode='r', encoding='utf-8') as f:
                sentences = f.readlines()
            for sen in sentences:
                sen = clean_data(sen.strip())
                # 将过短的句子抛弃
                if len(sen)<=5:
                    continue
                data.append(sen)
                label.append(idx)
    return data, label

def clean_data(sen):
    '''
    去掉标点符号和数字
    :param sen:
    :return:
    '''
    sen = sen.replace('0', '')
    sen = sen.replace('1', '')
    sen = sen.replace('2', '')
    sen = sen.replace('3', '')
    sen = sen.replace('4', '')
    sen = sen.replace('5', '')
    sen = sen.replace('6', '')
    sen = sen.replace('7', '')
    sen = sen.replace('8', '')
    sen = sen.replace('9', '')

    # 此处标点符号加入中文的，因为文本是中文的我
    sen = sen.replace('，', '')
    sen = sen.replace('。', '')
    sen = sen.replace('？', '')
    sen = sen.replace('！', '')
    sen = sen.replace('（', '')
    sen = sen.replace('）', '')
    sen = sen.replace('+', '')
    sen = sen.replace('-', '')
    sen = sen.replace('‘', '')
    sen = sen.replace('’', '')
    sen = sen.replace('“', '')
    sen = sen.replace('”', '')
    sen = sen.replace('《', '')
    sen = sen.replace('》', '')

    sen = sen.replace(',', '')
    sen = sen.replace('.', '')
    sen = sen.replace('?', '')
    sen = sen.replace('!', '')
    sen = sen.replace('(', '')
    sen = sen.replace(')', '')
    sen = sen.replace('', '')
    sen = sen.replace('', '')
    sen = sen.replace('"', '')
    sen = sen.replace('"', '')
    sen = sen.replace('[', '')
    sen = sen.replace(']', '')
    sen = sen.replace('<', '')
    sen = sen.replace('>', '')

    sen = sen.replace('\t', '')
    sen = sen.replace('\n', '')
    sen = sen.replace(' ', '')

    return sen

def split_data(data, label, train_per=0.8, val_per=0.1, test_per=0.1):
    '''
    切分数据集，并保存到data/文件下，train.txt, val.txt,test.txt,
    每一行格式：sentence \t label \n
    '''
    # print(data)
    # print(label)
    ziped = list(zip(data, label))
    # print(ziped)
    shuffle(ziped)
    data[:], label[:] = zip(*ziped)
    data_len = len(data)
    train_data = data[:int(data_len*train_per)]
    train_label = label[:int(data_len * train_per)]
    val_data = data[int(data_len*train_per):int(data_len*(train_per+val_per))]
    val_label = label[int(data_len*train_per):int(data_len * (train_per+ val_per))]
    test_data = data[int(data_len * (train_per+ val_per)):]
    test_label = label[int(data_len * (train_per+ val_per)):]

    # 写入文件
    with open('data/train.txt', mode='a', encoding='utf-8') as f:
        for i in range(len(train_data)):
            f.write(train_data[i]+'\t' + str(train_label[i] )+ '\n')
    f.close()
    with open('data/val.txt', mode='a', encoding='utf-8') as f:
        for i in range(len(val_data)):
            f.write(val_data[i]+'\t' + str(val_label[i] )+ '\n')
    f.close()
    with open('data/test.txt', mode='a', encoding='utf-8') as f:
        for i in range(len(test_data)):
            f.write(test_data[i]+'\t' + str(test_label[i]) + '\n')
    f.close()
    print('分割数据集完成')

def build_vocab(vocab_size=5000):
    '''
    使用train.txt建起词典后保存到data/下， vacob.txt
    :param data:
    :param vocab_size:
    :return:
    '''
    with open('data/train.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in (lines):
        data.append(line.split('\t')[0])
    word_count = collections.Counter()

    for sentence in data:
        for s in sentence:
            word_count[s] += 1
    word_list = word_count.most_common(vocab_size - 1)

    word_dict = {'pad':vocab_size-1} # 意义
    for idx, word in enumerate(word_list):
        word_dict[word[0]] = idx

    with open("data/vocab.json", "w") as f:
        json.dump(word_dict, f)
        print("写入字典文件完成...")



def pre_process(file_path, train_per=0.8, val_per=0.1, test_per=0.1, vocab_size=5000):
    data, label = load_data('D:\下载\THUCNews\THUCNews')
    # print(len(data))
    # print(len(label))
    # print(data)
    # print(label)
    split_data(data, label, train_per=0.8, val_per=0.1, test_per=0.1)
    build_vocab(vocab_size=5000)

    # pass

if __name__ == '__main__':
    data, label = load_data('D:\下载\THUCNews\THUCNews')
    print(len(data))
    print(len(label))
    print(data)
    print(label)
    split_data(data, label)
    build_vocab()
