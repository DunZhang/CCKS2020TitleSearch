# -*- coding: utf-8 -*-
"""
@Time : 2020/9/11 15:59 
@Author : WangBin
@File : data_aug.py 
@Software: PyCharm 
"""
import json
import pickle
import random

def convert_to_json():
    with open("./raw_data/book_ents.bin", "rb") as fr:
        ents = pickle.load(fr)

    print(len(ents))

    with open('./raw_data/book_ents.json', 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(ents, ensure_ascii=False, indent=2))


def book_info():
    with open("./raw_data/book_ents.bin", "rb") as fr:
        book_ents = pickle.load(fr)
    with open('./raw_data/book_label_data.txt', 'r', encoding='utf-8') as f:
        for ll in f:
            data_list = ll.strip().split('\t')
            if len(data_list) == 2:
                s_id = data_list[1]
                ent_dic = book_ents[s_id]
                print(data_list[0], ent_dic)
                print()


# book_info()
def get_book_train_data():
    with open("./raw_data/book_ents.bin", "rb") as fr:
        book_ents = pickle.load(fr)
    f = open('./raw_data/book_label_data_aug.txt', 'a', encoding='utf-8')
    for k, v in book_ents.items():
        f.write(book_ents[k]['name'] + '\t' + k + '\n')
    f.close()

# get_book_train_data()

def get_divide_train_dev_data():
    total_data = []
    f = open('./raw_data/book_label_data_aug.txt', 'r', encoding='utf-8')
    books = f.readlines()
    f.close()

    f = open('./raw_data/medical_label_data.txt', 'r', encoding='utf-8')
    medicals = f.readlines()
    f.close()

    for bb in books:
        data1 = bb.strip().split('\t')
        if len(data1) == 2:
            total_data.append(data1[0] + '\t' + 'book')
    num = len(total_data)
    print('book data number: {}'.format(num))
    for mm in medicals:
        data2 = mm.strip().split('\t')
        if len(data2) == 2:
            total_data.append(data2[0] + '\t' + 'medical')

    print('medical data number: {}'.format(len(total_data) - num))

    random.seed(42)
    random.shuffle(total_data)
    random.shuffle(total_data)
    dev = total_data[:int(len(total_data) / 10)]
    train = total_data[int(len(total_data) / 10):]

    fw = open('../data/data_all.txt', 'w', encoding='utf-8')
    for da in total_data:
        fw.write(da)
        fw.write('\n')
    fw.close()

    fw = open('../data/train.txt', 'w', encoding='utf-8')
    for da in train:
        fw.write(da)
        fw.write('\n')
    fw.close()

    fw = open('../data/dev.txt', 'w', encoding='utf-8')
    for da in dev:
        fw.write(da)
        fw.write('\n')
    fw.close()
    print('divide done!')

# book data number: 272660
# medical data number: 83146
# divide done!
# get_divide_train_dev_data()
