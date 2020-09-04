""" 三元组数据集 """
from FeatureExtractor import MedicalFeatutreExtractor
import time
import json
from collections import defaultdict
import random
from typing import List
import numpy as np
import torch


class TripletDataSet():
    def __init__(self, file_path, kb_path, batch_size=32):
        self.batch_size = batch_size
        self.num_examples_per_record = 10  # 每个数据生成的负类个数
        # kb
        with open(kb_path, "r", encoding="utf8") as fr:
            self.kb = json.load(fr)
        self.all_ent_id = list(self.kb.keys())
        # label data
        self.data = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                self.data.append(ss)
        self.data = self.data
        self.pos, self.neg = None, None
        self.idx = None
        self.start, self.end = -1, -1
        # get sub_id2text_id
        # self.subj_id2text_id = defaultdict(set)
        # for i in self.data:
        #     text_id, text, label_id = i
        #     self.subj_id2text_id[label_id].add(label_id)
        # self.subj_id2text_id = dict(self.subj_id2text_id)

    def get_model_input(self, all_triples: List):
        """ 获取模型的输出 """
        pos, neg = [], []
        for triple in all_triples:
            pos_fe = MedicalFeatutreExtractor.get_feature(triple[0], self.kb[triple[1]])
            neg_fe = MedicalFeatutreExtractor.get_feature(triple[0], self.kb[triple[2]])
            pos.append(pos_fe)
            neg.append(neg_fe)
        self.pos, self.neg = torch.FloatTensor(pos), torch.FloatTensor(neg)

    def get_all_triples(self):
        """ 获取所有的triples """
        all_trples = []
        for i in self.data:
            text_id, text, ent_id = i
            all_trples.extend(self.gen_triples(text, ent_id))
        random.shuffle(all_trples)
        return all_trples

    def gen_triples(self, text, ent_id):
        """ 为一组数据生成triples """
        res = []
        while len(res) < self.num_examples_per_record:
            neg_ent_id = random.choice(self.all_ent_id)
            if neg_ent_id != ent_id:
                res.append([text, ent_id, neg_ent_id])
        return res

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < 0:  # 首次迭代
            print("first init")
            all_triples = self.get_all_triples()
            self.get_model_input(all_triples)
            self.start, self.end = 0, self.batch_size
        else:
            self.start = self.end
            self.end += self.batch_size
        # print("self.pos.shape[0]:", self.pos.shape[0])
        if self.end >= self.pos.shape[0]:
            print("restart iter")
            all_triples = self.get_all_triples()
            self.get_model_input(all_triples)
            self.start, self.end = 0, self.batch_size
            raise StopIteration
        return self.pos[self.start:self.end], self.neg[self.start:self.end]


if __name__ == "__main__":
    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    data = TripletDataSet(file_path, kb_path, batch_size=32)
    for epoch in range(1):
        for step, batch_data in enumerate(data):
            pos, neg = batch_data
            pos, neg = pos.to("cpu").data.numpy(), neg.to("cpu").data.numpy()
            print(epoch, step)
