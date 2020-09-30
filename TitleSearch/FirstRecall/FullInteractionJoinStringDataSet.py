"""

三元组数据集

Bert的输入
[CLS]你好啊[SEP]嗯呢[SEP]
101, 765, 1681, 1359, 102, 1426, 1254, 102
0, 0, 0, 0, 0, 1, 1, 1
1, 1, 1, 1, 1, 1, 1, 1
"""
import time
import json
from collections import defaultdict
import random
from typing import List
import numpy as np
import torch
from transformers import BertTokenizer


class FullInteractionJoinStringDataSet():
    def __init__(self, file_path, ent_path, tokenizer: BertTokenizer, batch_size=32, max_len=80):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_examples_per_record = 10  # 每个数据生成的负类个数
        # 读取实体
        with open(ent_path, "r", encoding="utf8") as fr:
            self.ents = json.load(fr)
        # 实体信息变为字符串
        self.ent_id2str = None
        self.ent_to_string()
        self.all_ent_id = list(self.ents.keys())
        # 训练数据
        self.titles = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                self.titles.append(ss)
        # 把title和训练数据都变成topken_id
        for i in self.titles:
            i.append(tokenizer.encode(i[0]))
        # 因为是拼接不需要[CLS]
        self.ent_id2token_id = {k: tokenizer.encode(v[0:self.max_len])[1:] for k, v in self.ent_id2str.items()}

        self.pos_input_ids, self.pos_attention_mask, self.pos_token_type_ids = None, None, None
        self.neg_input_ids, self.neg_attention_mask, self.neg_token_type_ids = None, None, None
        self.idx = None
        self.start, self.end = -1, -1
        # 计算 每一轮会迭代的steps
        self.steps = len(self.titles) * self.num_examples_per_record // self.batch_size

    def ent_to_string(self):
        """ 实体信息变为字符串 """
        self.ent_id2str = {}
        for subj_id, ent_info in self.ents.items():
            ent_str = ""
            place = ent_info["place"]
            if not (place == "None" or place == "nan" or len(place) < 1):
                ent_str += place
            name = ent_info["name"]
            if not (name == "None" or name == "nan" or len(name) < 1):
                ent_str += name
            company = ent_info["company"]
            if not (company == "None" or company == "nan" or len(company) < 1):
                ent_str += company
            functions = ent_info["functions"]
            if not (functions == "None" or functions == "nan" or len(functions) < 1):
                ent_str += functions
            bases = ent_info["bases"]
            if not (bases == "None" or bases == "nan" or len(bases) < 1):
                ent_str += bases
            self.ent_id2str[subj_id] = ent_str
        # 长度统计
        # lens = [len(i) for i in self.ent_id2str.values()]
        # print("实体信息字符串字符统计：\n")
        # print("max len", max(lens))
        # print("min len", min(lens))
        # print("mean len", sum(lens) / len(lens))
        # print("(25 50 75)分位数", np.percentile(lens, (25, 50, 75), interpolation='midpoint'))

    def pairs2pt(self, pairs):
        """
        把pairs变成torch的格式
        pairs = [ (token_ids1,token_ids2), (token_ids3,token_ids4),...
        ]
        """
        input_ids, attention_mask, token_type_ids = [], [], []
        for ids1, ids2 in pairs:
            input_ids_t = (ids1 + ids2)
            input_ids_t = input_ids_t + [0] * (self.max_len - len(input_ids_t)) if len(
                input_ids_t) <= self.max_len else input_ids_t[0:self.max_len - 1] + [self.tokenizer.sep_token_id]

            token_type_ids_t = ([0] * len(ids1) + [1] * len(ids2))
            token_type_ids_t = token_type_ids_t + [0] * (self.max_len - len(token_type_ids_t)) if len(
                token_type_ids_t) <= self.max_len else token_type_ids_t[0:self.max_len]

            attention_mask_t = [1] * (len(ids1) + len(ids2))
            attention_mask_t = attention_mask_t + [0] * (self.max_len - len(attention_mask_t)) if len(
                attention_mask_t) <= self.max_len else attention_mask_t[0:self.max_len]

            ###
            input_ids.append(input_ids_t)
            attention_mask.append(attention_mask_t)
            token_type_ids.append(token_type_ids_t)
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids)

    def get_model_input(self, all_triples: List):
        """ 获取模型的输出 """
        pos_pairs = [i[0:2] for i in all_triples]
        neg_pairs = [[i[0], i[2]] for i in all_triples]
        print(len(pos_pairs), len(neg_pairs))
        self.pos_input_ids, self.pos_attention_mask, self.pos_token_type_ids = self.pairs2pt(pos_pairs)
        self.neg_input_ids, self.neg_attention_mask, self.neg_token_type_ids = self.pairs2pt(neg_pairs)

    def get_all_triples(self):
        """ 获取所有的triples """
        all_trples = []
        for i in self.titles:
            text, ent_id, token_ids = i
            all_trples.extend(self.gen_triples(token_ids, ent_id))
        random.shuffle(all_trples)
        # print("**********************all triples examples**********************")
        # s1, s2, s3 = all_trples[33]
        # s1, s2, s3 = self.tokenizer.decode(s1), self.tokenizer.decode(s2), self.tokenizer.decode(s3)
        # print(s1, s2, s3)
        return all_trples

    def gen_triples(self, token_ids, ent_id):
        """ 为一组数据生成triples """
        res = []
        while len(res) < self.num_examples_per_record:
            neg_ent_id = random.choice(self.all_ent_id)
            if neg_ent_id != ent_id:
                res.append([token_ids, self.ent_id2token_id[ent_id], self.ent_id2token_id[neg_ent_id]])
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
        if self.end >= self.pos_input_ids.shape[0]:
            print("restart iter")
            all_triples = self.get_all_triples()
            self.get_model_input(all_triples)
            self.start, self.end = 0, self.batch_size
            raise StopIteration
        return self.pos_input_ids[self.start:self.end], self.pos_attention_mask[
                                                        self.start:self.end], self.pos_token_type_ids[
                                                                              self.start:self.end], \
               self.neg_input_ids[self.start:self.end], self.neg_attention_mask[
                                                        self.start:self.end], self.neg_token_type_ids[
                                                                              self.start:self.end]


if __name__ == "__main__":
    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    model_path = r"G:\Data\roberta_tiny_clue"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    data = FullInteractionJoinStringDataSet(file_path, kb_path, tokenizer, batch_size=32, max_len=100)
    for epoch in range(100):
        print(epoch)
        for step, batch_data in enumerate(data):
            # print(epoch, step)
            pass
