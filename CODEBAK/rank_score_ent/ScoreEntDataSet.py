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
import pickle
from ITopKSearcher import ITopKSearcher
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ScoreEntDataSet():
    def __init__(self, file_path, ent_path, tokenizer: BertTokenizer, batch_size=32,
                 topk_searcher: ITopKSearcher = None):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.topk_searcher = topk_searcher
        self.num_examples_per_record = 6  # 每个数据生成的负类个数
        self.ipt_names = ["input_ids", "attention_mask", "token_type_ids"]
        self.max_len = 256
        self.attr2len = {"name": 50, "company": 40, "bases": 70, "functions": 70, "place": 40}
        self.attr_names = ["name", "company", "bases", "functions", "place"]
        self.pos_ipts, self.neg_ipts = None, None
        self.idx = None
        self.start, self.end = -1, -1
        self.ent_id2token_id = {}
        # 读取实体
        logger.info("读取实体库...")
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)
        logger.info("完成读取实体库")
        self.all_ent_id = list(self.ents.keys())
        # 训练数据
        logger.info("读取训练数据...")
        self.titles = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                self.titles.append(ss)
        logger.info("完成读取数据")
        # 把title和训练数据都变成topken_id
        logger.info("训练数据变为token id...")
        for i in self.titles:
            i.append(tokenizer.encode(i[0]))
        logger.info("完成训练数据变token id")
        logger.info("实体数据变为token id...")
        self.ents2token_id()
        logger.info("完成实体数据变token id")
        # 计算 每一轮会迭代的steps
        self.steps = len(self.titles) * self.num_examples_per_record // self.batch_size

    def ents2token_id(self):
        """
        ent 的每个字段做编码，空的为None
        :return:
        """
        for ent_id, ent_info in self.ents.items():
            t = []
            for attr_name in self.attr_names:
                attr_value = str(ent_info[attr_name])
                if len(attr_value) < 1 or attr_value.lower() in ["none", "nan", "null"]:
                    # print("attr 为空", attr_name, ent_info)
                    attr_token_id = [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id]
                else:
                    attr_token_id = self.tokenizer.encode(ent_info[attr_name], max_length=self.attr2len[attr_name],
                                                          pad_to_max_length=False)[1:]  # 因为是拼接不需要[CLS]
                t += attr_token_id
            self.ent_id2token_id[ent_id] = t

    def get_model_input(self, titles_ids, pos_ent_ids, neg_ent_ids):
        self.pos_ipts = {}
        res = self.pairs2pt(zip(titles_ids, [self.ent_id2token_id[ent_id] for ent_id in pos_ent_ids]))
        self.pos_ipts["input_ids"], self.pos_ipts["attention_mask"], self.pos_ipts["token_type_ids"] = res

        self.neg_ipts = {}
        res = self.pairs2pt(zip(titles_ids, [self.ent_id2token_id[ent_id] for ent_id in neg_ent_ids]))
        self.neg_ipts["input_ids"], self.neg_ipts["attention_mask"], self.neg_ipts["token_type_ids"] = res

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
        token_type_ids = torch.LongTensor(token_type_ids)
        # null_val_mask = torch.sign(torch.sum(token_type_ids, dim=1, keepdim=True))
        # return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), token_type_ids, null_val_mask
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), token_type_ids

    def get_all_triples(self):
        """ 获取所有的triples
        @:return [(title_ids, pos_ent_id, neg_ent_id), ...]
        """
        random.shuffle(self.titles)
        all_trples = []
        for i in self.titles:
            title, ent_id, title_ids = i
            all_trples.extend(self.gen_triples(title, title_ids, ent_id))
        random.shuffle(all_trples)
        return all_trples

    def gen_triples(self, title, title_ids, ent_id):
        """ 为一个title和ent生成triples 即(anchor,pos,neg) """
        res = []
        if self.topk_searcher:
            topk_ent_ids = self.topk_searcher.get_topk_ent_ids(title, topk=50)
            if ent_id in topk_ent_ids:
                topk_ent_ids.remove(ent_id)
            topk_ent_ids = random.sample(topk_ent_ids, self.num_examples_per_record)
            for neg_ent_id in topk_ent_ids:
                res.append([title_ids, ent_id, neg_ent_id])
        else:
            while len(res) < self.num_examples_per_record:
                neg_ent_id = random.choice(self.all_ent_id)
                if neg_ent_id != ent_id:
                    res.append([title_ids, ent_id, neg_ent_id])
        return res

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < 0:  # 首次迭代
            logger.info("first init")
            logger.info("获取所有的triples...")
            all_triples = self.get_all_triples()
            logger.info("完成获取所有的triples")
            titles_ids = [i[0] for i in all_triples]
            pos_ent_ids = [i[1] for i in all_triples]
            neg_ent_ids = [i[2] for i in all_triples]
            logger.info("把所有triples组合成bert的输入...")
            self.get_model_input(titles_ids, pos_ent_ids, neg_ent_ids)
            logger.info("完成把所有triples组合成bert的输入")
            self.start, self.end = 0, self.batch_size
        else:
            self.start = self.end
            self.end += self.batch_size
        # print("self.pos.shape[0]:", self.pos.shape[0])
        if self.end >= self.pos_ipts["input_ids"].shape[0]:
            print("restart iter")
            all_triples = self.get_all_triples()
            titles_ids = [i[0] for i in all_triples]
            pos_ent_ids = [i[1] for i in all_triples]
            neg_ent_ids = [i[2] for i in all_triples]
            self.get_model_input(titles_ids, pos_ent_ids, neg_ent_ids)
            self.start, self.end = 0, self.batch_size
            raise StopIteration
        batch_pos = {}
        for ipt_name in self.ipt_names:
            batch_pos[ipt_name] = self.pos_ipts[ipt_name][self.start:self.end]
        batch_neg = {}
        for ipt_name in self.ipt_names:
            batch_neg[ipt_name] = self.neg_ipts[ipt_name][self.start:self.end]
        return batch_pos, batch_neg


if __name__ == "__main__":
    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    model_path = r"G:\Data\roberta_tiny_clue"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    topk_searcher = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev_topk_ent_ids.bin")
    # file_path, ent_path, tokenizer: BertTokenizer, batch_size = 32,
    # topk_searcher: ITopKSearcher = None
    data = ScoreEntDataSet(file_path, kb_path, tokenizer, batch_size=3, topk_searcher=topk_searcher)
    for epoch in range(100):
        print(epoch)
        # print(data.ent_id2token_id["202637"])
        # print(tokenizer.decode(data.ent_id2token_id["202637"]["name"]))
        for step, batch_data in enumerate(data):
            print("=======================================================================================")
            batch_pos, batch_neg = batch_data
            print(tokenizer.decode(batch_pos["input_ids"][0].data.numpy().tolist()))
            print(batch_pos["input_ids"][0].data.numpy().tolist())
            print(batch_pos["attention_mask"][0])
            print(batch_pos["token_type_ids"][0])
            print("---------------------------------------------------------------------------------------")
            print(tokenizer.decode(batch_neg["input_ids"][0].data.numpy().tolist()))
            print(batch_neg["input_ids"][0].data.numpy().tolist())
            print(batch_neg["attention_mask"][0])
            print(batch_neg["token_type_ids"][0])
            x = input()
            if "stop" in x:
                exit(0)
