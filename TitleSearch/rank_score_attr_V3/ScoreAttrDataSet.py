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
from typing import List, Dict
import numpy as np
import torch
from transformers import BertTokenizer
import pickle
from ITopKSearcher import ITopKSearcher
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
import logging
from copy import deepcopy
from DataUtil import DataUtil
from AAttrFeatureExtractor import AAttrFeatureExtractor
from BasesFeatureExtractor import BasesFeatureExtractor
from CompanyFeatureExtractor import CompanyFeatureExtractor
from CountryFeatureExtractor import CountryFeatureExtractor
from SpecFeatureExtractor import SpecFeatureExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ScoreAttrDataSet():
    def __init__(self, file_path, ent_path, tokenizer: BertTokenizer, afes,
                 batch_size=32, max_sum_sens=None, topk_searcher: ITopKSearcher = None, use_manual_feature=False):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.afes = afes
        self.topk_searcher = topk_searcher
        self.use_manual_feature = use_manual_feature
        self.num_examples_per_record = 5  # 每个数据生成的负类个数
        self.ipt_names = ["input_ids", "attention_mask", "token_type_ids", "null_val_mask"]
        self.attr2len = {"name": 50, "company": 40, "bases": 70, "functions": 70, "place": 40}
        self.attr2token_id = {"name": 1, "functions": 2}  # unused 1 2 3 4 5
        self.attr_names = ["name", "company", "bases", "functions", "place", "spec"]
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
        if max_sum_sens:
            self.titles = random.sample(self.titles, max_sum_sens)
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
            t = {}
            for attr_name in ["name", "functions"]:
                attr_value = str(ent_info[attr_name])
                if DataUtil.is_null(attr_value):  # 实体不包含这个属性
                    # print("attr 为空", attr_name, ent_info)
                    t[attr_name] = []
                else:
                    t[attr_name] = self.tokenizer.encode(ent_info[attr_name])[1:]  # 因为是拼接不需要[CLS]
            self.ent_id2token_id[ent_id] = t

    def get_bert_model_input(self, titles_ids, pos_ent_ids, neg_ent_ids):
        # 首先生成两个需要bert的输入
        self.pos_ipts = {"name": {}, "company": {}, "bases": {}, "functions": {}, "place": {}, "spec": {}}
        for attr_name in ["name", "functions"]:
            t_titles_ids = deepcopy(titles_ids)
            for i in t_titles_ids:
                i[0] = self.attr2token_id[attr_name]
            res = self.pairs2pt(zip(t_titles_ids, [self.ent_id2token_id[ent_id][attr_name] for ent_id in pos_ent_ids]),
                                self.attr2len[attr_name])
            self.pos_ipts[attr_name] = {"input_ids": res[0], "attention_mask": res[1],
                                        "token_type_ids": res[2], "null_val_mask": res[3]}

        self.neg_ipts = {"name": {}, "company": {}, "bases": {}, "functions": {}, "place": {}, "spec": {}}
        for attr_name in ["name", "functions"]:
            t_titles_ids = deepcopy(titles_ids)
            for i in t_titles_ids:
                i[0] = self.attr2token_id[attr_name]
            res = self.pairs2pt(zip(t_titles_ids, [self.ent_id2token_id[ent_id][attr_name] for ent_id in neg_ent_ids]),
                                self.attr2len[attr_name])
            self.neg_ipts[attr_name] = {"input_ids": res[0], "attention_mask": res[1],
                                        "token_type_ids": res[2], "null_val_mask": res[3]}

    def get_manual_model_input(self, titles, pos_ent_ids, neg_ent_ids):
        for attr_name in ["company", "bases", "place", "spec"]:
            res = [self.afes[attr_name].extract_features(title, [self.ents[ent_id][attr_name]])[0] for title, ent_id in
                   zip(titles, pos_ent_ids)]
            res = torch.tensor(res)
            self.pos_ipts[attr_name] = {"features": res}

        for attr_name in ["company", "bases", "place", "spec"]:
            res = [self.afes[attr_name].extract_features(title, [self.ents[ent_id][attr_name]])[0] for title, ent_id in
                   zip(titles, neg_ent_ids)]
            res = torch.tensor(res)
            self.neg_ipts[attr_name] = {"features": res}

    def pairs2pt(self, pairs, max_len):
        """
        把pairs变成torch的格式
        pairs = [ (token_ids1,token_ids2), (token_ids3,token_ids4),...
        ]
        """
        input_ids, attention_mask, token_type_ids = [], [], []
        for ids1, ids2 in pairs:
            input_ids_t = (ids1 + ids2)
            input_ids_t = input_ids_t + [0] * (max_len - len(input_ids_t)) if len(
                input_ids_t) <= max_len else input_ids_t[0:max_len - 1] + [self.tokenizer.sep_token_id]

            token_type_ids_t = ([0] * len(ids1) + [1] * len(ids2))
            token_type_ids_t = token_type_ids_t + [0] * (max_len - len(token_type_ids_t)) if len(
                token_type_ids_t) <= max_len else token_type_ids_t[0:max_len]

            attention_mask_t = [1] * (len(ids1) + len(ids2))
            attention_mask_t = attention_mask_t + [0] * (max_len - len(attention_mask_t)) if len(
                attention_mask_t) <= max_len else attention_mask_t[0:max_len]

            ###
            input_ids.append(input_ids_t)
            attention_mask.append(attention_mask_t)
            token_type_ids.append(token_type_ids_t)
        token_type_ids = torch.LongTensor(token_type_ids)
        null_val_mask = torch.sign(torch.sum(token_type_ids, dim=1, keepdim=True))
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), token_type_ids, null_val_mask

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
        """
        为一个title和ent生成triples 即(anchor,pos,neg)
        :param title: 问题文本
        :param title_ids: 问题对应 bert token ids
        :param ent_id: title 对应的ent id
        :return:
        """
        res = []
        if self.topk_searcher:
            topk_ent_ids = self.topk_searcher.get_topk_ent_ids(title, topk=25)
            if ent_id in topk_ent_ids:
                topk_ent_ids.remove(ent_id)
            topk_ent_ids = random.sample(topk_ent_ids, self.num_examples_per_record)
            for neg_ent_id in topk_ent_ids:
                res.append([title, title_ids, ent_id, neg_ent_id])
        else:
            while len(res) < self.num_examples_per_record:
                neg_ent_id = random.choice(self.all_ent_id)
                if neg_ent_id != ent_id:
                    res.append([title, title_ids, ent_id, neg_ent_id])
        return res

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < 0:  # 首次迭代
            logger.info("first init")
            logger.info("获取所有的triples...")
            all_triples = self.get_all_triples()
            logger.info("完成获取所有的triples")
            titles = [i[0] for i in all_triples]
            titles_ids = [i[1] for i in all_triples]
            pos_ent_ids = [i[2] for i in all_triples]
            neg_ent_ids = [i[3] for i in all_triples]
            logger.info("把所有triples组合成bert的输入...")
            self.get_bert_model_input(titles_ids, pos_ent_ids, neg_ent_ids)
            if self.use_manual_feature:
                logger.info("把所有triples组合成手工特征打分器的输入...")
                self.get_manual_model_input(titles, pos_ent_ids, neg_ent_ids)
            logger.info("完成把所有triples组合")
            logger.info("==============================all input data shape================================")
            logger.info(self.pos_ipts["name"]["input_ids"].shape)
            self.start, self.end = 0, self.batch_size
        else:
            self.start = self.end
            self.end += self.batch_size
        # print("self.pos.shape[0]:", self.pos.shape[0])
        if self.end >= self.pos_ipts["name"]["input_ids"].shape[0]:
            print("restart iter")
            all_triples = self.get_all_triples()
            titles = [i[0] for i in all_triples]
            titles_ids = [i[1] for i in all_triples]
            pos_ent_ids = [i[2] for i in all_triples]
            neg_ent_ids = [i[3] for i in all_triples]
            self.get_bert_model_input(titles_ids, pos_ent_ids, neg_ent_ids)
            if self.use_manual_feature:
                logger.info("把所有triples组合成手工特征打分器的输入...")
                self.get_manual_model_input(titles, pos_ent_ids, neg_ent_ids)
            logger.info("完成把所有triples组合")
            self.start, self.end = 0, self.batch_size
            raise StopIteration
        batch_pos = {}
        for attr_name in ["name", "functions"]:
            batch_pos[attr_name] = {}
            for ipt_name in self.ipt_names:
                batch_pos[attr_name][ipt_name] = self.pos_ipts[attr_name][ipt_name][self.start:self.end]
        batch_neg = {}
        for attr_name in ["name", "functions"]:
            batch_neg[attr_name] = {}
            for ipt_name in self.ipt_names:
                batch_neg[attr_name][ipt_name] = self.neg_ipts[attr_name][ipt_name][self.start:self.end]
        ##########
        if self.use_manual_feature:
            for attr_name in ["company", "bases", "place", "spec"]:
                batch_pos[attr_name] = {}
                batch_pos[attr_name]["features"] = self.pos_ipts[attr_name]["features"][self.start:self.end]
            for attr_name in ["company", "bases", "place", "spec"]:
                batch_neg[attr_name] = {}
                batch_neg[attr_name]["features"] = self.neg_ipts[attr_name]["features"][self.start:self.end]
        return batch_pos, batch_neg


from os.path import join
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
if __name__ == "__main__":
    stop_words_path = join(PROJ_PATH, "data/external_data/stop_words.txt")
    areas_path = r"G:\Codes\CCKS2020TitleSearch\data\external_data\areas.txt"
    file_path = r"G:\Codes\CCKS2020TitleSearch\data\rank_score_attr\dev.txt"
    kb_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    model_path = join(PROJ_PATH, "PreTrainedModels/roberta_tiny_clue")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    topk_searcher = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\CCKS2020TitleSearch\data\rank_score_attr\dev_topk_ent_ids.bin")
    afes = {"company": CompanyFeatureExtractor(),
            "bases": BasesFeatureExtractor(stop_words_path),
            "place": CountryFeatureExtractor(areas_path),
            "spec": SpecFeatureExtractor()}
    data = ScoreAttrDataSet(file_path, kb_path, tokenizer, afes=afes, batch_size=3, topk_searcher=topk_searcher)
    for epoch in range(100):
        print(epoch)
        # print(data.ent_id2token_id["202637"])
        # print(tokenizer.decode(data.ent_id2token_id["202637"]["name"]))
        for step, batch_data in enumerate(data):
            # print(epoch, step)
            batch_pos, batch_neg = batch_data
            # for attr_name in data.attr2len.keys():
            for attr_name in ["name"]:
                print("=========================== {} ===================================".format(attr_name))
                print(tokenizer.decode(batch_pos[attr_name]["input_ids"][0].data.numpy().tolist()))
                print(batch_pos[attr_name]["input_ids"][0].data.numpy().tolist())
                print(batch_pos[attr_name]["attention_mask"][0])
                print(batch_pos[attr_name]["token_type_ids"][0])
                print(batch_pos[attr_name]["null_val_mask"][0])
                print("------------------------------------------------------------------------------------")
                print(tokenizer.decode(batch_neg[attr_name]["input_ids"][0].data.numpy().tolist()))
                print(batch_neg[attr_name]["input_ids"][0].data.numpy().tolist())
                print(batch_neg[attr_name]["attention_mask"][0])
                print(batch_neg[attr_name]["token_type_ids"][0])
                print(batch_neg[attr_name]["null_val_mask"][0])
            x = input()
            if "stop" in x:
                exit(0)
            # ["company", "bases", "place", "spec"]
            # for attr_name in ["spec"]:
            #     print("=========================== {} ===================================".format(attr_name))
            #     print(batch_pos[attr_name]["features"][0].data.numpy().tolist())
            #
            #     print("------------------------------------------------------------------------------------")
            #     print(batch_neg[attr_name]["features"][0].data.numpy().tolist())
            # x = input()
            # if "stop" in x:
            #     exit(0)
