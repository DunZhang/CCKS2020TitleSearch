import torch
import json
import re
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import numpy as np
from YWVecUtil import find_topk_by_vecs, VectorDataBase
from transformers import BertTokenizer, BertModel, BertConfig
import os
from typing import List
import pandas as pd
import logging
import pickle
from RetrievalModel import RetrievalModel
from ITopKSearcher import ITopKSearcher

logger = logging.getLogger(__name__)


class RFIJSTopKSearcher(ITopKSearcher):
    def __init__(self, model: RetrievalModel, ent_path, tokenizer, device, max_len=100, batch_size=100, topk=200):
        # other variables
        self.topk = topk
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        # get ent
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)
        # 实体信息变为字符串
        self.ent_id2str = None
        self.ent_to_string()
        self.all_ent_id = list(self.ents.keys())
        # 把数据都变成topken_id
        # 因为是拼接不需要[CLS]
        self.ent_id2token_id = {k: tokenizer.encode(v[0:self.max_len])[1:] for k, v in self.ent_id2str.items()}

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

    def gen_data_for_single_title(self, title_token_ids):
        """ 为一个title拼接知识库所有的实体 """
        # TODO 以后要做一个粗筛
        res, ent_ids = [], []
        for ent_id, ent_token_ids in self.ent_id2token_id.items():
            ent_ids.append(ent_id)
            res.append([title_token_ids, ent_token_ids])
        return res, ent_ids

    def get_data_loader(self, input_ids, attention_mask, token_type_ids):
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size, drop_last=False)
        return data_loader

    def get_topk_ids(self, title):
        title_token_ids = self.tokenizer.encode(title)
        pairs, ent_ids = self.gen_data_for_single_title(title_token_ids)
        input_ids, attention_mask, token_type_ids = self.pairs2pt(pairs)
        data_loader = self.get_data_loader(input_ids, attention_mask, token_type_ids)
        all_scores = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = [i.to(self.device) for i in batch_data]
                scores = self.model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                    token_type_ids=batch_data[2])
                scores = scores.to("cpu").data.numpy()[:, 0].tolist()
                all_scores.extend(scores)
        # 排序
        all_scores = list(zip(all_scores, ent_ids))
        all_scores.sort(key=lambda x: x[0], reverse=True)
        sorted_ent_ids = [i[1] for i in all_scores]
        return sorted_ent_ids[:self.topk]


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model"
    fc_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval_full_interaction_join_string\bert_tiny\best_model\fc_weight.bin"

    search_model = RetrievalModel(model_path, fc_path).to(device)

    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"

    topk_searcher = RFIJSTopKSearcher(search_model, ent_path, search_model.tokenizer, device, max_len=100, batch_size=1000,
                                      topk=200)
    topk_res = topk_searcher.get_topk_ids("班赛祛痘")
    print(topk_res)
    topk_res = topk_searcher.get_topk_ids("官网智满意一步定痛正品疼丹药房同款三盒起包邮")
    print(topk_res)
    topk_res = topk_searcher.get_topk_ids("女士短款钱包摧倩要时尚潮流款个性春季新款到货要恩爱夫点卡撒旦")
    print(topk_res)
    topk_res = topk_searcher.get_topk_ids("多维元素片29100片")
    print(topk_res)
