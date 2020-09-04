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
from RetrievalModel import RetrievalModel
from TopKSearcher import RFIJSTopKSearcher
import datetime
import pickle

logger = logging.getLogger(__name__)


class TopKAccEvaluator():
    def __init__(self, file_path, ent_path, tokenizer, device, max_len=100, batch_size=100,
                 topk_searcher: RFIJSTopKSearcher = None,
                 topks: List[int] = [1, 10, 20, 30, 50, 100, 200, 400, 600, 1000, 2000]):
        # other variables
        self.topk_searcher = topk_searcher
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        self.topks = topks
        # get ent
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)
        # 实体信息变为字符串
        self.ent_id2str = None
        self.ent_to_string()
        self.all_ent_id = list(self.ents.keys())
        # 测试数据
        self.titles = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                self.titles.append(ss)
        # self.titles = self.titles[0:15]
        # 把title和数据都变成topken_id
        for i in self.titles:
            i.append(tokenizer.encode(i[0]))
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

    def gen_data_for_single_title(self, title_token_ids, title=None):
        """ 为一个title拼接知识库所有的实体 """
        # TODO 以后要做一个粗筛
        res, ent_ids = [], []
        if self.topk_searcher:
            ent_ids = self.topk_searcher.get_topk_ent_ids(title)
            for ent_id in ent_ids:
                res.append([title_token_ids, self.ent_id2token_id[ent_id]])
        else:
            for ent_id, ent_token_ids in self.ent_id2token_id.items():
                ent_ids.append(ent_id)
                res.append([title_token_ids, ent_token_ids])
        return res, ent_ids

    def get_data_loader(self, input_ids, attention_mask, token_type_ids):
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size, drop_last=False)
        return data_loader

    def eval_acc(self, model, xls_path=None):
        num_corrects = [0] * len(self.topks)
        model.eval()
        xls_data = []
        c = 0
        for title, label_id, title_token_ids in self.titles:
            c += 1
            if c % 10 == 0:
                print("evaluating -- ", str(datetime.datetime.now()), c)
            t = [title, label_id]
            pairs, ent_ids = self.gen_data_for_single_title(title_token_ids, title)

            input_ids, attention_mask, token_type_ids = self.pairs2pt(pairs)

            data_loader = self.get_data_loader(input_ids, attention_mask, token_type_ids)
            all_scores = []
            with torch.no_grad():
                for batch_data in data_loader:
                    batch_data = [i.to(self.device) for i in batch_data]
                    scores = model(input_ids=batch_data[0], attention_mask=batch_data[1], token_type_ids=batch_data[2])
                    scores = scores.to("cpu").data.numpy()[:, 0].tolist()
                    all_scores.extend(scores)
            # 排序
            all_scores = list(zip(all_scores, ent_ids))
            all_scores.sort(key=lambda x: x[0], reverse=True)
            sorted_ent_ids = [i[1] for i in all_scores]
            # 计算topk acc
            t.append("|".join(sorted_ent_ids[0:20]))
            t.append(label_id in sorted_ent_ids[0:1000])
            xls_data.append(t)
            for idx, topk in enumerate(self.topks):
                num_corrects[idx] += 1 if label_id in sorted_ent_ids[0:topk] else 0
        # 输出结果
        for idx in range(len(self.topks)):
            print(self.topks[idx], num_corrects[idx] / len(self.titles))
        # 保存文件到本地
        if xls_path:
            pd.DataFrame(xls_data, columns=["Title", "Label", "Topk", "InTop1000"]).to_excel(xls_path,
                                                                                             index=False)
        model.train()
        return num_corrects[5] / len(self.titles)

    def eval_test(self, model, pred_path, xls_path=None):
        model.eval()
        xls_data = []
        write_data = []
        c = 0
        for title, title_token_ids in self.titles:
            c += 1
            if c % 10 == 0:
                print("evaluating -- ", str(datetime.datetime.now()), c)
            t = [title]
            pairs, ent_ids = self.gen_data_for_single_title(title_token_ids, title)

            input_ids, attention_mask, token_type_ids = self.pairs2pt(pairs)

            data_loader = self.get_data_loader(input_ids, attention_mask, token_type_ids)
            all_scores = []
            with torch.no_grad():
                for batch_data in data_loader:
                    batch_data = [i.to(self.device) for i in batch_data]
                    scores = model(input_ids=batch_data[0], attention_mask=batch_data[1], token_type_ids=batch_data[2])
                    scores = scores.to("cpu").data.numpy()[:, 0].tolist()
                    all_scores.extend(scores)
            # 排序
            all_scores = list(zip(all_scores, ent_ids))
            all_scores.sort(key=lambda x: x[0], reverse=True)
            sorted_ent_ids = [i[1] for i in all_scores]
            # 记录数据
            write_data.append(sorted_ent_ids[0] + "\n")
            t.append("|".join(sorted_ent_ids[0:20]))
            xls_data.append(t)
        # 保存文件到本地
        if xls_path:
            pd.DataFrame(xls_data, columns=["Title", "Topk"]).to_excel(xls_path, index=False)
        with open(pred_path, "w", encoding="utf8") as fw:
            fw.writelines(write_data)
        model.train()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalModel(r"G:\Data\roberta_tiny_clue").to(device)

    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval_interaction_manual_feature_data\dev.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"

    eva = TopKAccEvaluator(test_data_path, ent_path, model.tokenizer, device, 100)
    accs = eva.eval_acc(model, "test.xlsx")
    print(accs)
