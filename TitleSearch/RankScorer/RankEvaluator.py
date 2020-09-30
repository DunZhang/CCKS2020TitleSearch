from transformers import BertTokenizer
from ITopKSearcher import ITopKSearcher
from typing import Union, List
from DataUtil.DataUtil import DataUtil
import pickle
import pandas as pd
import logging
from os.path import join
import pickle
from collections import defaultdict
import re
from ITopKSearcher import ITopKSearcher
import random
from TopKSearcher import TopKSearcher
from transformers import BertTokenizer
from typing import Iterable, List, Tuple, Dict
import os
import torch
from FirstScorerModel import FirstScorerModel
import pylcs

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"


class RankEvaluator():
    def __init__(self, file_path, ent_path, tokenizer: BertTokenizer,
                 topk_searcher: ITopKSearcher, device,
                 topks: List[int] = [1, 2, 3, 4, 5, 6, 10, 20, 30, 50, 100, 200, 400, 600], topk=20,
                 task_name: str = Union["dev", "test"], max_num_sen: int = None,
                 entid2gid_path=None, gid2entids_path=None, attr2max_len=None, used_attrs=None, attr2cls_id=None):
        self.tokenizer = tokenizer
        self.topk_searcher = topk_searcher
        self.device = device
        self.topks = topks
        self.topk = topk
        self.task_name = task_name
        self.title_infos = []
        self.attr2max_len = attr2max_len
        self.attr2cls_id = attr2cls_id
        self.used_attrs = used_attrs
        if file_path is not None:
            with open(file_path, "r", encoding="utf8") as fr:
                if task_name == "dev":
                    for line in fr:
                        ss = line.strip().split("\t")
                        if len(ss) == 2:
                            self.title_infos.append(ss)
                else:
                    for line in fr:
                        ss = line.strip().split("\t")
                        if len(ss) == 1:
                            ss.append(None)
                            self.title_infos.append(ss)
        if max_num_sen:
            self.title_infos = self.title_infos[0:max_num_sen]
        with open(entid2gid_path, "rb") as fr:
            self.entid2gid = pickle.load(fr)

        with open(gid2entids_path, "rb") as fr:
            self.gid2entid = pickle.load(fr)
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)

    @staticmethod
    def trans_title_ent_to_bert_ipt(title: str, ent_info, tokenizer: BertTokenizer, attr2max_len: Dict,
                                    used_attrs: List[str]):
        """

        :param title:
        :param ent_info:
        :param tokenizer:
        :param attr2max_len:
        :param used_attrs:
        :return:Dict[attr_name:{"input_ids":..}]
        """
        ipt = {}
        for attr_name in used_attrs:
            attr_value = ent_info[attr_name]
            ipt[attr_name] = tokenizer.encode_plus(title, attr_value, max_length=attr2max_len[attr_name],
                                                   padding="max_length", truncation=True, return_tensors="pt")
        return ipt

    def eval_single_title(self, title, model, attr2cls_id, topk_gids=None):
        """

        :param title: str 数据类型
        :param model: 模型
        :return: pred info, sorted List[  gid,score ]
        """
        if topk_gids is None:
            topk_gids = self.topk_searcher.get_topk_g_ids(title, self.topk)
        topk_entinfo = [self.ents[self.gid2entid[g_id][-1]] for g_id in topk_gids]

        ipt_list = []
        for ent_info in topk_entinfo:
            ipt_list.append(
                RankEvaluator.trans_title_ent_to_bert_ipt(title, ent_info, self.tokenizer, self.attr2max_len,
                                                          self.used_attrs))
        # concat 一下
        all_ipt = {attr_name: {} for attr_name in self.used_attrs}
        for attr_name in self.used_attrs:
            for ipt in ipt_list:
                for bert_ipt_name, value in ipt[attr_name].items():
                    if bert_ipt_name not in all_ipt[attr_name]:
                        all_ipt[attr_name][bert_ipt_name] = [value]
                    else:
                        all_ipt[attr_name][bert_ipt_name].append(value)
            # 组合成2维数组，batch_size*max_len
            for bert_ipt_name, value in all_ipt[attr_name].items():
                all_ipt[attr_name][bert_ipt_name] = torch.cat(value, dim=0)
            # 把cls变成专门的ID
            if attr2cls_id is not None and attr_name in attr2cls_id:
                all_ipt[attr_name]["input_ids"][:, 0] = attr2cls_id[attr_name]
        # 送入到device中
        for attr_name in self.used_attrs:
            for bert_ipt_name in all_ipt[attr_name]:
                all_ipt[attr_name][bert_ipt_name] = all_ipt[attr_name][bert_ipt_name].to(self.device)
        # 计算score
        scores = {}
        with torch.no_grad():
            for attr_name in self.used_attrs:
                scores[attr_name] = model(**all_ipt[attr_name]).to("cpu").data.numpy()[:, 0].tolist()
        final_scores = None
        for attr_scores in scores.values():
            if not final_scores:
                final_scores = [i for i in attr_scores]
            else:
                for idx in range(len(attr_scores)):
                    final_scores[idx] += attr_scores[idx]

        pred_info = list(zip(topk_gids, final_scores))
        pred_info.sort(key=lambda x: x[1], reverse=True)
        return pred_info

    def get_ent_id_from_gids(self, title, sorted_gids, model_scores):
        """
        根据预测的gid获取ent id
        :param title:
        :param sorted_gids:
        :return:
        """
        topk_ent_ids = [self.gid2entid[gid][-1] for gid in sorted_gids]
        return topk_ent_ids[0]
        topk_ent_info = [self.ents[entid] for entid in topk_ent_ids]
        topk_scores = [self._score(title, ent_info) for ent_info in topk_ent_info][0:3]
        topk_scores = [(idx, score) for idx, score in enumerate(topk_scores)]
        topk_scores.sort(key=lambda x: x[1], reverse=True)

        if model_scores[0] - 1 > model_scores[1]:
            return topk_ent_ids[0]
        if topk_scores[0][1] < 2:
            return topk_ent_ids[0]
        elif topk_scores[0][1] > topk_scores[1][1]:
            return topk_ent_ids[topk_scores[0][0]]
        else:
            return topk_ent_ids[0]

    def _score(self, title, ent_info):
        comp, comp_score = ent_info["company"], 0
        if not DataUtil.is_null(comp):
            t = pylcs.lcs2(title, comp)
            if t > 1:
                comp_score = 1
        fun, fun_score = ent_info["functions"], 0
        if not DataUtil.is_null(fun):
            t = pylcs.lcs2(title, fun)
            if t > 1:
                fun_score = 1
        bases, bases_score = ent_info["bases"], 0
        if not DataUtil.is_null(bases):
            t = pylcs.lcs2(title, bases)
            if t > 1:
                bases_score = 1
        final_score = comp_score + fun_score + bases_score
        return final_score

    def get_topk_ent_ids(self, model, title, topk=3, topk_gids=None):
        pred_info = self.eval_single_title(title, model, self.attr2cls_id, topk_gids)[0:topk]
        return [self.gid2entid[i[0]][-1] for i in pred_info]

    def eval(self, model, pred_group_data_save_path=None, pred_ent_ids_save_path=None):
        pred_infos = []  # 每个元素代表对每个title的评测结果
        c = 0
        for title, label_ent_id in self.title_infos:
            c += 1
            if c % 100 == 0:
                print(c, len(self.title_infos))
            pred_info = self.eval_single_title(title, model, self.attr2cls_id)
            pred_infos.append(pred_info)
        # 计算topk gid acc
        if self.task_name == "dev":
            num_corrects = [0] * len(self.topks)
            for idx, pred_info in enumerate(pred_infos):
                label_g_id = self.entid2gid[self.title_infos[idx][1]]
                sorted_gids = [i[0] for i in pred_info]
                for idx, topk in enumerate(self.topks):
                    num_corrects[idx] += 1 if label_g_id in sorted_gids[0:topk] else 0
            for idx in range(len(self.topks)):  # 输出结果
                print("topk gid acc:", self.topks[idx], num_corrects[idx] / len(self.title_infos))

        # 获取topk gid的信息
        pred_group_data = []
        for idx in range(len(self.title_infos)):

            # 添加label 信息
            title, label_ent_id = self.title_infos[idx]
            label_g_id = self.entid2gid[label_ent_id]
            t = [title, label_ent_id, label_g_id, 9999, ",".join(self.gid2entid[label_g_id])]  # label 信息
            # 在 top3 但不在top1
            in_topk = label_g_id in [pred_info[0] for pred_info in pred_infos[idx][0:3]] and label_g_id != \
                      pred_infos[idx][0][0]
            t.append(in_topk)
            pred_group_data.append(t)
            # 添加 预测信息
            for pred_info in pred_infos[idx][0:5]:
                t = [title, label_ent_id, pred_info[0], pred_info[1], ",".join(self.gid2entid[pred_info[0]])]
                t.append(in_topk)
                pred_group_data.append(t)
        pred_group_data = pd.DataFrame(pred_group_data,
                                       columns=["Title", "LabelEntId", "PredGId", "PredScore", "EntIdsInGid", "InTopK"])
        if pred_group_data_save_path:
            pred_group_data.to_excel(pred_group_data_save_path, index=False)

        # 获取topk entid acc信息
        num_corr = 0
        pred_ent_ids = []
        for idx, pred_info in enumerate(pred_infos):
            title, label_ent_id = self.title_infos[idx]
            sorted_gids = [i[0] for i in pred_info]
            model_scores = [i[1] for i in pred_info]
            pred_ent_id = self.get_ent_id_from_gids(title, sorted_gids, model_scores)  # 不需要ltitle， 闭着眼搞就行了
            pred_ent_ids.append(str(pred_ent_id) + "\n")
            if pred_ent_id == label_ent_id:
                num_corr += 1

        if pred_ent_ids_save_path:
            with open(pred_ent_ids_save_path, "w", encoding="utf8") as fw:
                fw.writelines(pred_ent_ids)
        print("ent acc:", num_corr / len(self.title_infos))
        if self.task_name == "dev":
            return num_corrects[0] / len(self.title_infos)
        else:
            return 0


if __name__ == "__main__":
    pass
    # file_path = join(PROJ_PATH, "data/rank_score_attr/dev.txt")
    # ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    # gid2entids_path = join(PROJ_PATH, "data/format_data/g_id2ent_id.bin")
    # entid2gid_path = join(PROJ_PATH, "data/format_data/ent_id2g_id.bin")
    # topk_g_ids_path = join(PROJ_PATH, "data/format_data/topk_ent_ids.bin")
    # # attr2max_len = {"name": 80, "functions": 100}
    # # used_attrs = ["name", "functions"]
    # attr2max_len = {"name": 80}
    # used_attrs = ["name"]
    # topk_searcher = TopKSearcher(topk_g_ids_path, entid2gid_path)
    # model_dir = join(PROJ_PATH, "TitleSearch/FirstScorer/rbt_only_name/best_model")
    #
    # tokenizer = BertTokenizer.from_pretrained(model_dir)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FirstScorerModel(model_dir, join(model_dir, "fc_weight.bin")).to(device)
    # model.eval()
    # eva = RankEvaluator(file_path=file_path, ent_path=ent_path, tokenizer=tokenizer,
    #                     topk_searcher=topk_searcher, device=device,
    #                     topks=[1, 2, 3, 10, 20, 30, 50, 100, 200, 400, 600],topk=20,
    #                     task_name="dev", max_num_sen=10,
    #                     entid2gid_path=entid2gid_path, gid2entids_path=gid2entids_path, attr2max_len=attr2max_len,
    #                     used_attrs=used_attrs, attr2cls_id={"name1": 1, "functions": 2})
    #
    # eva.eval(model, "zdd.xlsx", None)
