"""

三元组数据集

Bert的输入
[CLS]你好啊[SEP]嗯呢[SEP]
101, 765, 1681, 1359, 102, 1426, 1254, 102
0, 0, 0, 0, 0, 1, 1, 1
1, 1, 1, 1, 1, 1, 1, 1
"""
from typing import List
import torch
from transformers import BertTokenizer
import pickle
from ITopKSearcher import ITopKSearcher
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
import logging
from EntScorerModel import EntScorerModel
from typing import Union
import os
import pandas as pd
from copy import deepcopy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class RSAEvaluator():
    def __init__(self, file_path, ent_path, tokenizer: BertTokenizer, topk_searcher: ITopKSearcher, device,
                 topks: List[int] = [1, 10, 20, 30, 50, 100, 200, 400, 600], task_name: str = Union["dev", "test"]):
        self.tokenizer = tokenizer
        self.topk_searcher = topk_searcher
        self.device = device
        self.topks = topks
        self.task_name = task_name
        self.num_examples_per_record = 5  # 每个数据生成的负类个数
        self.ipt_names = ["input_ids", "attention_mask", "token_type_ids", "null_val_mask"]
        self.attr2len = {"name": 60, "company": 50, "bases": 100, "functions": 100, "place": 50}
        self.attr2token_id = {"name": 1, "company": 2, "bases": 3, "functions": 4, "place": 5}  # unused 1 2 3 4 5
        self.attr_names = ["name", "company", "bases", "functions", "place"]
        # self.pos_ipts, self.neg_ipts = None, None
        # self.idx = None
        # self.start, self.end = -1, -1
        self.ent_id2token_id = {}
        # 读取实体
        logger.info("读取实体库...")
        with open(ent_path, "rb") as fr:
            self.ents = pickle.load(fr)
        logger.info("完成读取实体库")
        self.all_ent_id = list(self.ents.keys())
        # 训练数据
        logger.info("读取测试数据...")
        self.titles = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                self.titles.append(ss)
        logger.info("完成读取测试数据")
        # 把title和训练数据都变成topken_id
        logger.info("测试数据变为token id...")
        for i in self.titles:
            i.append(tokenizer.encode(i[0]))
        logger.info("测试训练数据变token id")
        logger.info("实体数据变为token id...")
        self.ents2token_id()
        logger.info("完成实体数据变token id")

    def ents2token_id(self):
        """
        ent 的每个字段做编码，空的为None
        :return:
        """
        for ent_id, ent_info in self.ents.items():
            t = {}
            for attr_name in self.attr2len.keys():
                attr_value = str(ent_info[attr_name])
                if len(attr_value) < 1 or attr_value.lower() in ["none", "nan", "null"]:
                    # print("attr 为空", attr_name, ent_info)
                    t[attr_name] = []
                else:
                    t[attr_name] = self.tokenizer.encode(ent_info[attr_name])[1:]  # 因为是拼接不需要[CLS]
            self.ent_id2token_id[ent_id] = t

    def get_model_input(self, titles_ids, cand_ent_ids):
        ipts = {"name": {}, "company": {}, "bases": {}, "functions": {}, "place": {}}
        for attr_name in ipts.keys():
            t_titles_ids = deepcopy(titles_ids)
            for i in t_titles_ids:
                i[0] = self.attr2token_id[attr_name]
            res = self.pairs2pt(zip(t_titles_ids, [self.ent_id2token_id[ent_id][attr_name] for ent_id in cand_ent_ids]),
                                self.attr2len[attr_name])
            ipts[attr_name] = {"input_ids": res[0].to(self.device), "attention_mask": res[1].to(self.device),
                               "token_type_ids": res[2].to(self.device), "null_val_mask": res[3].to(self.device)}
        return ipts

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

    def gen_triples(self, title, title_ids):
        """ 为一个title生成带预测数据即(title,cand_ent_id) """
        res = []
        topk_ent_ids = self.topk_searcher.get_topk_ent_ids(title, topk=100)
        for ent_id in topk_ent_ids:
            res.append([title_ids, ent_id])
        return res

    def eval_single_title(self, title_info, model):
        if self.task_name == "dev":
            title, label_ent_id, title_id = title_info
        else:
            label_ent_id = None
            title, title_id = title_info
        all_triples = self.gen_triples(title, title_id)
        titles_ids = [i[0] for i in all_triples]
        cand_ent_ids = [i[1] for i in all_triples]
        ipts = self.get_model_input(titles_ids, cand_ent_ids)
        scores = model(ipts)
        for k in scores:
            scores[k] = scores[k].to("cpu").data.numpy()[:, 0].tolist()
        # cand_ent_ids, scores
        return cand_ent_ids, scores, label_ent_id

    def eval(self, model, xls_path=None, test_data_save_path=None):
        model.eval()
        data = []
        pred_idxs = []
        num_corrects = [0] * len(self.topks)
        with torch.no_grad():
            for c, title_info in enumerate(self.titles):
                if c % 10 == 0:
                    logger.info("eval:{}/{}".format(c, len(self.titles)))
                cand_ent_ids, scores, label_ent_id = self.eval_single_title(title_info, model)
                # comput topk acc
                sorted_ent_ids = list(zip(cand_ent_ids, scores["total_score"], list(range(len(cand_ent_ids)))))
                sorted_ent_ids.sort(key=lambda x: x[1], reverse=True)
                sorted_indexs = [i[2] for i in sorted_ent_ids]
                sorted_ent_ids = [i[0] for i in sorted_ent_ids]
                pred_idxs.append(sorted_ent_ids[0])
                for idx, topk in enumerate(self.topks):
                    num_corrects[idx] += 1 if label_ent_id in sorted_ent_ids[0:topk] else 0
                # format pred result

                # 添加title和真实标签ent的信息数据
                if label_ent_id in sorted_ent_ids:
                    idx = sorted_indexs[sorted_ent_ids.index(label_ent_id)]
                    t = [title_info[0], label_ent_id, label_ent_id, sorted_ent_ids[0] == label_ent_id,
                         round(scores["total_score"][idx], 3)]
                    for attr_name in self.attr_names:
                        t.append(
                            str(self.ents[label_ent_id][attr_name]) + "-->" + str(
                                round(scores[attr_name + "_score"][idx], 3)))
                    data.append(t)
                # 添加top20的信息
                for idx in sorted_indexs[0:20]:
                    pred_ent_id = cand_ent_ids[idx]
                    t = [title_info[0], label_ent_id, pred_ent_id, sorted_ent_ids[0] == label_ent_id,
                         round(scores["total_score"][idx], 3)]
                    for attr_name in self.attr_names:
                        t.append(
                            str(self.ents[pred_ent_id][attr_name]) + "-->" + str(
                                round(scores[attr_name + "_score"][idx], 3)))
                    data.append(t)
        # 输出结果
        for idx in range(len(self.topks)):
            print(self.topks[idx], num_corrects[idx] / len(self.titles))
        model.train()
        # 存储data
        if xls_path:
            cols = ["Title", "LabelID", "PredID", "PredRes", "TotalScore"] + self.attr_names
            pd.DataFrame(data, columns=cols).to_excel(xls_path, index=False)
        # 存储预测结果
        if test_data_save_path:
            with open(test_data_save_path, "w", encoding="utf8") as fw:
                fw.writelines([str(i).strip() + "\n" for i in pred_idxs])
        return num_corrects[0] / len(self.titles)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev.txt"
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    model_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\rank_score_attr\rbt3_relu_share_weight_diff_cls\best_model"
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "bases_attr_scoer"))
    model = EntScorerModel(model_path, share_weight=True).to(device)
    topk_searcher = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\dev_topk_ent_ids.bin")
    # file_path, ent_path, model: EntScorerModel, tokenizer: BertTokenizer,
    # topk_searcher: ITopKSearcher, topks: List[int] = [1, 10, 20, 30, 50, 100, 200, 400, 600],
    # task_name: str = Union["dev", "test"])
    evaluator = RSAEvaluator(file_path, ent_path, tokenizer, topk_searcher=topk_searcher, device=device,
                             task_name="dev")
    evaluator.eval(model, "rbt3_relu_share_weight_diff_cls.xlsx")
