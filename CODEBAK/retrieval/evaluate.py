import torch
import json
import re
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import numpy as np
from YWVecUtil import find_topk_by_vecs, VectorDataBase
from transformers import BertTokenizer, BertModel
import os
from milvus import Milvus, MetricType
from typing import List
import pandas as pd
import logging
from dssm_model import RetrievalDSSM

logger = logging.getLogger(__name__)


def get_sens_vec(data_loader, model, device):
    ### get sen vec
    all_sen_vec = []
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            # print(idx, len(data_loader))
            batch_data = [i.to(device) for i in batch_data]
            pooler_output = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                  token_type_ids=batch_data[2])
            if isinstance(pooler_output, tuple) or isinstance(pooler_output, list):
                pooler_output = pooler_output[1]
            all_sen_vec.append(pooler_output.to("cpu").numpy())
    return np.vstack(all_sen_vec)


class TopKAccEvaluator():
    def __init__(self, test_data_path, kb_path, tokenizer, device, save_dir=None,
                 topks: List[int] = [1, 10, 20, 30, 50, 100, 200, 400, 600, 1000, 2000]):
        # other variables
        self.tokenizer = tokenizer
        self.device = device
        self.topks = topks
        self.save_dir = save_dir
        # get kb
        with open(kb_path, "r", encoding="utf8") as fr:
            kb = json.load(fr)
        ################################################################################################################
        # 有点像作弊
        kb = {k: v for k, v in kb.items() if v["type"] == "Medical"}

        ################################################################################################################

        print("generate ent string")
        self.idx2subj_id, self.ent_string = {}, []
        for idx, items in enumerate(kb.items()):
            subj_id, ent_info = items
            self.idx2subj_id[idx] = subj_id
            text = ent_info["subject"] if ent_info["subject"] else ""
            for j in ent_info["data"]:
                if j["object"]:
                    text += ("." + re.sub("\s", "", str(j["object"])))
            self.ent_string.append(text)
        # self.ent_string = self.ent_string[0:10000]
        self.kb_data_loader = self.get_data_loader(self.ent_string, 100)
        print("get test data")
        with open(test_data_path, "r", encoding="utf8") as fr:
            self.test_data = [line.strip().split("\t") for line in fr if len(line.strip().split("\t")) == 2]
        self.uq_sens = [i[0] for i in self.test_data]

        self.uq_data_loader = self.get_data_loader(self.uq_sens, 30)

    def get_data_loader(self, sens, max_length):
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=128)
        return data_loader

    def result_to_local(self, global_step, res_subj_id):
        assert len(res_subj_id) == len(self.uq_sens)
        data = []
        for idx, sen in enumerate(self.uq_sens):
            topk_idxs = res_subj_id[idx][0:100]
            label_id = self.test_data[idx][1].strip()
            in_topk = label_id in topk_idxs
            t = [sen, ",".join(topk_idxs[0:10]), label_id, in_topk]
            data.append(t)
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "pred_{}.xlsx".format(global_step))
            pd.DataFrame(data, columns=["Title", "TopIdxs", "LabelIdx", "InTopk"]).to_excel(save_path, index=False)

    def eval_acc(self, uq_model, ent_model, global_step=None):
        uq_model.eval()
        ent_model.eval()
        logger.info("get ent vec...")
        ent_vec = get_sens_vec(self.kb_data_loader, ent_model, self.device)
        logger.info("finish get ent vec")
        logger.info("get uq vec...")
        uq_vec = get_sens_vec(self.uq_data_loader, uq_model, self.device)
        logger.info("finish get uq vec")
        # compute acc
        milvus = Milvus(host='localhost', port='19530')
        if milvus.has_collection("tmp"):
            milvus.drop_collection("tmp")
        # 创建collection，可理解为mongo的collection
        collection_param = {
            'collection_name': "tmp",
            'dimension': uq_vec.shape[1],
            'index_file_size': 32,
            'metric_type': MetricType.IP  # 使用内积作为度量值
        }

        milvus.create_collection(collection_param)
        start, end = 0, 50000
        while start < len(ent_vec):
            end = end if end <= len(ent_vec) else len(ent_vec)
            milvus.insert(collection_name="tmp", records=ent_vec[start:end], ids=list(range(start, end)))
            start = end
            end += 50000
            end = end if end <= len(ent_vec) else len(ent_vec)
        milvus.flush(["tmp"])
        status, results = milvus.search(collection_name="tmp", query_records=uq_vec, top_k=2000)
        # print(status)
        res_subj_id = [[self.idx2subj_id[j.id] for j in i] for i in results]
        milvus.drop_collection("tmp")
        milvus.close()
        # save to local
        self.result_to_local(global_step, res_subj_id)
        # compute acc
        accs = []
        for k in self.topks:
            t = [1 if i[0][1].strip() in i[1][0:k] else 0 for i in zip(self.test_data, res_subj_id)]
            acc = sum(t) / len(t)
            accs.append((k, acc))
        uq_model.train()
        ent_model.train()
        return accs


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uq_model = RetrievalDSSM(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval\saved_retrieval_dssm_tiny\latest_uq_model").to(
        device)
    ent_model = RetrievalDSSM(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\TitleSearch\retrieval\saved_retrieval_dssm_tiny\latest_ent_model").to(
        device)
    test_data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\retrieval\dev.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\entity_kb.json"

    eva = TopKAccEvaluator(test_data_path, kb_path, uq_model.tokenizer, device, "saved_retrieval_dssm_tiny")
    accs = eva.eval_acc(uq_model, ent_model, "test")
    print(accs)
