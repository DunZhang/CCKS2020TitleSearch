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
from SRScorerModel import SRScorerModel
from SREvaluator import SREvaluator

logging.basicConfig(level=logging.INFO)
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"

if __name__ == "__main__":
    file_path = join(PROJ_PATH, "data/rank_score_attr/dev.txt")
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    gid2entids_path = join(PROJ_PATH, "data/format_data/g_id2ent_id.bin")
    entid2gid_path = join(PROJ_PATH, "data/format_data/ent_id2g_id.bin")
    topk_g_ids_path = join(PROJ_PATH, "data/format_data/final_test_topk_ent_ids.bin")
    # attr2max_len = {"name": 80, "functions": 100}
    # used_attrs = ["name", "functions"]
    attr2max_len = {"name": 80}
    used_attrs = ["name"]
    topk_searcher = TopKSearcher(topk_g_ids_path, entid2gid_path)
    model_dir = join(PROJ_PATH, "TitleSearch/SecondRecall/rbt_tiny/best_model")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRScorerModel(model_dir, join(model_dir, "fc_weight.bin")).to(device)
    model.eval()
    eva = SREvaluator(file_path=file_path, ent_path=ent_path, tokenizer=tokenizer,
                      topk_searcher=topk_searcher, device=device,
                      topks=[1, 2, 3, 10, 20, 30, 50, 100, 200, 400, 600], topk=100,
                      task_name="dev", max_num_sen=None,
                      entid2gid_path=entid2gid_path, gid2entids_path=gid2entids_path, attr2max_len=attr2max_len,
                      used_attrs=used_attrs, attr2cls_id={"name1": 1, "functions": 2})
    ###################################################################################################################
    read_path = join(PROJ_PATH, "data/format_data/final_test.txt")
    save_path = join(PROJ_PATH, "data/format_data/r_final_test_topk_ent_ids.bin")
    res = {}
    with open(read_path, "r", encoding="utf8") as fr:
        for idx, line in enumerate(fr):
            if idx % 200 == 0:
                print(idx)
            sen = line.strip().split("\t")[0]
            ent_ids = eva.get_topk_ent_ids(model, sen, 50)
            # print(line, ent_ids)
            res[sen] = ent_ids

    with open(save_path, "wb") as fw:
        pickle.dump(res, fw)
