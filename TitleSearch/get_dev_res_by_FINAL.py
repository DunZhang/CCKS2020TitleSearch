import warnings

warnings.filterwarnings("ignore")
import pylcs
import logging
import os
import sys
from os.path import join
from typing import List
import numpy as np
import torch
from YWVecUtil import find_topk_by_vecs, BERTSentenceEncoder, VectorDataBase
import transformers
from transformers import BertTokenizer

from FirstScorerModel import FirstScorerModel
from RankEvaluator import RankEvaluator
from TopKSearcher import TopKSearcher
from FINALRESULT import FINALRESULT
import random

logging.getLogger("YWVecUtil.SentenceEncoder.BERTSentenceEncoder").setLevel(logging.WARN)
logging.getLogger("YWVecUtil.VectorSearch.VectorSearchUtil").setLevel(logging.WARN)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging.basicConfig(level=logging.ERROR)
if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sen_encoder_dir = join(PROJ_PATH, "PreTrainedModels/simbert_torch")
    historical_titles_path = join(PROJ_PATH, "data/format_data/all_label_data.txt")  # 所有的训练数据
    historical_titles_vec_path = join(PROJ_PATH, "data/format_data/all_label_data_vecs.npy")  # 所有的训练数据
    #
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    gid2entids_path = join(PROJ_PATH, "data/format_data/g_id2ent_id.bin")
    entid2gid_path = join(PROJ_PATH, "data/format_data/ent_id2g_id.bin")
    recall_topk_entids_path = join(PROJ_PATH, "data/format_data/topk_ent_ids.bin")
    rank_score_model_dir = join(PROJ_PATH, "TitleSearch/FirstScorer/rbt_only_name_V1/best_model")

    # 初始化模型
    zd = FINALRESULT(sen_encoder_dir=None, device=device, historical_titles_path=historical_titles_path,
                     historical_titles_vec_path=historical_titles_vec_path,
                     recall_topk_entids_path=recall_topk_entids_path, entid2gid_path=entid2gid_path,
                     ent_path=ent_path, gid2entids_path=gid2entids_path, rank_score_model_dir=rank_score_model_dir,
                     )
    # 获取最终结果
    write_data = []
    num_corr = 0
    with open(r"G:\Codes\CCKS2020TitleSearch\data\format_data\train.txt", "r", encoding="utf8") as fr:
        sens = [line.strip().split("\t") for line in fr]
    sens = random.sample(sens, 2000)
    for idx, (sen, label) in enumerate(sens):
        if idx % 200 == 0:
            print(idx, len(sens))
        pred = zd.get_final_result(sen)
        if pred == label:
            num_corr += 1
        write_data.append(str(pred) + "\n")

    print(num_corr / len(sens))
    # with open("zdd0928.txt", "w", encoding="utf8") as fw:
    #     fw.writelines(write_data)
