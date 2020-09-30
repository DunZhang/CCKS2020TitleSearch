"""
集成所有子模块获取最终结果
"""
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

logging.getLogger("YWVecUtil.SentenceEncoder.BERTSentenceEncoder").setLevel(logging.WARN)
logging.getLogger("YWVecUtil.VectorSearch.VectorSearchUtil").setLevel(logging.WARN)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging.basicConfig(level=logging.ERROR)
if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"


class FINALRESULT():
    def __init__(self, sen_encoder_dir, device, historical_titles_path, historical_titles_vec_path,
                 recall_topk_entids_path, entid2gid_path,
                 ent_path, gid2entids_path, rank_score_model_dir,
                 ):
        # 一些非常重要的超参数
        self.title_sim_threshold = 0.8

        #
        self.sen_encoder = None
        if sen_encoder_dir:
            self.sen_encoder = BERTSentenceEncoder(sen_encoder_dir, device)
        print("获取所有训练数据的句向量编码...")
        self.historical_titles, self.historical_titles_labels, self.title2id = [], [], {}
        with open(historical_titles_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) != 2:
                    continue
                self.historical_titles.append(ss[0])
                self.historical_titles_labels.append(ss[1])
                self.title2id[ss[0]] = ss[1]
        if historical_titles_vec_path is None:
            vecs = self.sen_encoder.get_sens_vec(self.historical_titles)
        else:
            vecs = np.load(historical_titles_vec_path)
        self.history_titles_vec_db = VectorDataBase(vecs)
        #
        print("获取recall searcher...")
        self.recall_topk_searcher = TopKSearcher(recall_topk_entids_path, entid2gid_path)
        # 获取evaluator当作topk searcher 用
        print("获取rank topk searcher的相关变量...")
        tokenizer = BertTokenizer.from_pretrained(rank_score_model_dir)
        self.rank_evaluator = RankEvaluator(file_path=None, ent_path=ent_path, tokenizer=tokenizer,
                                            topk_searcher=None, device=device,
                                            topks=[1, 2, 3, 10, 20, 30, 50, 100, 200, 400, 600], topk=30,
                                            task_name="dev", max_num_sen=None,
                                            entid2gid_path=entid2gid_path, gid2entids_path=gid2entids_path,
                                            attr2max_len={"name": 60, "functions": 100},
                                            used_attrs=["name"], attr2cls_id={"name1": 1, "functions": 2})
        self.rank_score_model = FirstScorerModel(rank_score_model_dir, join(rank_score_model_dir, "fc_weight.bin")).to(
            device)

    def historical_titles_search(self, query: str):
        """

        :param query:
        :return: most_similar_title_idx, score
        """
        query_vec = self.sen_encoder.get_sens_vec([query])
        res_index, res_distance = find_topk_by_vecs(query_vec, self.history_titles_vec_db, topk=3)
        return res_index[0, 0], res_distance[0, 0]

    def bin_clf(self, title: str = None, clf_model=None):
        return "medical"

    def recall_get_topk_gids(self, query) -> List[str]:
        """
        :param query:
        :return:  topk gids
        """
        return self.recall_topk_searcher.get_topk_g_ids(query, 20)

    def rank_get_topk_ent_ids(self, query, topk_gids) -> List[str]:
        """
        :param query:
        :param topk_gids:
        :return: top3 ent id
        """
        return self.rank_evaluator.get_topk_ent_ids(self.rank_score_model, query, 3, topk_gids)

    def strategy(self, query, topk_ent_ids):
        """
        返回最终结果
        :param query:
        :param topk_ent_ids:
        :return:
        """
        return topk_ent_ids[0]

    def score(self, query, ent_info):
        pylcs.lcs2(query, ent_info["functions"])

    def get_final_result(self, query):
        # 先用simbert 找一下非常相似的query 并直接返回结果
        if self.sen_encoder is not None:
            title_idx, score = self.historical_titles_search(query)
            if score > self.title_sim_threshold:
                # print("有相似的query")
                return self.historical_titles_labels[title_idx]

        # 进行二分类 判断是属于 医药还是书籍
        # 在这里默认是属于医药的

        # 获取topk gid
        topk_gids = self.recall_get_topk_gids(query)

        # 根据topk gid 获取 最终前三个ent id
        topk_entids = self.rank_get_topk_ent_ids(query, topk_gids)

        # 根据topk_entids 获取最终结果
        pred_ent_id = self.strategy(query, topk_entids)

        return pred_ent_id


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
    recall_topk_entids_path = join(PROJ_PATH, "data/format_data/final_test_topk_ent_ids.bin")
    rank_score_model_dir = join(PROJ_PATH, "TitleSearch/FirstScorer/rbt_only_name_V1/best_model")

    # 初始化模型
    zd = FINALRESULT(sen_encoder_dir=sen_encoder_dir, device=device, historical_titles_path=historical_titles_path,
                     historical_titles_vec_path=historical_titles_vec_path,
                     recall_topk_entids_path=recall_topk_entids_path, entid2gid_path=entid2gid_path,
                     ent_path=ent_path, gid2entids_path=gid2entids_path, rank_score_model_dir=rank_score_model_dir,
                     )
    # 获取最终结果
    write_data = []
    with open(r"G:\Codes\CCKS2020TitleSearch\data\format_data\final_test.txt", "r", encoding="utf8") as fr:
        sens = [line.strip() for line in fr]
    num_corr = 0
    for idx, sen in enumerate(sens):
        if idx % 200 == 0:
            print(idx, len(sens))
        pred = zd.get_final_result(sen)
        write_data.append(str(pred) + "\n")

    with open("zdd0929_80.txt", "w", encoding="utf8") as fw:
        fw.writelines(write_data)
