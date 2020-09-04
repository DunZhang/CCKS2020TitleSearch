"""
基于算好结果TopK searcher 速度会非常快
"""
from ITopKSearcher import ITopKSearcher
import pickle
from typing import List
import random

import logging

logger = logging.getLogger(__name__)


class TopKSearcherBasedRFIJSRes(ITopKSearcher):
    def __init__(self, res_path):
        with open(res_path, "rb") as fr:
            self.title2topk_ent_ids = pickle.load(fr)

    def get_topk_ent_ids(self, title=str, topk: int = 100) -> List[str]:
        if title in self.title2topk_ent_ids:
            return self.title2topk_ent_ids[title][0:topk]
        else:
            logger.warning("title:{} not in ents".format(title))
            return random.sample(self.title2topk_ent_ids.keys(), topk)


if __name__ == "__main__":
    t = TopKSearcherBasedRFIJSRes(
        r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\train_topk_ent_ids.bin")
    sens_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\rank_score_attr\train.txt"
    num_all, c = 0, 0
    with open(sens_path, "r", encoding="utf8") as fr:
        for idx, line in enumerate(fr):
            if idx % 20 == 0:
                print(idx)
            num_all += 1
            title, ent_id = line.strip().split("\t")
            topk_ent_ids = t.get_topk_ent_ids(title)
            c += 1 if ent_id in topk_ent_ids[0:50] else 0
    print(c / num_all)
