"""
基于算好结果TopK searcher 速度会非常快
"""
from ITopKSearcher import ITopKSearcher
import pickle
from typing import List
import random

import logging

logger = logging.getLogger(__name__)


class TopKSearcher(ITopKSearcher):
    def __init__(self, res_path, entid2gid_path=None):
        with open(res_path, "rb") as fr:
            self.title2topk_ent_ids = pickle.load(fr)
        if entid2gid_path:
            with open(entid2gid_path, "rb") as fr:
                self.entid2gid = pickle.load(fr)
        else:
            self.entid2gid = None

    def __get_topk_gids_from_ent_ids(self, ent_ids, topk):
        topk_gids, t_set = [], set()
        for ent_id in ent_ids:
            gid = self.entid2gid[ent_id]
            if gid not in t_set:
                topk_gids.append(gid)
                t_set.add(gid)
        return topk_gids[0:topk]

    def get_topk_g_ids(self, title=str, topk: int = 100) -> List[str]:
        if title in self.title2topk_ent_ids:
            if self.entid2gid:
                ent_ids = self.title2topk_ent_ids[title][0:topk * 10]
                return self.__get_topk_gids_from_ent_ids(ent_ids, topk)
            else:
                return self.title2topk_ent_ids[title][0:topk]
        else:
            logger.warning("title:{} not in ents".format(title))
            return random.sample(self.title2topk_ent_ids.keys(), topk)

    def get_topk_ent_ids(self, title=str, topk: int = 100) -> List[str]:
        if title in self.title2topk_ent_ids:
            return self.title2topk_ent_ids[title][0:topk]
        else:
            logger.warning("title:{} not in ents".format(title))
            return random.sample(self.title2topk_ent_ids.keys(), topk)


if __name__ == "__main__":
    # t = TopKSearcher(r"G:\Codes\CCKS2020TitleSearch\data\format_data\topk_ent_ids.bin")
    t = TopKSearcher(r"G:\Codes\CCKS2020TitleSearch\data\format_data\second_recall_topk_ent_ids.bin")
    sens_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
    num_all, c = 0, 0
    write_data = []
    save_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\medica_label_clean.txt"
    with open(sens_path, "r", encoding="utf8") as fr:
        for idx, line in enumerate(fr):
            # if idx % 20 == 0:
            #     print(idx)
            num_all += 1
            title, ent_id = line.strip().split("\t")
            topk_ent_ids = t.get_topk_g_ids(title)
            c += 1 if ent_id in topk_ent_ids[0:20] else 0
            if ent_id not in topk_ent_ids[0:100]:
                write_data.append(line)
    print(c / num_all)

    if save_path:
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines(write_data)
