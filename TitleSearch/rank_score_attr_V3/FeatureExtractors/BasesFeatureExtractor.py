from typing import List, Union, Set
from AAttrFeatureExtractor import AAttrFeatureExtractor
import logging
import pickle
import pandas as pd
from DataUtil import DataUtil
import jieba
import re
import pkuseg
import os

logger = logging.getLogger(__name__)


class BasesFeatureExtractor(AAttrFeatureExtractor):
    def __init__(self, stop_words_path=None):
        self.seg = pkuseg.pkuseg(model_name='medicine')
        if isinstance(stop_words_path, str) and os.path.exists(stop_words_path):
            with open(stop_words_path, "r", encoding="utf8") as fr:
                self.stop_words = set([line.strip() for line in fr])
        else:
            self.stop_words = set()

    def extract_features(self, title: str, attr_values: List[str],
                         title_set: Set = None, attr_values_set: List[Set] = None,
                         title_seg_set: Set = None, attr_values_seg_set: List[Set] = None) -> List[List[float]]:
        title = re.sub("\d", "", title)
        if not title_set:
            title_set = self.txt2set(title)
        if not title_seg_set:
            title_seg_set = self.txt2seg_set(title)
        if not attr_values_set:
            attr_values_set = []
            for v in attr_values:
                attr_values_set.append(self.txt2set(v))
        if not attr_values_seg_set:
            attr_values_seg_set = []
            for v in attr_values:
                attr_values_seg_set.append(self.txt2seg_set(v))
        fes = []
        for attr_value, attr_value_set, attr_value_seg_set in zip(attr_values, attr_values_set, attr_values_seg_set):
            if DataUtil.is_null(attr_value):
                fes.append([0.0])
                continue
            fes.append(
                [len(attr_value_seg_set.intersection(title_seg_set)) * 1.0])
        return fes

    def txt2set(self, txt: str):
        return set([i for i in txt if i not in self.stop_words])

    def txt2seg_set(self, txt: str):
        return set([i for i in self.seg.cut(txt) if i not in self.stop_words and len(i) > 1])

    def title_contain_attr(self, title: str, **kwargs) -> Union[None, str]:
        raise NotImplementedError

    def get_num_features(self) -> int:
        return 1

    def get_names(self):
        return ["BaseNumSameChar", "BaseNumSameWord"]


from os.path import join
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"

if __name__ == "__main__":

    stop_words_path = join(PROJ_PATH, "data/stop_words.txt")
    ent_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    data_path = join(PROJ_PATH, "data/format_data/medical_label_data.txt")
    cfe = BasesFeatureExtractor(stop_words_path)
    with open(ent_path, "rb") as fr:
        ents = pickle.load(fr)
    # print(cfe.area2gid)
    with open(data_path, "r", encoding="utf8") as fr:
        lines = fr.readlines()
    data = []
    for idx, line in enumerate(lines):
        ss = line.strip().split("\t")
        sen, ent_id = ss
        # if "代购" not in sen:
        #     continue
        res = cfe.extract_features(sen, [ents[ent_id]["bases"]])
        # print(idx, sen, ents[ent_id]["place"], res[0][0])
        data.append((sen, ents[ent_id]["bases"], res[0][0], res[0][1]))
        # x = input()
    pd.DataFrame(data, columns=["Title", "AttrValue"] + cfe.get_names()).to_excel("bases.xlsx", index=False)
