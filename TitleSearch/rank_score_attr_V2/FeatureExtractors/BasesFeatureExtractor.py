from typing import List, Union, Set
from AAttrFeatureExtractor import AAttrFeatureExtractor
import logging
import pickle
import pandas as pd
from DataUtil import DataUtil
import jieba
import re

logger = logging.getLogger(__name__)


class BasesFeatureExtractor(AAttrFeatureExtractor):
    def __init__(self, ):
        pass

    def extract_features(self, title: str, attr_values: List[str],
                         title_set: Set[str] = None, attr_values_set: List[Set[str]] = None) -> List[List[float]]:
        title = re.sub("\d", "", title)
        if not title_set:
            title_set = set(title)
        if not attr_values_set:
            attr_values_set = [set(i) for i in attr_values]
        fes = []
        for attr_value, attr_value_set in zip(attr_values, attr_values_set):
            if DataUtil.is_null(attr_value):  # 实体不包含公司信息
                fes.append([0.0])
                continue
            # print(attr_value)
            fes.append([len(attr_value_set.intersection(title_set)) * 1.0])
        return fes

    def extract_features_use_seg(self, title: str, attr_values: List[str],
                                 title_set: Set[str] = None, attr_values_set: List[Set[str]] = None,
                                 title_seg_set: Set[str] = None, attr_values_seg_set: List[Set[str]] = None) -> List[
        List[float]]:
        # 分词
        if not title_seg_set:
            title_seg_set = set(jieba.cut(title))
        if not attr_values_seg_set:
            attr_values_seg_set = [set(jieba.cut(i)) for i in attr_values]
        if not title_set:
            title_set = set(title)
        if not attr_values_set:
            attr_values_set = [set(i) for i in attr_values]
        fes = []
        for attr_value, attr_value_set, attr_value_seg_set in zip(attr_values, attr_values_set, attr_values_seg_set):
            if DataUtil.is_null(attr_value):  # 实体不包含base信息
                fes.append([0.0, 0.0])
                continue
            # print(attr_value)
            fes.append([
                len(attr_value_set.intersection(title_set)) * 1.0,
                len(attr_value_seg_set.intersection(title_set)) * 1.0,
            ])
        return fes

    def title_contain_attr(self, title: str, **kwargs) -> Union[None, str]:
        raise NotImplementedError

    def get_num_features(self) -> int:
        return 1


if __name__ == "__main__":
    cfe = BasesFeatureExtractor()
    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
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
        data.append((sen, ents[ent_id]["bases"], res[0][0]))
        # x = input()
    pd.DataFrame(data, columns=["Title", "AttrValue", "Feature"]).to_excel("bases.xlsx", index=False)
