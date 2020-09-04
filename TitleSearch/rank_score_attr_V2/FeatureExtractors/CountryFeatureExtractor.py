from typing import List, Union
from AAttrFeatureExtractor import AAttrFeatureExtractor
import logging
import pickle
import pandas as pd
from DataUtil import DataUtil

logger = logging.getLogger(__name__)


class CountryFeatureExtractor(AAttrFeatureExtractor):
    def __init__(self, areas_path: str):
        logger.info("读取国家地区省份信息...")
        self.areas_path = areas_path
        self.all_areas = None
        self.area2name = None
        self.read_areas()

    def read_areas(self):
        self.area2name = {}
        name2areas = {}
        with open(self.areas_path, "r", encoding="utf8") as fr:
            for idx, line in enumerate(fr):
                ss = line.strip().split("\t")
                if ss[0] in name2areas:
                    name2areas[ss[0]].extend(ss[1:])
                else:
                    name2areas[ss[0]] = ss[1:]
        for k, v in name2areas.items():
            self.area2name[k] = k
            for i in v:
                self.area2name[i] = k
        self.all_areas = set(self.area2name.keys())

    def extract_features(self, title: str, attr_values: List[str]) -> List[List[float]]:
        title_areas = self.title_contain_attr(title)
        if not title_areas:  # title 不包含国家地区信息，返回平均值0.5
            return [[0.5] for _ in range(len(attr_values))]
        ###
        fes = []
        for attr_value in attr_values:
            if DataUtil.is_null(attr_value):  # 实体不包含地区信息
                fes.append([0.2])
                continue
            # print(attr_value)
            attr_areas = set([self.area2name[i] for i in attr_value.split(";")])
            if title_areas.isdisjoint(attr_areas): # 两者都包含地区信息，但是地区不同
                fes.append([0.0])
            else: # 有相同地区
                fes.append([1.0])
        return fes

    def title_contain_attr(self, title: str, **kwargs) -> Union[None, str]:
        areas = []
        for area in self.all_areas:
            if area in title and area + "代购" not in title:
                areas.append(self.area2name[area])
        if len(areas) < 1:
            return None
        return set(areas)

    def get_num_features(self) -> int:
        return 1


if __name__ == "__main__":
    cfe = CountryFeatureExtractor(r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\areas.txt")
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
        res = cfe.extract_features(sen, [ents[ent_id]["place"]])
        # print(idx, sen, ents[ent_id]["place"], res[0][0])
        data.append((sen, ents[ent_id]["place"], res[0][0]))
        # x = input()
    pd.DataFrame(data, columns=["Title", "AttrValue", "Feature"]).to_excel("country.xlsx", index=False)
