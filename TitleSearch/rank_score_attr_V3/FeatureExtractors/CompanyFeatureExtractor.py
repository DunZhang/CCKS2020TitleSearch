from typing import List, Union
from AAttrFeatureExtractor import AAttrFeatureExtractor
import logging
import pickle
import pandas as pd
from DataUtil import DataUtil
import pylcs

logger = logging.getLogger(__name__)


class CompanyFeatureExtractor(AAttrFeatureExtractor):
    def __init__(self, ):
        pass

    def extract_features(self, title: str, attr_values: List[str]) -> List[List[float]]:
        fes = []
        title_set = set(title)
        for attr_value in attr_values:
            if DataUtil.is_null(attr_value):  # 实体不包含公司信息
                fes.append([0.0])
                continue
            # print(attr_value)
            # fes.append([len(set(attr_value).intersection(title_set)) * 1.0])
            lcs_len = pylcs.lcs2(title, attr_value)
            if lcs_len < 2: lcs_len = 0.0
            if lcs_len > 3: lcs_len = 3.0
            fes.append([lcs_len])
        return fes

    def title_contain_attr(self, title: str, **kwargs) -> Union[None, str]:
        raise NotImplementedError

    def get_num_features(self) -> int:
        return 1

    def get_names(self) -> List[str]:
        return ["CompanyNumSameChar"]


if __name__ == "__main__":
    cfe = CompanyFeatureExtractor()
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
        res = cfe.extract_features(sen, [ents[ent_id]["company"]])
        # print(idx, sen, ents[ent_id]["place"], res[0][0])
        data.append((sen, ents[ent_id]["company"], res[0][0]))
        # x = input()
    pd.DataFrame(data, columns=["Title", "AttrValue", "Feature"]).to_excel("company.xlsx", index=False)
