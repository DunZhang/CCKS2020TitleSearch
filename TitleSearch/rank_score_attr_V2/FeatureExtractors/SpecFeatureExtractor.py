from typing import List, Union
from AAttrFeatureExtractor import AAttrFeatureExtractor
import logging
import pickle
import pandas as pd
from DataUtil import DataUtil

logger = logging.getLogger(__name__)


class SpecFeatureExtractor(AAttrFeatureExtractor):
    def __init__(self, ):
        pass

    def extract_features(self, title: str, attr_values: List[str]) -> List[List[float]]:
        fes = []
        title = title.replace("包邮", "")
        for attr_value in attr_values:
            if DataUtil.is_null(attr_value):  # 实体不包含规格信息
                fes.append([0.0])
                continue
            # print(attr_value)
            fes.append([sum([1.0 if i in title else 0.0 for i in attr_value.split(";")])])
        return fes

    def title_contain_attr(self, title: str, **kwargs) -> Union[None, str]:
        raise NotImplementedError

    def get_num_features(self) -> int:
        return 1


if __name__ == "__main__":
    cfe = SpecFeatureExtractor()
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
        res = cfe.extract_features(sen, [ents[ent_id]["spec"]])
        # print(idx, sen, ents[ent_id]["place"], res[0][0])
        data.append((sen, ents[ent_id]["spec"], res[0][0]))
        # x = input()
    pd.DataFrame(data, columns=["Title", "AttrValue", "Feature"]).to_excel("spec.xlsx", index=False)
