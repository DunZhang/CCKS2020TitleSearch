"""
医药产品信息抽取类

name 名称：
交集；交集/len(名称长度)

company 生产企业：
交集；交集/len(名称长度)


bases 主要成分：
交集

functions 症状（能治什么病）：
交集

place 产地信息：
暂时不用

"""
import jieba
from typing import List, Dict
import json
import pandas as pd


class MedicalFeatutreExtractor():
    """ 抽取医疗产品和title的相关特征 """

    def __init__(self):
        pass

    @classmethod
    def get_feature(cls, title: str, ent_info: Dict) -> List:
        default_value = 0.1
        fe = []
        title_seg_set = set(jieba.cut(title))
        title_set = set(title)
        # start extract features

        # name 交集；交集/len(名称长度)
        name = ent_info["name"]
        name_seg_set = set(ent_info["name_seg"].split(" "))
        if name == "None" or name == "nan" or len(name) < 1:
            fe.extend([default_value, default_value, default_value, default_value])
        else:
            name_set = set(name)
            t = len(title_set.intersection(name_set))
            fe.append(t)
            fe.append(t / len(name_set))

            t = len(title_seg_set.intersection(name_seg_set))
            fe.append(t)
            fe.append(t / len(name_seg_set))

        # company 交集；交集/len(名称长度)
        company = ent_info["company"]
        company_seg_set = set(ent_info["company_seg"].split(" "))
        if company == "None" or company == "nan" or len(company) < 1:
            fe.extend([default_value, default_value, default_value, default_value])
        else:
            company_set = set(company)
            t = len(title_set.intersection(company_set))
            fe.append(t)
            fe.append(t / len(company_set))

            t = len(title_seg_set.intersection(company_seg_set))
            fe.append(t)
            fe.append(t / len(company_seg_set))

        # bases
        bases = ent_info["bases"]
        bases_seg_set = set(ent_info["bases_seg"].split(" "))
        if bases == "None" or bases == "nan" or len(bases) < 1:
            fe.extend([default_value, default_value])
        else:
            bases_set = set(bases)
            fe.append(len(bases_set.intersection(title_set)))
            fe.append(len(bases_seg_set.intersection(title_seg_set)))
        # functions
        functions = ent_info["functions"]
        functions_seg_set = set(ent_info["functions_seg"].split(" "))
        if functions == "None" or functions == "nan" or len(functions) < 1:
            fe.extend([default_value, default_value])
        else:
            functions_set = set(functions)
            fe.append(len(functions_set.intersection(title_set)))
            fe.append(len(functions_seg_set.intersection(title_seg_set)))
        return fe

    @classmethod
    def get_feature_set(cls, title_set: set, title_seg_set: set, ent_info: Dict) -> List:
        """ 入参为set的版本 """
        default_value = 0.1
        fe = []

        # name 交集；交集/len(名称长度)
        name_set = ent_info["name"]
        name_seg_set = ent_info["name_seg"]
        if not name_set:
            fe.extend([default_value, default_value, default_value, default_value])
        else:
            t = len(title_set.intersection(name_set))
            fe.append(t)
            fe.append(t / len(name_set))

            t = len(title_seg_set.intersection(name_seg_set))
            fe.append(t)
            fe.append(t / len(name_seg_set))

        # company 交集；交集/len(名称长度)
        company_set = ent_info["company"]
        company_seg_set = ent_info["company_seg"]
        if not company_set:
            fe.extend([default_value, default_value, default_value, default_value])
        else:
            t = len(title_set.intersection(company_set))
            fe.append(t)
            fe.append(t / len(company_set))

            t = len(title_seg_set.intersection(company_seg_set))
            fe.append(t)
            fe.append(t / len(company_seg_set))

        # bases
        bases_set = ent_info["bases"]
        bases_seg_set = ent_info["bases_seg"]
        if not bases_set:
            fe.extend([default_value, default_value])
        else:
            fe.append(len(bases_set.intersection(title_set)))
            fe.append(len(bases_seg_set.intersection(title_seg_set)))
        # functions
        functions_set = ent_info["functions"]
        functions_seg_set = ent_info["functions_seg"]
        if not functions_set:
            fe.extend([default_value, default_value])
        else:
            fe.append(len(functions_set.intersection(title_set)))
            fe.append(len(functions_seg_set.intersection(title_seg_set)))
        return fe

    @classmethod
    def get_feature_names(cls):
        return ["Name交集", "Name交集Norm", "NameSeg交集", "NameSeg交集Norm",
                "Company交集", "Company交集Norm", "CompanySeg交集", "CompanySeg交集Norm",
                "Base交集", "BaseSeg交集",
                "Fun交集", "FunSeg交集"]

if __name__ == "__main__":
    # 做一个测试
    read_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
    kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
    save_path = "res.xlsx"

    with open(kb_path, "r", encoding="utf8") as fr:
        kb = json.load(fr)

    data = []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")
            t = [ss[1], ss[2]]
            t.extend(MedicalFeatutreExtractor.get_feature(ss[1], kb[ss[2]]))
            data.append(t)
    cols = ["Title", "LabelID"]
    cols.extend(MedicalFeatutreExtractor.get_feature_names())
    pd.DataFrame(data, columns=cols).to_excel(save_path, index=False)

    for x,y in [(1,2),(11,33),(4,5)]:
        print(x,y)