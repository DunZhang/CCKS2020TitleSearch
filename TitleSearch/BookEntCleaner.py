"""
清洗实体库
"""

import json
import re
from openccpy.opencc import Opencc
import jieba
import pandas as pd
import pickle
from typing import List, Tuple
import logging
from collections import defaultdict
import pandas as pd
import json
import os
from DataUtil import DataUtil
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
logger = logging.getLogger(__name__)


class BookEntCleaner:
    def __init__(self, ):
        self.attr_names = ["出版社", "作者", "出版时间", "卷数", "装帧"]

    def clean_text(self, text):
        text = re.sub("\s", "", str(text))
        text = Opencc.to_simple(text)
        return text

    def clean_medical_ent_info(self, ent_info):
        subj_id = str(ent_info["subject_id"]).strip()
        # 清洗name
        name = self.clean_text(ent_info["subject"])
        # 获取原始信息
        ori_attr_info = {attr: None for attr in self.attr_names}
        ori_attr_info["name"] = ent_info["subject"]
        for item in ent_info["data"]:
            pred, obj = item["predicate"].strip(), item["object"]
            ori_attr_info[pred] = obj
        # print(len(ori_attr_info))
        if len(ori_attr_info) != len(self.attr_names) + 1:
            print(ori_attr_info)
            raise RuntimeError
        clean_attr_info = {"name": name,
                           "press": self.clean_text(ori_attr_info["出版社"]),
                           "author": self.clean_text(ori_attr_info["作者"]),
                           "time": self.clean_text(ori_attr_info["出版时间"]),
                           "volume": self.clean_text(ori_attr_info["卷数"]),
                           "layout": self.clean_text(ori_attr_info["装帧"])}
        return subj_id, ori_attr_info, clean_attr_info

    def clean_ori_data(self, read_path, save_path, xls_save_path=None):
        clean_di, ori_di = {}, {}
        data = []
        with open(read_path, "r", encoding="utf8") as fr:
            for line in fr:
                data.append(json.loads(line))
        for i in data:
            if i["type"] == "Publication":
                subj_id, ori_attr_info, clean_attr_info = self.clean_medical_ent_info(i)
                clean_di[subj_id] = clean_attr_info
                ori_di[subj_id] = ori_attr_info

        with open(save_path, "wb") as fw:
            pickle.dump(clean_di, fw)

        if xls_save_path:
            df = []
            for k, v in clean_di.items():
                df.append(
                    [k, ori_di[k]["name"], v["name"],
                     ori_di[k]["出版社"], v["press"],
                     ori_di[k]["作者"], v["author"],
                     ori_di[k]["出版时间"], v["time"],
                     ori_di[k]["卷数"], v["volume"],
                     ori_di[k]["装帧"], v["layout"]])
            df = pd.DataFrame(df, columns=["SubjID", "名字", "name",
                                           "出版社", "press",
                                           "作者", "author",
                                           "出版时间", "time",
                                           "卷数", "volume",
                                           "装帧", "layout"])
            df.to_excel(xls_save_path, index=False)
        return clean_di

    @staticmethod
    def dencode_ori_data(read_path, save_path):
        """
        把ent kb变为字符串形式存储
        :param read_path:
        :param save_path:
        :return:
        """
        write_data = []
        with open(read_path, "r", encoding="utf8") as fr:
            for line in fr:
                line = str(json.loads(line)) + "\n"
                if "Medical" in line:
                    write_data.append(line)
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines(write_data)

    @staticmethod
    def convert_ent_to_xlsx(read_path, save_path):
        with open(read_path, "rb") as fr:
            ents = pickle.load(fr)

        data = []
        for ent_id, ent_info in ents.items():
            t = [ent_id, ent_info["name"], ent_info["functions"], ent_info["spec"], ent_info["place"],
                 ent_info["bases"], ent_info["company"]]
            data.append(t)
        data = pd.DataFrame(data, columns=["EntID", "name", "functions", "spec", "place", "bases", "company"])
        data.to_excel(save_path, index=False)


if __name__ == "__main__":
    read_path = os.path.join(PROJ_PATH, "data/ori_data/entity_kb.txt")
    save_path = os.path.join(PROJ_PATH, "data/format_data/book_ents.bin")
    xls_save_path = os.path.join(PROJ_PATH, "data/format_data/book_ents.xlsx")
    bec = BookEntCleaner()
    res = bec.clean_ori_data(read_path, save_path, xls_save_path)
    # MedicalEntCleaner.get_single_char_for_spec(save_path)
    # MedicalEntCleaner.convert_ent_to_xlsx(
    #     r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents_added_fun.bin",
    #     "ents.xlsx")
