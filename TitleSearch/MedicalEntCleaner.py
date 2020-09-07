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


class MedicalEntCleaner:
    def __init__(self, areas_path: str, foreign_name2ch_name_path: str = None, external_fun_path: str = None):
        # 无用的公司后缀名前缀名
        self.company_names = ["生物工程有限公司", "制药有限责任公司", "制药有限公司", "药业股份有限公司", "药业集团有限公司", "药业有限公司",
                              "生物科技股份有限公司", "制药股份有限公司", "医疗器械有限公司", "生物电子技术有限公司",
                              "国际贸易有限公司", "药业有限责任公司", "生物医药科技有限公司", "医药股份有限公司",
                              "药业集团股份有限公司", "股份有限公司", "有限公司", "公司", "株式会社"]
        # 属性名
        self.attr_names = ["生产企业", "主要成分", "症状", "规格", "产地", "功能"]
        # 基本单位
        self.basic_unit = ["片", "粒", "胶囊", "瓶", "盒", "支", "颗粒", "膏药", "膏", "台", "板", "袋", "包", "丸", "贴",
                           "只", "个", "液", "凝胶", "条", "水", "毫克", "毫升", "厘米", "克", "毫米", "ml", "mg", "cm", "mm", "g"]
        # 基本单位 英文->中文
        self.basic_unit2ch_name = {"mg": "毫克", "ml": "毫升", "cm": "厘米", "g": "克", "mm": "毫米"}
        # 数字模式
        self.digit_pattern = re.compile("\d\.{0,1}\d*")
        logger.info("读取国家地区省份信息...")
        self.areas_path = areas_path
        self.all_areas = None
        self.area2name = None
        self.read_areas()
        logger.info("加载中英文名字...")
        self.eng_name2ch_name = {}
        if foreign_name2ch_name_path and os.path.exists(foreign_name2ch_name_path):
            with open(foreign_name2ch_name_path, "r", encoding="utf8") as fr:
                self.eng_name2ch_name = json.load(fr)
        logger.info("加载外部实体功能表...")
        self.ext_fun = {}
        if external_fun_path and os.path.exists(external_fun_path):
            with open(external_fun_path, "r", encoding="utf8") as fr:
                self.ext_fun = json.load(fr)

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

    def clean_extract_company(self, text: str):
        """
        清洗公司名，并从中提取产地信息
        :param text:
        :return:
        """
        text = text.strip().replace("（", "(").replace("）", ")")
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        ### 开始反复提取括号里的内容
        companies, countries = [], []
        flag = True
        while flag:
            flag = False
            for i in re.finditer("\([^())]*\)", text):
                value = text[i.start():i.end()][1:-1]
                if value in self.all_areas:
                    countries.append(value)
                else:
                    companies.append(value)
                text = text[0:i.start()] + text[i.end():]
                flag = True
                break
        companies.append(text)
        companies = list(set(companies))
        ### 对于公司名，提取产地信息,并移除产地信息
        t = []
        for comp in companies:
            for area, name in self.area2name.items():
                if area in comp:
                    comp = comp.replace(area, "")
                    countries.append(area)
            t.append(comp)
        companies = t
        ### 继续清洗companies
        t = []
        for comp in companies:
            for i in self.company_names:
                comp = comp.replace(i, "")
            t.append(comp)
        companies = list(set(t))

        companies = [i[1:] if i.startswith("市") else i for i in companies]
        return list(set(companies)), list(set(countries))

    def clean_company_place(self, company_text, place_text):
        """
        产地和生产企业需要一起清洗
        :return:
        """
        ### 清洗 place
        place_text = Opencc.to_simple(str(place_text))
        place_text = re.sub("\s", "", place_text)
        place_text = re.sub("；", "", place_text)
        place_text = re.sub("印第安", "美国", place_text)
        place_text = re.sub("葛兰素史克", "", place_text)
        place_text = re.sub("zg", "", place_text)
        ss = re.split(";+", place_text)
        ss = [i.strip() for i in ss]
        places = [i for i in ss if len(i) > 1 and i != "nan" and i.lower() != "none"]
        ### 清洗  company
        company_text = Opencc.to_simple(str(company_text)).strip()
        company_text = re.sub("其他", "", company_text)
        ss = re.split(";+", company_text)
        ss = [i.strip() for i in ss]
        comp_list = [i for i in ss if len(i) > 1 and i != "nan" and i.lower() != "none"]
        companies = []
        for i in comp_list:
            comps, countries = self.clean_extract_company(i)
            places.extend(countries)
            companies.extend(comps)

        return ";".join(list(set(places))), ";".join(list(set(companies)))

    def clean_symptom(self, text):
        text = Opencc.to_simple(str(text))
        text = re.sub("\s", "", text)
        return text

    def clean_function(self, text):
        text = Opencc.to_simple(str(text))
        text = re.sub("\s", "", text)
        return text

    def clean_bases(self, text):
        text = Opencc.to_simple(str(text))
        text = re.sub("\s", "", text)
        return text

    def clean_spec(self, text, name):
        if isinstance(text, list):
            if len(text) > 0:
                text = text[0]
            else:
                text = ""
        text = Opencc.to_simple(str(text).strip())
        text = text.replace("包装", "")
        name = name.replace("包邮", "")
        # 匹配基本单位
        spec_info = []
        for i in self.basic_unit:
            if i in text or i in name:
                spec_info.append(i)
                text = text.replace(i, "")
                name = name.replace(i, "")
                if i in self.basic_unit2ch_name:  # 考虑中文
                    spec_info.append(self.basic_unit2ch_name[i])

        if len(spec_info) < 1 and "s" in text:
            spec_info = ["片"]
        # 数量
        for i in re.findall(self.digit_pattern, text):
            spec_info.append(i)
        return ";".join(spec_info)

    def clean_name(self, text):
        """
        清洗name
        :param text:
        :return:
        """
        text = str(text).strip().lower()
        for eng_name, ch_name in self.eng_name2ch_name:
            text = text.replace(eng_name, eng_name + "," + ch_name)
        return text

    def clean_medical_ent_info(self, ent_info):
        subj_id = str(ent_info["subject_id"]).strip()
        # 清洗name
        name = self.clean_name(ent_info["subject"])
        # 获取原始信息
        ori_attr_info = {attr: None for attr in self.attr_names}
        ori_attr_info["name"] = ent_info["subject"]
        for item in ent_info["data"]:
            pred, obj = item["predicate"].strip(), item["object"]
            ori_attr_info[pred] = obj
        # print(len(ori_attr_info))
        # assert len(ori_attr_info) == len(self.attr_names)
        # 开始清洗
        # 清洗 company 和 place
        places, companies = self.clean_company_place(str(ori_attr_info["生产企业"]), str(ori_attr_info["产地"]))
        # 清洗 症状和功能 合并为功能
        symptom, function = self.clean_symptom(ori_attr_info["症状"]), self.clean_function(ori_attr_info["功能"])
        if not DataUtil.is_null(symptom):
            function = symptom
        # 外部信息补全
        if DataUtil.is_null(function) and subj_id in self.ext_fun:
            function = self.ext_fun[subj_id]
        # 清洗成分
        bases = self.clean_bases(ori_attr_info["主要成分"])
        # 清洗规格
        spec = self.clean_spec(ori_attr_info["规格"], name)
        clean_attr_info = {"name": name,
                           "company": companies,
                           "bases": bases,
                           "functions": function,
                           "place": places,
                           "spec": spec}
        return subj_id, ori_attr_info, clean_attr_info

    def clean_ori_data(self, read_path, save_path, xls_save_path=None):
        clean_di, ori_di = {}, {}
        data = []
        with open(read_path, "r", encoding="utf8") as fr:
            for line in fr:
                data.append(json.loads(line))
        for i in data:
            if i["type"] == "Medical":
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
                     ori_di[k]["生产企业"], v["company"],
                     ori_di[k]["主要成分"], v["bases"],
                     ori_di[k]["功能"], v["functions"],
                     ori_di[k]["产地"], v["place"],
                     ori_di[k]["规格"], v["spec"]])
            df = pd.DataFrame(df, columns=["SubjID", "名字", "name",
                                           "生产企业", "company",
                                           "主要成分", "bases",
                                           "功能", "functions",
                                           "产地", "place",
                                           "规格", "spec"])
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

    @staticmethod
    def get_single_char_for_spec(read_path):
        with open(read_path, "rb") as fr:
            ents = pickle.load(fr)
        c = defaultdict(int)
        for subj_id, ent_info in ents.items():
            text = ent_info["spec"]
            for i in text:
                c[i] += 1
        c = [(k, v) for k, v in c.items()]
        c.sort(key=lambda x: x[1], reverse=True)
        for i in c:
            print(i)


if __name__ == "__main__":
    read_path = os.path.join(PROJ_PATH, "data/ori_data/entity_kb.txt")
    save_path = os.path.join(PROJ_PATH, "data/format_data/medical_ents.bin")
    xls_save_path = os.path.join(PROJ_PATH, "data/format_data/medical_ents.xlsx")
    area_path = os.path.join(PROJ_PATH, "data/external_data/areas.txt")
    eng2ch_name_path = os.path.join(PROJ_PATH, "data/external_data/ForeignName2CHName.json.json")
    ext_fun_path = os.path.join(PROJ_PATH, "data/external_data/funnctions.json")
    mec = MedicalEntCleaner(areas_path=area_path, foreign_name2ch_name_path=eng2ch_name_path,
                            external_fun_path=ext_fun_path)
    res = mec.clean_ori_data(read_path, save_path, xls_save_path)
    # MedicalEntCleaner.get_single_char_for_spec(save_path)
    # MedicalEntCleaner.convert_ent_to_xlsx(
    #     r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents_added_fun.bin",
    #     "ents.xlsx")
