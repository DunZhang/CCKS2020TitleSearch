import pickle
import jieba
from DataUtil import DataUtil
import random
import pandas as pd
import re

if __name__ == "__main__":
    reps = ["该产品用于", "本品适用于", "主用于", "适应症", "适用于", "用于", "本品主要用于"]
    reps.sort(key=lambda x: len(x), reverse=True)

    ent_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    data_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
    with open(ent_path, "rb") as fr:
        ents = pickle.load(fr)
    # print(cfe.area2gid)
    with open(data_path, "r", encoding="utf8") as fr:
        lines = fr.readlines()
    pos_data, neg_data = [], []
    for ent_id, ent_info in ents.items():
        functions = ent_info["functions"]
        name = ent_info["name"]
        if not DataUtil.is_null(functions):
            for i in reps:
                functions = functions.replace(i, "")
        fun_list = [i for i in re.split("[，、。；,.;]", functions) if len(i) > 1 and not DataUtil.is_null(i)]
        for i in fun_list:
            pos_data.append(i + "\t1\n")

        if not DataUtil.is_null(name):
            neg_data.append(name + "\t0\n")
    pos_data = random.sample(pos_data, 5000)
    data = pos_data + neg_data
    random.shuffle(data)
    with open("train.txt", "w", encoding="utf8") as fw:
        fw.writelines(data[1000:])
    with open("dev.txt", "w", encoding="utf8") as fw:
        fw.writelines(data[:1000])
