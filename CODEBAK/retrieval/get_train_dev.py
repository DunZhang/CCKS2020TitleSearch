""" 生成训练集和开发集 """
import json
import random
import re

from os.path import join

PROJ_PATH = r"G:\Codes\PythonProj\CCKS2020TitleSearch"


def ent2str(ent_info):
    """
    transfor entity info to a string
    :return:
    """
    text = ""
    if ent_info["type"] == "Medical":

        # 产地
        for i in ent_info["data"]:
            if i["predicate"] == "产地" and i["object"] != "中国":
                text += i["object"]
        # subject
        if ent_info["subject"] != "nan" and len(ent_info["subject"]) > 0:
            text += ent_info["subject"]
        #  症状
        for i in ent_info["data"]:
            if i["predicate"] == "症状":
                text += i["object"]
    else:
        text = str(ent_info["subject"])
        if len(text) < 1:
            text = "无书名"
    return text


def get_train_dev_data(kb_path, data_path, train_save_path, dev_save_path):
    useless_idxs = ['265848', '216530', '20848', '122269', '34312', '236778', '80348', '143491', '140025', '275653']
    # get kb
    with open(kb_path, "r", encoding="utf8") as fr:
        kb = json.load(fr)

    # get data
    data = []
    with open(data_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")
            if len(ss) != 3 or ss[1] in useless_idxs:
                continue
            data.append(ss[0:2])
    random.shuffle(data)

    # split data
    # keep dev as origin format
    train_data, dev_data = data[10000:], data[:10000]

    # generate train data
    for i in train_data:
        ent_info = kb[str(i[1])]
        i[1] = ent2str(ent_info)

    # save train data and dev data
    with open(train_save_path, "w", encoding="utf8") as fw:
        fw.writelines(["{}\t{}\n".format(*i) for i in train_data])
    with open(dev_save_path, "w", encoding="utf8") as fw:
        fw.writelines(["{}\t{}\n".format(*i) for i in dev_data])

    print("ok")


if __name__ == "__main__":
    kb_path = join(PROJ_PATH, "data/format_data/entity_kb.json")
    data_path = join(PROJ_PATH, "data/format_data/label_data.txt")
    train_save_path = join(PROJ_PATH, "data/retrieval/train.txt")
    dev_save_path = join(PROJ_PATH, "data/retrieval/dev.txt")
    get_train_dev_data(kb_path, data_path, train_save_path, dev_save_path)
