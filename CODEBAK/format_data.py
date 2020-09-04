""" 重新整理数据 """
import json
import re
from openccpy.opencc import Opencc
from copy import deepcopy
from os.path import join
import pandas as pd

PROJ_PATH = r"G:\Codes\PythonProj\CCKS2020TitleSearch"


# 处理知识库
def clean_format_ent_kb():
    read_path = join(PROJ_PATH, "data/ori_data/entity_kb.txt")
    save_path = join(PROJ_PATH, "data/format_data/entity_kb.json")
    watch_path = join(PROJ_PATH, "data/format_data/entity_kb.txt")

    kb, write_data = {}, []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in fr:
            di = json.loads(line)
            sub_id = di.pop("subject_id")

            di["subject"] = re.sub("\s", "", str(di["subject"]))
            di["subject"] = Opencc.to_simple(di["subject"])
            for i in di["data"]:
                i["object"] = str(i["object"])
                i["object"] = re.sub("\s", "", i["object"])
                i["object"] = Opencc.to_simple(i["object"])

            sub_id = str(sub_id).strip()
            if sub_id in kb:
                print("Error: 存在相同的subject id")
            kb[sub_id] = di

            # gen watch data
            tdi = deepcopy(di)
            tdi["subject_id"] = sub_id
            write_data.append(json.dumps(tdi, ensure_ascii=False) + "\n")

    with open(save_path, "w", encoding="utf8") as fw:
        json.dump(kb, fw, ensure_ascii=False)
    with open(watch_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)


# 处理训练数据
def clean_format_data(remove_space=True):
    medical_pred2idx = {'生产企业': 2,
                        '主要成分': 3,
                        '症状': 4,
                        '规格': 5,
                        '产地': 6,
                        '功能': 7}

    kb_path = join(PROJ_PATH, "data/format_data/entity_kb.json")
    read_path = join(PROJ_PATH, "data/ori_data/train.txt")
    if remove_space:
        save_path = join(PROJ_PATH, "data/format_data/label_data_no_space.txt")
        xls_save_path = join(PROJ_PATH, "data/format_data/label_no_with_space.xlsx")
    else:
        save_path = join(PROJ_PATH, "data/format_data/label_data_with_space.txt")
        xls_save_path = join(PROJ_PATH, "data/format_data/label_data_with_space.xlsx")
    # get kb
    with open(kb_path, "r", encoding="utf8") as fr:
        kb = json.load(fr)
    data, df = [], []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in fr:
            di = json.loads(line)
            text = di["text"]
            if remove_space:
                text = re.sub("\s", "", str(text))
            else:
                text = str(text).strip()
                text = re.sub("\s+", "|", text)

            text = Opencc.to_simple(text)
            if len(text) < 1:
                continue
            for item in di["implicit_entity"]:
                label_id = str(item['subject_id']).strip()
                data.append("{}\t{}\t{}\n".format(text, label_id, kb[label_id]["type"]))

                # xls
                
                if kb[label_id]["type"] == "Medical":
                    t = [text, kb[label_id]["subject"], "", "", "", "", "", ""]
                    for i in kb[label_id]["data"]:
                        t[medical_pred2idx[i["predicate"]]] = i["object"]
                else:
                    t = [text, kb[label_id]["subject"], "", "", "", "", "", ""]
                df.append(t)

    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines(data)

    df = pd.DataFrame(df, columns=["Title", "名称", "生产企业", "主要成分", "症状", "规格", "产地", "功能"])
    df.to_excel(xls_save_path,index=False)


if __name__ == "__main__":
    clean_format_data(True)
    # clean_format_ent_kb()
