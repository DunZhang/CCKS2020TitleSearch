""" 输出实体name里英文字符串，人工翻译，作为数据补全 """
import re
import pickle
from os.path import join
import sys

if "win" in sys.platform:
    PROJ_PATH = r"G:\Codes\CCKS2020TitleSearch"
else:
    PROJ_PATH = "/home/wcm/ZhangDun/CCKS2020TitleSearch"
if __name__ == "__main__":
    data_path = join(PROJ_PATH, "data/format_data/medical_ents.bin")
    pattern = "[a-zA-Z ]{4,100}"
    with open(data_path, "rb") as fr:
        data = pickle.load(fr)
    for v in data.values():
        name = v["company"]
        ss = name.strip().split("\t")
        sen = ss[0]
        find_res = re.findall(pattern, sen)
        if len(find_res) > 0:
            print(sen)
            print(find_res[0].strip())
            print("\n")
