import requests
from bs4 import BeautifulSoup
import re
import pickle
from DataUtil import DataUtil
import time


def get_fun_by_baidu_baike(name):
    url = "https://baike.baidu.com/item/{}".format(name)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"}
    response = requests.get(url, headers=headers)
    s = response.content.decode(encoding="utf-8")
    text = BeautifulSoup(s, "lxml").get_text()
    ss = re.split("[\r\n]", text)
    ss = [i for i in ss if len(i) > 0]
    for idx in range(len(ss)):
        if name in ss[idx] and "功能主治" in ss[idx]:
            fun = ss[idx + 1]
            fun = re.sub("\s", "", fun)
            if len(fun) > 2:
                return fun
    return None


def get_fun_by_jianke(name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"}
    # 搜索最相关的药品
    response = requests.get("https://search.jianke.com/prod?wd={}".format(name), headers=headers)
    s = response.content.decode(encoding="utf-8")
    beau = BeautifulSoup(s, "lxml")
    productcode = beau.find("div", attrs={"class": "sect"}).attrs["productcode"]

    # 获取药品信息  适应症/功能主治   
    response = requests.get("https://www.jianke.com/product/{}.html".format(productcode), headers=headers)
    s = response.content.decode(encoding="utf-8")
    text = BeautifulSoup(s, "lxml").get_text()
    ss = re.split("[\r\n]", text)
    ss = [re.sub("\s", "", i) for i in ss]
    ss = [i for i in ss if len(i) > 0]
    try:
        for idx in range(len(ss)):
            if "适应症/功能主治" in ss[idx]:
                fun = ss[idx + 1]
                if len(fun) > 2:
                    return fun

    except:
        pass
    for idx in range(len(ss)):
        if "主要作用" in ss[idx]:
            fun = ss[idx + 1]
            if len(fun) > 2:
                return fun
    return None


def enrich_medical_ents(read_path, save_path):
    count = 0
    with open(read_path, "rb") as fr:
        ents = pickle.load(fr)
    for ent_id, ent_info in ents.items():
        if not DataUtil.is_null(ent_info["functions"]):
            continue
        name, fun = ent_info["name"], None
        try:
            fun = get_fun_by_baidu_baike(name)
        except:
            fun = None
        if DataUtil.is_null(fun):
            try:
                fun = get_fun_by_jianke(name)
            except:
                fun = None
        time.sleep(2)
        if not DataUtil.is_null(fun):
            count += 1
            ent_info["functions"] = fun
            print("=" * 10 + str(count) + "=" * 10)
            print("ent id:", ent_id)
            print("name:", name)
            print(fun)
    with open(save_path, "wb") as fw:
        pickle.dump(ents, fw)


if __name__ == "__main__":
    read_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_ents.bin"
    save_path = r"medical_ents_added_fun.bin"
    enrich_medical_ents(read_path, save_path)

    # name = "阿达帕林凝胶"
    # print(get_fun_by_jianke(name))
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36"}
    # # 搜索最相关的药品
    # response = requests.get("https://search.jianke.com/prod?wd={}".format(name), headers=headers)
    # s = response.content.decode(encoding="utf-8")
    # beau = BeautifulSoup(s,"lxml")
    # productcode=beau.find("div",attrs={"class":"sect"}).attrs["productcode"]
    #
    # # 获取药品信息  适应症/功能主治
    # response = requests.get("https://www.jianke.com/product/{}.html".format(productcode), headers=headers)
    # s = response.content.decode(encoding="utf-8")
    # text = BeautifulSoup(s, "lxml").get_text()
    # ss = re.split("[\r\n]", text)
    # ss = [ re.sub("\s", "", i) for i in ss]
    # ss = [i for i in ss if len(i) > 0]
    # try:
    #     for idx in range(len(ss)):
    #         if "适应症/功能主治" in ss[idx]:
    #             fun = ss[idx + 1]
    #             if len(fun) > 2:
    #                 return fun
    #
    # except:
    #     pass
    # for idx in range(len(ss)):
    #     if "主要作用" in ss[idx]:
    #         fun = ss[idx + 1]
    #         if len(fun) > 2:
    #             return fun
