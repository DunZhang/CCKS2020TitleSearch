"""
观察数据
"""
import json
from collections import Counter, defaultdict

# read_path  = "data/ori_data/entity_kb.txt"
#
# data = []
# with open(read_path,"r",encoding="utf8") as fr:
#     for line in fr:
#         data.append(json.loads(line))

# watch_data = data[123:345]        

# In[] 统计不同产品类型的个数
# di = defaultdict(int)
# for i in data:
#     di[i["type"]]+=1
# di = dict(di)
# print(di)
# """
# {'Publication': 272618, 'Medical': 4678}
# """
# In[]看看medical的数据
# data_medical = []
# for i in data:
#     if i["type"] == "Medical":
#         data_medical.append(i)


# In[] 观察 训练集
# read_path  = "data/train.txt"
# train_data = []
# with open(read_path,"r",encoding="utf8") as fr:
#     for line in fr:
#         train_data.append(json.loads(line))


# In[] 观察不同类型产品的包含的属性
# di = {}
# for i in data:
#     product_type = i["type"]
#     if product_type not in di:
#         di[product_type] = defaultdict(int)
#     for j in i["data"]:
#         pred = j["predicate"]
#         di[product_type][pred] +=1

'''

{'Publication': defaultdict(int,
             {'出版社': 272618,
              '作者': 272618,
              '出版时间': 272618,
              '卷数': 272618,
              '装帧': 272618}),
              
 'Medical': defaultdict(int,
             {'生产企业': 4678,
              '主要成分': 4678,
              '症状': 4678,
              '规格': 4678,
              '产地': 1791,
              '功能': 2887})}
'''

# In[] 观察训练集属于不同产品的情况
# train_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\label_data.txt"
# kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\entity_kb.json"

# # get kb
# with open(kb_path, "r", encoding="utf8") as fr:
#     kb = json.load(fr)

# print("get test data")
# with open(train_path, "r", encoding="utf8") as fr:
#     train_data = [line.strip().split("\t") for line in fr if len(line.strip().split("\t")) == 3]

# counts = defaultdict(int)
# for i in train_data:
#     subj_id = i[1]
#     typ = kb[subj_id]["type"]
#     counts[typ] += 1

# print(counts) 
# """
#  {'Medical': 83146, 'Publication': 42}
# """

# In[] 获取 药品信息的各类属性集合
# attrs = []
# for i in data:
#     if i["type"] == "Medical":
#         for item in i["data"]:
#             if item["predicate"] == "产地":
#                 attrs.append(item["object"])

# attrs = list(set(attrs))

# In[] 观察训练集LabelID分布情况
# train_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\medical_label_data.txt"
#
# with open(train_path, "r", encoding="utf8") as fr:
#     train_data = [line.strip().split("\t") for line in fr if len(line.strip().split("\t")) == 2]
#
# di = defaultdict(int)
# for i in train_data:
#     di[i[1]] += 1
# res = [(k, v) for k, v in di.items()]
# print("占了多少种id", len(res))
# res.sort(key=lambda x: x[1], reverse=True)
# print(res)
# print(len([1 for i in res if i[1] > 1]))

# In[] 观察实体库的空值情况
# '''
#
# {'Publication': defaultdict(int,
#              {'出版社': 272618,
#               '作者': 272618,
#               '出版时间': 272618,
#               '卷数': 272618,
#               '装帧': 272618}),
#
#  'Medical': defaultdict(int,
#              {'生产企业': 4678,
#               '主要成分': 4678,
#               '症状': 4678,
#               '规格': 4678,
#               '产地': 1791,
#               '功能': 2887})}
# '''
# read_path  = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\medical_entity_kb.json"
# with open(read_path,"r",encoding="utf8") as fr:
#     for line in fr:
#         train_data.append(json.loads(line))
#
# di = {}
# for i in data:
#     product_type = i["type"]
#     if product_type not in di:
#         di[product_type] = defaultdict(int)
#     for j in i["data"]:
#         pred = j["predicate"]
#         di[product_type][pred] +=1
