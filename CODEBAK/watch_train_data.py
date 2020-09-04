import json
from collections import defaultdict

train_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\train.txt"
kb_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\entity_kb.json"

# get kb
with open(kb_path, "r", encoding="utf8") as fr:
    kb = json.load(fr)

print("get test data")
with open(train_path, "r", encoding="utf8") as fr:
    train_data = [line.strip().split("\t") for line in fr if len(line.strip().split("\t")) == 2]

counts = defaultdict(int)
for i in train_data:
    subj_id = i[1]
    typ = kb[subj_id]["type"]
    counts[typ] += 1

print(counts)
