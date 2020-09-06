from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained(r"G:\data\RBT3")

# res = tokenizer.encode("阿迪斯阿斯顿阿三大是阿斯顿阿三大速度啊", max_length=3)
# print(res)
import json

print("test")
data = {}
with open(r"C:\Users\Zhdun\Desktop\外部数据.txt", "r", encoding="utf8") as fr:
    lines = [line.strip() for line in fr]

for idx, line in enumerate(lines):
    if line.startswith("======="):
        ent_id = lines[idx + 1].replace("ent id:","").strip()
        fun = lines[idx + 3].strip()
        data[ent_id] = fun

with open(r"G:\Codes\CCKS2020TitleSearch\data\external_data\funnctions.json", "w", encoding="utf8") as fw:
    json.dump(data, fw, ensure_ascii=False)
