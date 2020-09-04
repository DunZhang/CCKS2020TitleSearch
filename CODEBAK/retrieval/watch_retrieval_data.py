from collections import defaultdict

c = defaultdict(int)
read_path = r"G:\Codes\PythonProj\CCKS2020TitleSearch\data\format_data\label_data.txt"

with open(read_path, "r", encoding="utf8") as fr:
    for line in fr:
        ss = line.strip().split("\t")
        if len(ss) != 3:
            continue
        subj_id = ss[1]
        c[subj_id] += 1

data = [ (k,v) for k,v in c.items()]
data.sort(key= lambda x : x[1] ,reverse=True)
t = [ i[0] for i in data[0:10]]
print(t)
for i in data:
    print(i)