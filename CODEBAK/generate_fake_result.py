"""  generate fake result"""

write_data = []
with open("data/ori_data/dev.txt", "r", encoding="utf8") as fr:
    for line in fr:
        write_data.append("N/A\n")

with open("pred.txt", "w", encoding="utf8") as fw:
    fw.writelines(write_data)
