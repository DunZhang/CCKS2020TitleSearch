from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r"G:\data\RBT3")

res = tokenizer.encode("阿迪斯阿斯顿阿三大是阿斯顿阿三大速度啊", max_length=3)
print(res)