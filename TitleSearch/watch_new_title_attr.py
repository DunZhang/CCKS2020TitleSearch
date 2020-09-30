"""
把训练集的Title的作为新属性，计算相似度值
"""
from YWVecUtil import find_topk_by_sens, BERTSentenceEncoder
import torch
import os
import logging
import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_test_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\final_test.txt"
    train_path = r"G:\Codes\CCKS2020TitleSearch\data\format_data\all_label_data.txt"
    
    
    
    sen_encoder = BERTSentenceEncoder(r"G:\Data\simbert_torch", device)

    with open(final_test_path, "r", encoding="utf8") as fr:
        test_sens = [line.strip() for line in fr]
    train_sens,sen2id=[],{}
    with open(train_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")
            if len(ss)!=2:
                continue
            train_sens.append(ss[0])
            sen2id[ss[0]] = ss[1]
    
    # vecs = sen_encoder.get_sens_vec(train_sens)
    # np.save(r"G:\Codes\CCKS2020TitleSearch\data\format_data\all_label_data_vecs.npy",vecs)
    ##########################################################################     
    # 用模型做补充
    # added = {}
    # with open("nosimbert.txt", "r", encoding="utf8") as fr:
    #     for line in fr:
    #         ss = line.strip().split("\t")
    #         if len(ss)!=2:
    #             print(ss)
    #             continue
    #         t = ss[0]
    #         if t.startswith("\""):
    #             t = t[1:]
    #         if t.endswith("\""):
    #             t = t[0:-1]
    #         added[t] = ss[1]
    ##########################################################################             
    res = find_topk_by_sens(sen_encoder, test_sens, train_sens, 10)
    
    
    
    
    
    
    
    data = []
    upload = []
    c = 0
    for i in res:
        if i[2][0]>=0.77:
            c +=1
            upload.append(str(sen2id[i[1][0]])+"\n")
            data.append([i[0],i[1][0],i[2][0],sen2id[i[1][0]]])
        else:
            data.append([i[0],i[1][0],i[2][0],99999999])
            upload.append("999999999\n")
        # for j,k in zip(i[1],i[2]):
        #     data.append([i[0],j,k])
    # pd.DataFrame(data,columns=["测试问题","训练集问题","score","EntID"]).to_excel("testsen.xlsx",index=False)
    
    print(c/len(res),len(res)-c,c)
    with open("clean90.txt","w",encoding="utf8") as fw:
        fw.writelines(upload)