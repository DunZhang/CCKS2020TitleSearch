# -*- coding: utf-8 -*-
"""
@Time : 2020/9/12 13:47 
@Author : WangBin
@File : Test.py 
@Software: PyCharm 
"""

import logging
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from Evaluate import get_pre_for_test
from TitleCLFDataSet import TitleCLFDataSet
from TitleTypeCLFModel import TitleTypeCLFModel


def test(conf):
    ### device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(conf.test_pretrain_model_path)
    ### test data
    if conf.test_file_path and os.path.exists(conf.test_file_path):
        logging.info("get test data loader...")
        test_data_set = TitleCLFDataSet(file_path=conf.test_file_path, tokenizer=tokenizer, max_len=conf.seq_length,
                                        has_label=False)
        test_data_loader = DataLoader(dataset=test_data_set, batch_size=conf.batch_size,
                                      sampler=SequentialSampler(test_data_set))
    else:
        logging.warning("not provide test file path")
        test_data_loader = None

    ### model
    logging.info("define model...")
    model_fc_path = os.path.join(conf.test_pretrain_model_path, 'classifier.bin')
    clf_model = TitleTypeCLFModel(conf.test_pretrain_model_path, conf.num_labels, model_fc_path).to(conf.device)

    if test_data_loader:
        label2id = {'0': "medical", '1': "book"}
        pres = get_pre_for_test(clf_model, test_data_loader, conf.device)
        pres = [label2id[str(pp)] for pp in pres]
        with open(conf.test_file_path, 'r', encoding='utf-8') as f_test:
            test_data = f_test.readlines()

        print('preds number: {}'.format(len(pres)))
        print('test data number: {}'.format(len(test_data)))
        from collections import Counter
        count = Counter(pres)
        print(count)
        with open(os.path.join(conf.out_dir, 'test_preds.txt'), 'w', encoding='utf-8')as ff:
            for ppp, tdata in zip(pres, test_data):
                ff.write(tdata.strip() + '\t' + ppp + '\n')
        f_book = open(os.path.join(conf.out_dir, 'test_preds_book.txt'), 'w', encoding='utf-8')
        with open(os.path.join(conf.out_dir, 'test_preds_medical.txt'), 'w', encoding='utf-8')as f_m:
            for ppp, tdata in zip(pres, test_data):
                if ppp == 'medical':
                    f_m.write(tdata.strip() + '\t' + ppp + '\n')
                elif ppp == 'book':
                    f_book.write(tdata.strip() + '\t' + ppp + '\n')
                else:
                    print('wrong!!!!')
        f_book.close()
        print('Write test data result to file {} done!'.format('test_preds.txt'))

    print('test done !!!!')
