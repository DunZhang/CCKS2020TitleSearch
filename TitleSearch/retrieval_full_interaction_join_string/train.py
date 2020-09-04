import os
import torch
import logging
from transformers import BertTokenizer, BertModel, BertConfig
from FullInteractionJoinStringDataSet import FullInteractionJoinStringDataSet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from RankModel import RetrievalModel
import json
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from Evaluator import TopKAccEvaluator
import torch.nn.functional as F


def train(conf):
    ### device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### directory
    os.makedirs(conf.out_dir, exist_ok=True)
    best_model_dir = os.path.join(conf.out_dir, "best_model")
    os.makedirs(best_model_dir)
    latest_model_dir = os.path.join(conf.out_dir, "latest_model")
    os.makedirs(latest_model_dir)
    ### global variables
    best_acc = -1
    loss_fw = open(os.path.join(conf.out_dir, "loss.txt"), "w", encoding="utf8")
    acc_fw = open(os.path.join(conf.out_dir, "acc.txt"), "w", encoding="utf8")

    ### train data
    logging.info("get train data loader...")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path)
    train_data_iter = FullInteractionJoinStringDataSet(file_path=conf.train_file_path, ent_path=conf.ent_path,
                                                       tokenizer=tokenizer, batch_size=conf.batch_size,
                                                       max_len=conf.max_len)

    total_steps = int(train_data_iter.steps * conf.num_epochs)
    steps_per_epoch = train_data_iter.steps
    if conf.warmup < 1:
        warmup_steps = int(total_steps * conf.warmup)
    else:
        warmup_steps = int(conf.warmup)
    ### get dev evaluator
    evaluator = TopKAccEvaluator(conf.dev_file_path, conf.ent_path, tokenizer, conf.device,
                                 batch_size=conf.dev_batch_size)
    ### model
    logging.info("define model...")
    if "min" in conf.out_dir:
        logging.info("use min bert model for recall!!!!!!!")
        model = RetrievalModel(BertConfig(vocab_size=8021, hidden_size=66, num_hidden_layers=3, num_attention_heads=3,
                                          intermediate_size=66)).to(conf.device)
        model.tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path)
    else:
        model = RetrievalModel(conf.pretrain_model_path).to(conf.device)

    model.train()
    ### optimizer
    logging.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    logging.info("=================================== trained parameters ===================================")
    for n, p in paras.items():
        logging.info("{}".format(n))
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    ### train
    global_step = 0
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, batch in enumerate(train_data_iter):
            global_step += 1
            step += 1
            batch_data = [i.to(conf.device) for i in batch]
            if step < 2:
                print(batch_data[0].shape)
            pos_score = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                              token_type_ids=batch_data[2])
            neg_score = model(input_ids=batch_data[3], attention_mask=batch_data[4],
                              token_type_ids=batch_data[5])
            loss = torch.nn.functional.relu(neg_score - pos_score + conf.margin)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            if step % conf.print_steps == 0:
                logging.info("epoch:{},\tstep:{}/{},\tloss:{}".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.write("epoch:{},\tstep:{}/{},\tloss:{}\n".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.flush()

            if step % conf.evaluation_steps == 0 or step == steps_per_epoch - 1:
                logging.info("start evaluate...")
                ### eval dataset
        logging.info("start evaluate...")
        acc = evaluator.eval_acc(model, "test.xlsx")

        if acc > best_acc:
            logging.info("save best model to {}".format(best_model_dir))
            best_acc = acc
            model.save(best_model_dir)
        model.save(latest_model_dir)
        acc_fw.write("epoch:{},\tstep:{}/{},\tacc:{}\n".format(epoch, step, steps_per_epoch, acc))

    acc_fw.close()
    loss_fw.close()
