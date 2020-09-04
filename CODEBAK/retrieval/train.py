import os
import torch
import logging
from transformers import BertTokenizer, BertModel
from retrieval_dataset import RetrievalDSSMDataSet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dssm_model import RetrievalDSSM
import json
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from evaluate import TopKAccEvaluator
import torch.nn.functional as F

def train(conf):
    ### device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### directory
    os.makedirs(conf.out_dir, exist_ok=True)
    best_ent_model_dir = os.path.join(conf.out_dir, "best_ent_model")
    os.makedirs(best_ent_model_dir)
    best_uq_model_dir = os.path.join(conf.out_dir, "best_uq_model")
    os.makedirs(best_uq_model_dir)

    latest_ent_model_dir = os.path.join(conf.out_dir, "latest_ent_model")
    os.makedirs(latest_ent_model_dir)
    latest_uq_model_dir = os.path.join(conf.out_dir, "latest_uq_model")
    os.makedirs(latest_uq_model_dir)
    ### global variables
    best_acc = -1
    loss_fw = open(os.path.join(conf.out_dir, "loss.txt"), "w", encoding="utf8")
    acc_fw = open(os.path.join(conf.out_dir, "acc.txt"), "w", encoding="utf8")

    ### train data
    logging.info("get train data loader...")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_uq_model_path)
    train_data_set = RetrievalDSSMDataSet(file_path=conf.train_file_path, tokenizer=tokenizer)
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=conf.batch_size,
                                   sampler=RandomSampler(train_data_set), drop_last=True)

    total_steps = int(len(train_data_loader) * conf.num_epochs)
    steps_per_epoch = len(train_data_loader)
    if conf.warmup < 1:
        warmup_steps = int(len(train_data_loader) * conf.num_epochs * conf.warmup)
    else:
        warmup_steps = int(conf.warmup)
    ### get dev evaluator
    evaluator = TopKAccEvaluator(conf.dev_file_path, conf.kb_path, tokenizer, conf.device, conf.out_dir)

    # ### dev data
    # logging.info("get dev data loader...")
    # dev_data_set = RetrievalDSSMDataSet(file_path=conf.dev_file_path, tokenizer=tokenizer)
    # dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=conf.batch_size,
    #                              sampler=SequentialSampler(dev_data_set))
    # ### test data
    # if os.path.exists(conf.test_file_path):
    #     logging.info("get test data loader...")
    #     test_data_set = RetrievalDSSMDataSet(file_path=conf.test_file_path, tokenizer=tokenizer)
    #     test_data_loader = DataLoader(dataset=test_data_set, batch_size=conf.batch_size,
    #                                   sampler=SequentialSampler(test_data_set))
    # else:
    #     logging.warning("not provide test file path")
    #     test_data_loader = None
    ### labels
    labels = torch.arange(start=0, end=conf.batch_size, dtype=torch.long, device=conf.device)
    ### loss
    loss_model = torch.nn.CrossEntropyLoss().to(conf.device)

    ### model
    logging.info("define model...")
    uq_model = RetrievalDSSM(conf.pretrain_uq_model_path).to(conf.device)
    uq_model.train()
    ent_model = RetrievalDSSM(conf.pretrain_ent_model_path).to(conf.device)
    ent_model.train()
    ### optimizer
    logging.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(uq_model.named_parameters())
    ent_paras = dict(ent_model.named_parameters())
    logging.info("=================================== trained parameters ===================================")
    for n, p in paras.items():
        logging.info("{}".format(n))
    optimizer_grouped_parameters = [
        # uq model
        {
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},

        # ent model
        {
            "params": [p for n, p in ent_paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in ent_paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    ### train
    global_step = 0
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            step += 1
            batch_data = [i.to(conf.device) for i in batch]
            if step < 2:
                print(batch_data[0].shape)
            uq_repr = uq_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                               token_type_ids=batch_data[2])
            ent_repr = ent_model(input_ids=batch_data[3], attention_mask=batch_data[4],
                                 token_type_ids=batch_data[5])
            # cos
            # uq_repr = torch.nn.functional.normalize(uq_repr, dim=1)
            # ent_repr = torch.nn.functional.normalize(ent_repr, dim=1)
            # logits = torch.mm(uq_repr, ent_repr.t())
            # dot production
            logits = torch.mm(uq_repr, ent_repr.t())

            # euc
            # F.pairwise_distance(logits, logits, p=2)

            if step < 2:
                print("logits shape:", logits.shape)
            ### compute loss
            loss = loss_model(logits, labels)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
            optimizer.step()
            scheduler.step()
            uq_model.zero_grad()
            optimizer.zero_grad()
            if step % conf.print_steps == 0:
                logging.info("epoch:{},\tstep:{}/{},\tloss:{}".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.write("epoch:{},\tstep:{}/{},\tloss:{}\n".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.flush()

            if step % conf.evaluation_steps == 0 or step == steps_per_epoch - 1:
                logging.info("start evaluate...")
                ### eval dataset
                accs = evaluator.eval_acc(uq_model, ent_model, global_step)
                logging.info("==========================TOPK ACC=================================")
                for i in accs:
                    print(i)
                if accs[5][1] > best_acc:
                    logging.info("save best model to {}".format(best_ent_model_dir))
                    best_acc = accs[5][1]
                    uq_model.save(best_uq_model_dir)
                    ent_model.save(best_ent_model_dir)
                uq_model.save(latest_uq_model_dir)
                ent_model.save(latest_ent_model_dir)

    acc_fw.close()
    loss_fw.close()
