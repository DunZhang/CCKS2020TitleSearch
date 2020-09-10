import os
import torch
import logging
from transformers import BertTokenizer, BertModel
from TitleCLFDataSet import TitleCLFDataSet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from TitleTypeCLFModel import TitleTypeCLFModel
import json
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from Evaluate import eval


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
    train_data_set = TitleCLFDataSet(file_path=conf.train_file_path, tokenizer=tokenizer, max_len=conf.seq_length)
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=conf.batch_size,
                                   sampler=RandomSampler(train_data_set))

    total_steps = int(len(train_data_loader) * conf.num_epochs)
    steps_per_epoch = len(train_data_loader)
    if conf.warmup < 1:
        warmup_steps = int(len(train_data_loader) * conf.num_epochs * conf.warmup)
    else:
        warmup_steps = int(conf.warmup)
    ### dev data
    logging.info("get dev data loader...")
    dev_data_set = TitleCLFDataSet(file_path=conf.dev_file_path, tokenizer=tokenizer, max_len=conf.seq_length)
    dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=conf.batch_size,
                                 sampler=SequentialSampler(dev_data_set))
    ### test data
    if conf.test_file_path and os.path.exists(conf.test_file_path):
        logging.info("get test data loader...")
        test_data_set = TitleCLFDataSet(file_path=conf.test_file_path, tokenizer=tokenizer, max_len=conf.seq_length)
        test_data_loader = DataLoader(dataset=test_data_set, batch_size=conf.batch_size,
                                      sampler=SequentialSampler(test_data_set))
    else:
        logging.warning("not provide test file path")
        test_data_loader = None

    ### model
    logging.info("define model...")
    clf_model = TitleTypeCLFModel(conf.pretrain_model_path, conf.num_labels, None).to(conf.device)
    clf_model.train()

    ### optimizer
    logging.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(clf_model.named_parameters())
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
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, batch in enumerate(train_data_loader):
            step += 1
            batch_data = [i.to(conf.device) for i in batch]
            if step < 2:
                print(batch_data[0].shape)
                print(tokenizer.decode(batch_data[0][0].cpu().numpy().tolist()))
                print("=" * 20)
                print(batch_data[1][0].cpu().numpy().tolist())
                print("=" * 20)
                print(batch_data[2][0].cpu().numpy().tolist())
                print("=" * 20)
                print(batch_data[3][0].cpu().numpy().tolist())

            logits, loss = clf_model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                     token_type_ids=batch_data[2], labels=batch_data[3])
            ### compute loss\
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
            optimizer.step()
            scheduler.step()
            clf_model.zero_grad()
            optimizer.zero_grad()
            if step % conf.print_steps == 0:
                logging.info("epoch:{},\tstep:{}/{},\tloss:{}".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.write("epoch:{},\tstep:{}/{},\tloss:{}\n".format(epoch, step, steps_per_epoch, loss.data))
                loss_fw.flush()

            if step % conf.evaluation_steps == 0:
                logging.info("start evaluate...")
                ### eval dataset
                eval_res, eval_acc = eval(clf_model, dev_data_loader, conf.device)
                logging.info(
                    "epoch:{},\tstep:{}/{}, dev eval result:".format(epoch, step, steps_per_epoch))
                logging.info("\n{}".format(eval_res))
                acc_fw.write(
                    "epoch:{},\tstep:{}/{},\tdev_dataloader,\tacc:{}\n".format(epoch, step, steps_per_epoch, eval_acc))
                acc_fw.flush()
                if eval_acc > best_acc:
                    logging.info("save best model to {}".format(best_model_dir))
                    best_acc = eval_acc
                    clf_model.save(best_model_dir)
                clf_model.save(latest_model_dir)
                ### test dataset
                if test_data_loader:
                    test_res, _ = eval(clf_model, test_data_loader, conf.device)
                    logging.info(
                        "epoch:{},\tstep:{}/{}, test eval result:".format(epoch, step, steps_per_epoch))
                    logging.info("\n{}".format(test_res))

    acc_fw.close()
    loss_fw.close()
