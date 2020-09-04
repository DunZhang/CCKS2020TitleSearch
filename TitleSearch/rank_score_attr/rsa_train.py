import os
import torch
import logging
from transformers import BertTokenizer
from EntScorerModel import EntScorerModel

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from ScoreAttrDataSet import ScoreAttrDataSet
from TopKSearcherBasedRFIJSRes import TopKSearcherBasedRFIJSRes
from RSAEvaluator import RSAEvaluator


def train(conf):
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # directory
    os.makedirs(conf.out_dir, exist_ok=True)
    best_model_dir = os.path.join(conf.out_dir, "best_model")
    os.makedirs(best_model_dir)
    latest_model_dir = os.path.join(conf.out_dir, "latest_model")
    os.makedirs(latest_model_dir)
    # global variables
    best_acc = -1
    loss_fw = open(os.path.join(conf.out_dir, "loss.txt"), "w", encoding="utf8")
    acc_fw = open(os.path.join(conf.out_dir, "acc.txt"), "w", encoding="utf8")

    # topk seacher
    logging.info("get topk searcher")
    train_topk_searcher = TopKSearcherBasedRFIJSRes(conf.train_topk_ent_path)
    dev_topk_searcher = TopKSearcherBasedRFIJSRes(conf.dev_topk_ent_path)

    # train data
    logging.info("get train data loader...")
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_model_path)
    train_data_iter = ScoreAttrDataSet(file_path=conf.train_file_path, ent_path=conf.ent_path, tokenizer=tokenizer,
                                       batch_size=conf.batch_size, topk_searcher=train_topk_searcher)

    total_steps = int(train_data_iter.steps * conf.num_epochs)
    steps_per_epoch = train_data_iter.steps
    if conf.warmup < 1:
        warmup_steps = int(total_steps * conf.warmup)
    else:
        warmup_steps = int(conf.warmup)
    # get dev evaluator
    evaluator = RSAEvaluator(file_path=conf.dev_file_path, ent_path=conf.ent_path, tokenizer=tokenizer,
                             topk_searcher=dev_topk_searcher, device=conf.device, task_name="dev")
    # model
    logging.info("define model...")
    model = EntScorerModel(conf.pretrain_model_path, share_weight=conf.share_weight).to(conf.device)
    model.train()
    # optimizer
    logging.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    logging.info("=================================== trained parameters ===================================")
    for n, p in paras.items():
        logging.info("{}:\t{}".format(n, p.shape))
    optimizer_grouped_parameters = [{
        "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # train
    global_step = 0
    logging.info("start train")
    for epoch in range(conf.num_epochs):
        for step, batch_data in enumerate(train_data_iter):
            global_step += 1
            step += 1
            for data_di in batch_data:
                for ipts in data_di.values():
                    for ipt_name in ipts:
                        ipts[ipt_name] = ipts[ipt_name].to(conf.device)
            pos_score = model(batch_data[0])["total_score"]
            neg_score = model(batch_data[1])["total_score"]
            if step < 2:
                for ipt_name in train_data_iter.ipt_names:
                    print(ipt_name, batch_data[0]["name"][ipt_name].shape)
                print("score shape", pos_score.shape, neg_score.shape)
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

            if step % conf.evaluation_steps == 0 or step == 10:
                logging.info("start evaluate...")
                acc = evaluator.eval(model)
                if acc > best_acc:
                    logging.info("save best model to {}".format(best_model_dir))
                    best_acc = acc
                    model.save(best_model_dir)
                model.save(latest_model_dir)
                acc_fw.write("epoch:{},\tstep:{}/{},\tacc:{}\n".format(epoch, step, steps_per_epoch, acc))
                acc_fw.flush()

        logging.info("start evaluate...")
        acc = evaluator.eval(model)
        if acc > best_acc:
            logging.info("save best model to {}".format(best_model_dir))
            best_acc = acc
            model.save(best_model_dir)
        model.save(latest_model_dir)
        acc_fw.write("epoch:{},\tstep:{}/{},\tacc:{}\n".format(epoch, step, steps_per_epoch, acc))
        acc_fw.flush()
    acc_fw.close()
    loss_fw.close()
