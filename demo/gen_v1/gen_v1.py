from datetime import datetime
import os
import argparse


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from demo.gen_v1 import utils
from demo.gen_v1.data_utils import ABSADataset, read_line_examples_from_file
from eval_utils import compute_scores

# 固定随机种子
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构




# initialization
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task_type", default='aste', type=str,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--dataset", default='14lap', type=str,)
    parser.add_argument("--model_name_or_path", default='../../bert/T5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction', type=str,
                        help="The way to construct target sentence, selected from: [annotation, extraction]")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test","zero_shot"])

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument('--seed', type=int, default=7, help="random seed for initialization")
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--epoch_num', type=int, default=30, help='训练的次数')

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)


    # logger & save_path
    parser.add_argument("--log_path", default='./log', type=str)
    parser.add_argument("--save_model_path", default='./check_point', type=str)
    parser.add_argument("--add_note", default='', type=str)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    if not os.path.exists('./check_point'):
        os.mkdir('./check_point')
    if not os.path.exists('./log'):
        os.mkdir('./log')
    task_dir = f"./outputs/{args.task_type}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    print("\n", "="*30, f"NEW EXP: {args.task_type.upper()} on {args.dataset}", "="*30, "\n")


    # 常规固定种子，加载logger
    same_seeds(args.seed)
    dt = datetime.now()
    logger, fh, sh = utils.get_logger(args.log_path,time=dt.strftime('%Y-%m-%d-%H-%M-%S'))
    args.save_model_path = args.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if args.add_note != '':
        args.save_model_path += args.add_note + '.pth'
    else:
        args.save_model_path += '.pth'
    print('\n', args.save_model_path, '\n')


    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # 输入和输出的例子
    print(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
    dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev',
                          paradigm=args.paradigm, task=args.task_type, max_len=args.max_seq_length)
    data_sample = dataset[2]  # a random data sample
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

    #train or test or fewshot

    if args.mode=='train':

        logger.info('train')
        logger.info('Initialize dataset')

        # load dataset
        train_dataset=ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='train',paradigm=args.paradigm, task=args.task_type, max_len=args.max_seq_length)
        dev_dataset=ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev',paradigm=args.paradigm, task=args.task_type, max_len=args.max_seq_length)
        test_dataset=ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='test',paradigm=args.paradigm, task=args.task_type, max_len=args.max_seq_length)

        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, drop_last=True, shuffle=True, num_workers=1)
        t_total = (
            (len(train_dataloader.dataset) // (args.train_batch_size * max(1, 0)))
            // args.gradient_accumulation_steps
            * float(args.epoch_num)
        )



        # initialize model and opti
        logger.info('initialize model and opti')
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        lr_scheduler = scheduler

        if args.gpu is True:
            model=model.cuda()

        start_epoch = 1
        logger.info('begin training......')
        for epoch in range(start_epoch, args.epoch_num + 1):
            logger.info("train")
            print('epoch:',epoch)
            model.train()
            model.zero_grad()
            for batch_index, batch_dict in enumerate(tqdm(train_dataloader,total=len(train_dataset)//args.train_batch_size)):
                optimizer.zero_grad()
                #print(batch_dict)
                lm_labels = batch_dict["target_ids"].cuda()
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                batch_dict["source_ids"]=batch_dict["source_ids"].cuda()
                batch_dict["source_mask"]=batch_dict["source_mask"].cuda()
                batch_dict["target_mask"]=batch_dict["target_mask"].cuda()

                outputs = model(
                    input_ids=batch_dict["source_ids"],
                    attention_mask=batch_dict["source_mask"],
                    labels=lm_labels,
                    decoder_attention_mask=batch_dict['target_mask']
                )

                loss = outputs[0]
                if batch_index %50==0:
                    print(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

            logger.info('dev')
            dev_dataloader = DataLoader(dev_dataset, batch_size=1, drop_last=True, shuffle=True, num_workers=1)
            model.eval()
            outputs=[]
            targets=[]
            with torch.no_grad():
                for batch_dict in tqdm(dev_dataloader):
                    outs = model.generate(input_ids=batch_dict['source_ids'].cuda(),
                                                attention_mask=batch_dict['source_mask'].cuda(),
                                                max_length=128)

                    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                    target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_dict["target_ids"]]

                    outputs.extend(dec)
                    targets.extend(target)
                sents, _ = read_line_examples_from_file(f'./dataset/{args.task_type}/{args.dataset}/dev.txt')
                raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets,
                                                                                                  sents, args.paradigm, args.task_type)
                results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
                           'preds': all_preds, 'preds_fixed': all_preds_fixed}
                logger.info(results)

            logger.info('dev')
            dev_dataloader = DataLoader(dev_dataset, batch_size=2, drop_last=True, shuffle=True, num_workers=1)
            model.eval()
            outputs=[]
            targets=[]
            with torch.no_grad():
                for batch_dict in tqdm(dev_dataloader):
                    outs = model.generate(input_ids=batch_dict['source_ids'].cuda(),
                                                attention_mask=batch_dict['source_mask'].cuda(),
                                                max_length=128)

                    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                    target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_dict["target_ids"]]

                    outputs.extend(dec)
                    targets.extend(target)
                sents, _ = read_line_examples_from_file(f'./dataset/{args.task_type}/{args.dataset}/dev.txt')
                raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets,
                                                                                                  sents, args.paradigm, args.task_type)
                results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
                           'preds': all_preds, 'preds_fixed': all_preds_fixed}
                logger.info(results)

        logger.info('test')
        dev_dataloader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=True, num_workers=1)
        model.eval()
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_dict in tqdm(dev_dataloader):
                outs = model.generate(input_ids=batch_dict['source_ids'].cuda(),
                                      attention_mask=batch_dict['source_mask'].cuda(),
                                      max_length=128)

                dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_dict["target_ids"]]

                outputs.extend(dec)
                targets.extend(target)
            sents, _ = read_line_examples_from_file(f'./dataset/{args.task_type}/{args.dataset}/test.txt')
            raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets,
                                                                                              sents, args.paradigm,
                                                                                              args.task_type)
            results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
                       'preds': all_preds, 'preds_fixed': all_preds_fixed}
            logger.info(results)

    elif args.mode=='test':
        pass
    elif args.mode=='few_shot':
        pass
