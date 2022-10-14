import argparse
import math
import os
from datetime import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import DatasetCapsulation as Data
from tools import Utils, Model
from transformers import BertTokenizer


from demo.BMRCs import utils
from models.BMRC import BMRC,RoBMRC


inference_beta = [0.90, 0.90, 0.90, 0.90] # 推理时候的阈值

"""
    该代码的阅读要点：
        ①
        ②inference(test)和train的过程中有不同的操作

"""

def test(): # 推理过程
    pass

def train(opt): # 训练过程
    optimizer = None


    pass

def dev(opt): # dev过程
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMRCs')
    parser.add_argument('--data_path', type=str, default="../../data/preprocess/")
    parser.add_argument('--data_name', type=str, default=1, choices=1)
    parser.add_argument('--log_path', type=str, default="./log/",help='日志保存的地点')
    parser.add_argument('--save_model_path', type=str, default="./model/",help='训练的模型保存的地点')
    parser.add_argument('--model_name', type=str, default="BMRC",choices=["BMRC","ROBMRC","ATBMRC"],help="选择使用的模型")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"])
    parser.add_argument('--max_aspect_num', type=str, default="max_aspect_num", choices=["max_aspect_num"])

    parser.add_argument('--reload', type=bool, default=False)

    parser.add_argument('--bert_model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.90)

    # 训练过程的超参数
    parser.add_argument('--gpu', type=bool, default=True,help='是否使用GPU')
    parser.add_argument('--epoch_num', type=int, default=50,help='训练的次数')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)
    opt = parser.parse_args()

    # init logger
    logger = utils.get_logger(opt.log_path)

    opt.epoch = opt.train_fuse_model_epoch + opt.epoch

    dt = datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    # 加载BERT相关
    if opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('/bert-base-uncased/vocab.txt')

    if opt.model_name=='BMRC':
        model=BMRC(opt)
    elif opt.model_name=='ROBMRC':
        model=RoBMRC(opt)
    elif opt.model_name=='ATBMRC':
        model=1
    else:
        pass

    if opt.gpu:
        model=model.cuda()

    logger.info(opt)


    if opt.mode=='train':
        args.save_model_path = args.save_model_path + args.data_name + '_' + args.model_name + '.pth'
        train_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'train')
        dev_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'dev')
        test_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'test')



        train(opt)
    



