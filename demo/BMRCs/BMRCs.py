import argparse
import os
#####
import torch
from transformers import  BertTokenizer

#####
from models.BMRC import RoBMRC
from models.BMRC import BMRC


"""
    该代码的阅读要点：
        ①
        ②inference(test)和train的过程中有不同的操作

"""
def test(): # 推理过程
    pass
def train(): # 训练过程
    pass

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--run_type', type=int,
                       default=1, help='1: train, 2: debug train, 3: dev, 4: test')
    parse.add_argument('--save_model_path', type=str,
                       default='checkpoint', help='模型保存的路径')
    parse.add_argument('--add_note', type=str, default='', help='一些保存时用到的信息')
    parse.add_argument('--gpu_num', type=str, default='0', help='选择的GPU的id')
    parse.add_argument('--gpu0_bsz', type=int, default=0,
                       help='第一个gpu的大小，主要用于多卡训练的一种情况')
    parse.add_argument('--epoch', type=int, default=10, help='训练次数')
    parse.add_argument('--batch_size', type=int, default=8,
                       help='batch大小')
    parse.add_argument('--model_type', type=int, default='BMRC',
                       help='BMRC,RoBMRC')
    opt = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)

    if opt.run_type==1:   # 训练
        pass
    elif opt.run_type==2: # debug_train
        pass
    elif opt.run_type==3: # dev
        pass
    elif opt.run_type==3: # test
        pass
    else:
        raise SystemExit('参数run_type有问题，请检查')

    



