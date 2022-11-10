# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-12
# @Do: 对BMRC相关数据的读取与处理

import os
import json
import copy


#import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from transformers import RobertaTokenizer

def get_query_and_answer(domain_type):
    """
    目的是模拟NER中的阅读理解任务第一版（BERT-MRF）
    :param domain_type:
        LAPTOP
        RESTAURANT
    :return:
        forward_category_query_list：
            包含本领域的category
            list的长度是category的数量
        forward_category_answer_list:
            上面的query对应的答案
    """
    if domain_type=='laptop':
        l = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE',
         'CPU#OPERATION_PERFORMANCE',
         'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE', 'POWER_SUPPLY#CONNECTIVITY',
         'SOFTWARE#USABILITY',
         'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY',
         'FANS&COOLING#DESIGN_FEATURES',
         'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY', 'POWER_SUPPLY#GENERAL',
         'PORTS#QUALITY',
         'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY', 'MOUSE#GENERAL',
         'KEYBOARD#MISCELLANEOUS',
         'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS', 'SOFTWARE#PRICE',
         'FANS&COOLING#OPERATION_PERFORMANCE',
         'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL',
         'DISPLAY#OPERATION_PERFORMANCE',
         'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY', 'KEYBOARD#PRICE',
         'SUPPORT#OPERATION_PERFORMANCE',
         'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY',
         'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES',
         'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL', 'PORTS#USABILITY',
         'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY',
         'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'HARDWARE#DESIGN_FEATURES',
         'MEMORY#USABILITY',
         'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY', 'OS#PRICE',
         'SUPPORT#QUALITY',
         'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL',
         'COMPANY#OPERATION_PERFORMANCE',
         'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES', 'Out_Of_Scope#OPERATION_PERFORMANCE',
         'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY', 'DISPLAY#USABILITY',
         'POWER_SUPPLY#QUALITY',
         'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY', 'HARDWARE#GENERAL',
         'COMPANY#PRICE',
         'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'SOFTWARE#PORTABILITY',
         'HARD_DISC#OPERATION_PERFORMANCE',
         'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE',
         'OS#USABILITY',
         'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE', 'LAPTOP#PRICE',
         'OS#GENERAL', 'HARDWARE#PRICE',
         'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'FANS&COOLING#QUALITY',
         'BATTERY#OPERATION_PERFORMANCE',
         'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE', 'KEYBOARD#GENERAL',
         'SOFTWARE#QUALITY',
         'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'WARRANTY#QUALITY',
         'HARD_DISC#QUALITY',
         'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
    else:
        l = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
    cat_to_id={}
    id_to_cat={}
    num=0
    for item in l:
        cat_to_id[item]=num
        id_to_cat[num]=item
        num+=1
    return cat_to_id,id_to_cat

class SYZDataset(Dataset):
    # 四元组Dataset
    def __init__(self,
                 opt,
                 data_path,
                 data_type,
                 dataset_type,
                 ):
        """
        :param opt:  args得到的东西，详情见main函数
        :param data_path:  数据存储的位置，
        :param data_type:  train，dev，还是test，用于构建相应的文件（夹）
        """
        self.data_type = data_type         # ['laptop','restarunt']
        self.dataset_type = dataset_type  #'['train','dev','test']

        # 读文件并保存
        if opt.bert_model_type=='bert-base-uncased':
            self.tokenizer=BertTokenizer.from_pretrained('../../bert/bert-base-uncased/vocab.txt')

        self.cat_to_id,self.id_to_cat=get_query_and_answer(data_type)


        self.data=[]
        if True:#os.path.exists('xxx/xxx/filename') is False:  # 如果没处理过，那就处理
            file_read = open(data_path, 'r', encoding='utf-8')
            file_content = json.load(file_read)
            file_read.close()
            self.data_id_list = [] # 保存id，方便找错
            self.text_list = []    # 保存text，方便debug
            self.quad_list=[]      # 保存四元组
            self.QAs = []          # 保存QA对，实际上处理这个就行了
            self.QAs_r=[]
            self.max_length=0
            for data in file_content:
                self.data_id_list.append(data['ID'])
                self.text_list.append(data['text'])
                self.quad_list.append(data['quad'])
                self.QAs.append(data['QAs'])

            for qa_index,qa in enumerate(self.QAs):
                text=self.text_list[qa_index] # 原文
                # 原数据
                S_A_QUERY=qa['S_A_QUERY']
                S_A_ANSWER=qa['S_A_ANSWER']

                A_O_QUERY=qa['A_O_QUERY']
                A_O_ANSWER=qa['A_O_ANSWER']

                AO_C_QUERY=qa['AO_C_QUERY']
                AO_C_ANSWER=qa['AO_C_ANSWER']

                A_C_QUERY=qa['A_C_QUERY']
                A_C_ANSWER=qa['A_C_ANSWER']

                AC_O_QUERY=qa['AC_O_QUERY']
                AC_O_ANSWER=qa['AC_O_ANSWER']

                AOC_P_QUERY=qa['AOC_P_QUERY']
                AOC_P_ANSWER=qa['AOC_P_ANSWER']

                S_O_QUERY=qa['S_O_QUERY']
                S_O_ANSWER=qa['S_O_ANSWER']

                O_A_QUERY=qa['O_A_QUERY']
                O_A_ANSWER=qa['O_A_ANSWER']

                OA_C_QUERY=qa['OA_C_QUERY']
                OA_C_ANSWER=qa['OA_C_ANSWER']

                O_C_QUERY=qa['O_C_QUERY']
                O_C_ANSWER=qa['O_C_ANSWER']

                OC_A_QUERY=qa['OC_A_QUERY']
                OC_A_ANSWER=qa['OC_A_ANSWER']

                S_C_QUERY=qa['S_C_QUERY']
                S_C_ANSWER=qa['OC_A_ANSWER']

                C_A_QUERY=qa['C_A_QUERY']
                C_A_ANSWER=qa['C_A_ANSWER']

                C_O_QUERY=qa['C_A_QUERY']
                C_O_ANSWER=qa['C_O_ANSWER']

                # 保存一些常用的,当然有些tokenizer直接加就行，这个因地制宜（其实拿hf的bert那个接口能更方便的哈哈）
                T_CLS=self.tokenizer.convert_tokens_to_ids(['[CLS]'])
                T_SEP=self.tokenizer.convert_tokens_to_ids(['[SEP]'])
                T_text=self.tokenizer.convert_tokens_to_ids([word.lower() for word in text])

                ###############
                # 一些初始化
                ###############
                ## 其中->A 或->O的为抽取式任务，构建方法相似；->C或-S为分类任务（对于本次测试来说），构建方法相似
                # （1）S->A
                _forward_S_A_query=[]
                _forward_S_A_query_mask=[]
                _forward_S_A_query_seg=[]
                _forward_S_A_answer_start=[]
                _forward_S_A_answer_end=[]

                # （2）S->O
                _forward_S_O_query=[]
                _forward_S_O_query_mask=[]
                _forward_S_O_query_seg=[]
                _forward_S_O_answer_start=[]
                _forward_S_O_answer_end=[]

                # （3）S->C   # C的暂时先保留，可能使用|C|轮问答也有可能用别的
                _forward_S_C_query=[]
                _forward_S_C_query_mask=[]
                _forward_S_C_query_seg=[]

                _forward_S_C_answer_start=[]
                _forward_S_C_answer_end=[]

                # (4)A->O
                _forward_A_O_query=[]
                _forward_A_O_query_mask=[]
                _forward_A_O_query_seg=[]
                _forward_A_O_answer_start=[]
                _forward_A_O_answer_end=[]

                # (5)A->C
                _forward_A_C_query=[]
                _forward_A_C_query_mask=[]
                _forward_A_C_query_seg=[]

                _forward_A_C_answer=[]# 分类问题，并不需要start和end
                # (6)A,O->C
                _forward_AO_C_query=[]
                _forward_AO_C_query_mask=[]
                _forward_AO_C_query_seg=[]

                _forward_AO_C_answer=[]# 分类问题，并不需要start和end


                # (7)A,C->O
                _forward_AC_O_query=[]
                _forward_AC_O_query_mask=[]
                _forward_AC_O_query_seg=[]
                _forward_AC_O_answer_start=[]
                _forward_AC_O_answer_end=[]

                # (8)O->A
                _forward_O_A_query=[]
                _forward_O_A_query_mask=[]
                _forward_O_A_query_seg=[]

                _forward_O_A_answer_start=[]
                _forward_O_A_answer_end=[]

                # (9)O->C
                _forward_O_C_query=[]
                _forward_O_C_query_mask=[]
                _forward_O_C_query_seg=[]

                _forward_O_C_answer=[]# 分类问题，并不需要start和end

                # (10)O,A->C
                _forward_OA_C_query=[]
                _forward_OA_C_query_mask=[]
                _forward_OA_C_query_seg=[]

                _forward_OA_C_answer=[]# 分类问题，并不需要start和end

                # (11)O,C->A
                _forward_OC_A_query=[]
                _forward_OC_A_query_mask=[]
                _forward_OC_A_query_seg=[]

                _forward_OC_A_answer_start=[]
                _forward_OC_A_answer_end=[]

                # (12)A,O,C->P
                _forward_AOC_P_query=[]
                _forward_AOC_P_query_mask=[]
                _forward_AOC_P_query_seg=[]

                _forward_AOC_P_answer=[]  # 分类问题，并不需要start和end

                # (13) C->A  #在这个的时候其实也得到C了
                _forward_C_A_query=[]
                _forward_C_A_query_mask=[]
                _forward_C_A_query_seg=[]

                _forward_C_A_answer_start=[]
                _forward_C_A_answer_end=[]
                # (14) C->O
                _forward_C_O_query=[]
                _forward_C_O_query_mask=[]
                _forward_C_O_query_seg=[]

                _forward_C_O_answer_start=[]
                _forward_C_O_answer_end=[]


                ################
                # 一系列数据检查操作(待写）
                ################


                ###############
                # 构建模型输入
                # 同类型的：
                #  （1）标注抽取的任务
                #  S->A
                #  S->O
                #  A->O
                #  O->A
                #  A,C->O
                #  O,C->A
                #
                ###############

                # （1）S->A
                T_Q_S_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in S_A_QUERY for word_ in word])
                _forward_S_A_query.append(T_CLS+T_Q_S_A+T_SEP+T_text+T_SEP)
                _S_A_query_mask_temp = [1] * len(_forward_S_A_query[0]) # 因为就1个，直接取0就行
                _S_A_query_seg_temp = [0] * (len(_forward_S_A_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                _forward_S_A_query_mask.append(_S_A_query_mask_temp)
                _forward_S_A_query_seg.append(_S_A_query_seg_temp)

                _forward_S_A_answer_start_temp = [-1] * (len(T_Q_S_A)+2) + S_A_ANSWER[0][0]+[-1]
                _forward_S_A_answer_end_temp = [-1] * (len(T_Q_S_A)+2) + S_O_ANSWER[0][1]+[-1]

                _forward_S_A_answer_start.append(_forward_S_A_answer_start_temp)
                _forward_S_A_answer_end.append(_forward_S_A_answer_end_temp)

                # （2）S->O
                T_Q_S_O=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in S_O_QUERY for word_ in word])
                _forward_S_O_query.append(T_CLS+T_Q_S_O+T_SEP+T_text+T_SEP)# build_inputs_with_special_tokens等效
                _S_O_query_mask_temp = [1] * len(_forward_S_O_query[0]) # 因为就1个，直接取0就行
                _S_O_query_seg_temp = [0] * (len(_forward_S_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                _forward_S_O_query_mask.append(_S_O_query_mask_temp)
                _forward_S_O_query_seg.append(_S_O_query_seg_temp)

                _forward_S_O_answer_start_temp = [-1] * (len(T_Q_S_O)+2) + S_O_ANSWER[0][0]+[-1]
                _forward_S_O_answer_end_temp = [-1] * (len(T_Q_S_O)+2) + S_O_ANSWER[0][1]+[-1]

                _forward_S_O_answer_start.append(_forward_S_O_answer_start_temp)
                _forward_S_O_answer_end.append(_forward_S_O_answer_end_temp)

                # (3)S->C  我建议先不写这个了
                # ### 方法1：多分类问题
                # T_Q_S_C=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in S_C_QUERY for word_ in word])
                # _forward_S_O_query.append(T_CLS+T_Q_S_C+T_SEP+T_text+T_SEP)# build_inputs_with_special_tokens等效
                # _S_O_query_mask_temp = [1] * len(_forward_S_O_query[0]) # 因为就1个，直接取0就行
                # _S_O_query_seg_temp = [0] * (len(_forward_S_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                #
                #
                # if data_type!='laptop':
                #     quest = ["What", "categories","?"]
                #     for c_item in self.cat_to_id:
                #         pass


                ### 方法2：标签融合问题

                # (4)A->O
                # 好像不需要考虑多组问题，A_O_QUERY?
                for ao_index,ao_item in enumerate(A_O_QUERY):
                    T_Q_A_O=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in [ao_item] for word_ in word])
                    _forward_A_O_query.append(T_CLS+T_Q_A_O+T_SEP+T_text+T_SEP)
                    _A_O_query_mask_temp = [1] * len(_forward_A_O_query[0]) # 因为就1个，直接取0就行
                    _A_O_query_seg_temp = [0] * (len(_forward_A_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_A_O_query_mask.append(_A_O_query_mask_temp)
                    _forward_A_O_query_seg.append(_A_O_query_seg_temp)

                    _forward_A_O_answer_start_temp = [-1] * (len(T_Q_A_O)+2) + A_O_ANSWER[ao_index][0]+[-1]
                    _forward_A_O_answer_end_temp = [-1] * (len(T_Q_A_O)+2) + A_O_ANSWER[ao_index][1]+[-1]

                    _forward_A_O_answer_start.append(_forward_A_O_answer_start_temp)
                    _forward_A_O_answer_end.append(_forward_A_O_answer_end_temp)


                # (5)A->C
                # 方法1： 多分类
                # for ac_index,ac_item in enumerate(A_C_QUERY):
                #     T_Q_A_C=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in [ac_item] for word_ in word])
                #     _forward_A_O_query.append(T_CLS+T_Q_A_C+T_SEP+T_text+T_SEP)
                #     _A_O_query_mask_temp = [1] * len(_forward_A_O_query[0]) # 因为就1个，直接取0就行
                #     _A_O_query_seg_temp = [0] * (len(_forward_A_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                #     _forward_A_O_query_mask.append(_A_O_query_mask_temp)
                #     _forward_A_O_query_seg.append(_A_O_query_seg_temp)
                #
                #     _forward_A_O_answer_start_temp = [-1] * (len(T_Q_A_O)+2) + A_O_ANSWER[0][0]+[-1]
                #     _forward_A_O_answer_end_temp = [-1] * (len(T_Q_A_O)+2) + A_O_ANSWER[0][1]+[-1]
                #
                #     _forward_A_O_answer_start.append(_forward_A_O_answer_start_temp)
                #     _forward_A_O_answer_end.append(_forward_A_O_answer_end_temp)
                # 方法2： BMRC构造多个问题

                # (6)A,O->C


                # (7)A,C->O
                for ac_o_index, ac_o_item in enumerate(AC_O_QUERY):
                    T_Q_AC_O = self.tokenizer.convert_tokens_to_ids(
                        [word_.lower() for word in [ac_o_item] for word_ in word])

                    _forward_AC_O_query.append(T_CLS + T_Q_AC_O + T_SEP + T_text + T_SEP)

                    _AC_O_query_mask_temp = [1] * len(_forward_AC_O_query[0])  # 因为就1个，直接取0就行
                    _AC_O_query_seg_temp = [0] * (len(_forward_AC_O_query[0]) - len(T_text) - 1) + [1] * (len(T_text) + 1)

                    _forward_AC_O_query_mask.append(_AC_O_query_mask_temp)
                    _forward_AC_O_query_seg.append(_AC_O_query_seg_temp)

                    _forward_AC_O_answer_start_temp = [-1] * (len(T_Q_AC_O) + 2) + AC_O_ANSWER[ac_o_index][0] + [-1]
                    _forward_AC_O_answer_end_temp = [-1] * (len(T_Q_AC_O) + 2) + AC_O_ANSWER[ac_o_index][1] + [-1]

                    _forward_AC_O_answer_start.append(_forward_AC_O_answer_start_temp)
                    _forward_AC_O_answer_end.append(_forward_AC_O_answer_end_temp)

                # (8)O->A
                for oa_index,oa_item in enumerate(O_A_QUERY):
                    T_Q_O_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in oa_item for word_ in word])
                    _forward_O_A_query.append(T_CLS+T_Q_O_A+T_SEP+T_text+T_SEP)
                    _O_A_query_mask_temp = [1] * len(_forward_O_A_query[0]) # 因为就1个，直接取0就行
                    _O_A_query_seg_temp = [0] * (len(_forward_O_A_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_O_A_query_mask.append(_O_A_query_mask_temp)
                    _forward_O_A_query_seg.append(_O_A_query_seg_temp)

                    _forward_O_A_answer_start_temp = [-1] * (len(T_Q_O_A)+2) + O_A_ANSWER[oa_index][0]+[-1]
                    _forward_O_A_answer_end_temp = [-1] * (len(T_Q_O_A)+2) + O_A_ANSWER[oa_index][1]+[-1]

                    _forward_O_A_answer_start.append(_forward_O_A_answer_start_temp)
                    _forward_O_A_answer_end.append(_forward_O_A_answer_end_temp)
                # (9)O->C

                # (10)O,A->C

                # (11)O,C->A

                for oc_a_index,oc_a_item in enumerate(O_A_QUERY):
                    T_Q_OC_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in oc_a_item for word_ in word])
                    _forward_OC_A_query.append(T_CLS+T_Q_OC_A+T_SEP+T_text+T_SEP)
                    _OC_A_query_mask_temp = [1] * len(_forward_OC_A_query[0]) # 因为就1个，直接取0就行
                    _OC_A_query_seg_temp = [0] * (len(_forward_OC_A_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_OC_A_query_mask.append(_OC_A_query_mask_temp)
                    _forward_OC_A_query_seg.append(_OC_A_query_seg_temp)

                    _forward_OC_A_answer_start_temp = [-1] * (len(T_Q_OC_A)+2) + OC_A_ANSWER[oc_a_index][0]+[-1]
                    _forward_OC_A_answer_end_temp = [-1] * (len(T_Q_OC_A)+2) + OC_A_ANSWER[oc_a_index][1]+[-1]

                    _forward_OC_A_answer_start.append(_forward_OC_A_answer_start_temp)
                    _forward_OC_A_answer_end.append(_forward_OC_A_answer_end_temp)

                # (12)A,O,C->P

                for AOC_P_index,AOC_P_item in enumerate(AOC_P_QUERY):
                    T_Q_AOC_P=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in AOC_P_item for word_ in word])
                    _forward_AOC_P_query.append(T_CLS+T_Q_AOC_P+T_SEP+T_text+T_SEP)
                    _AOC_P_query_mask_temp = [1] * len(_forward_AOC_P_query[0]) # 因为就1个，直接取0就行
                    _AOC_P_query_seg_temp = [0] * (len(_forward_AOC_P_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_AOC_P_query_mask.append(_AOC_P_query_mask_temp)
                    _forward_AOC_P_query_seg.append(_AOC_P_query_seg_temp)

                    _forward_AOC_P_answer_temp = AOC_P_ANSWER[AOC_P_index]

                    _forward_AOC_P_answer.append(_forward_AOC_P_answer_temp)

                # (13)C->A
                for C_A_index, C_A_item in enumerate(AOC_P_QUERY):
                    T_Q_C_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in C_A_item for word_ in word])
                    _forward_C_A_query.append(T_CLS+T_Q_C_A+T_SEP+T_text+T_SEP)
                    _C_A_query_mask_temp = [1] * len(_forward_C_A_query[0]) # 因为就1个，直接取0就行
                    _C_A_query_seg_temp = [0] * (len(_forward_C_A_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_C_A_query_mask.append(_C_A_query_mask_temp)
                    _forward_C_A_query_seg.append(_C_A_query_seg_temp)

                    _forward_C_A_answer_start_temp = [-1] * (len(T_Q_C_A)+2) + C_A_ANSWER[C_A_index][0]+[-1]
                    _forward_C_A_answer_end_temp = [-1] * (len(T_Q_C_A)+2) + C_A_ANSWER[C_A_index][1]+[-1]

                    _forward_C_A_answer_start.append(_forward_C_A_answer_start_temp)
                    _forward_C_A_answer_end.append(_forward_C_A_answer_end_temp)

                # (14)C->O
                for C_O_index, C_O_item in enumerate(AOC_P_QUERY):
                    T_Q_C_O=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in C_O_item for word_ in word])
                    _forward_C_O_query.append(T_CLS+T_Q_C_O+T_SEP+T_text+T_SEP)
                    _C_O_query_mask_temp = [1] * len(_forward_C_O_query[0]) # 因为就1个，直接取0就行
                    _C_O_query_seg_temp = [0] * (len(_forward_C_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_C_O_query_mask.append(_C_O_query_mask_temp)
                    _forward_C_O_query_seg.append(_C_O_query_seg_temp)

                    _forward_C_O_answer_start_temp = [-1] * (len(T_Q_C_O)+2) + C_O_ANSWER[C_O_index][0]+[-1]
                    _forward_C_O_answer_end_temp = [-1] * (len(T_Q_C_O)+2) + C_O_ANSWER[C_O_index][1]+[-1]

                    _forward_C_O_answer_start.append(_forward_C_O_answer_start_temp)
                    _forward_C_O_answer_end.append(_forward_C_O_answer_end_temp)

                # 统计一下目前最长的？
                # 也不知道当时我写这个干啥的哈哈哈
                print(max([len(leng) for leng in _forward_S_A_answer_start]))


                ####################
                # 这里要加个assert 判断一下各个问题和答案的格式是不是对应的
                ####################
                result = {
                'ID':qa_index,
                # （1）S->A
                '_forward_S_A_query':_forward_S_A_query,
                '_forward_S_A_query_mask':_forward_S_A_query_mask,
                '_forward_S_A_query_seg':_forward_S_A_query_seg,
                '_forward_S_A_answer_start':_forward_S_A_answer_start,
                '_forward_S_A_answer_end':_forward_S_A_answer_end,

                # （2）S->O
                '_forward_S_O_query':_forward_S_O_query,
                '_forward_S_O_query_mask':_forward_S_O_query_mask,
                '_forward_S_O_query_seg':_forward_S_O_query_seg,
                '_forward_S_O_answer_start':_forward_S_O_answer_start,
                '_forward_S_O_answer_end':_forward_S_O_answer_end,

                # （3）S->C   # C的暂时先保留，可能使用|C|轮问答也有可能用别的
                '_forward_S_C_query':_forward_S_C_query,
                '_forward_S_C_query_mask':_forward_S_C_query_mask,
                '_forward_S_C_query_seg':_forward_S_C_query_seg,

                '_forward_S_C_answer_start':_forward_S_C_answer_start,
                '_forward_S_C_answer_end':_forward_S_C_answer_end,

                # (4)A->O
                '_forward_A_O_query':_forward_A_O_query,
                '_forward_A_O_query_mask':_forward_A_O_query_mask,
                '_forward_A_O_query_seg':_forward_A_O_query_seg,
                '_forward_A_O_answer_start':_forward_A_O_answer_start,
                '_forward_A_O_answer_end':_forward_A_O_answer_end,

                # (5)A->C
                '_forward_A_C_query':_forward_A_C_query,
                '_forward_A_C_query_mask':_forward_A_C_query_mask,
                '_forward_A_C_query_seg':_forward_A_C_query_seg,

                '_forward_A_C_answer':_forward_A_C_answer,# 分类问题，并不需要start和end
                # (6)A,O->C
                '_forward_AO_C_query':_forward_AO_C_query,
                '_forward_AO_C_query_mask':_forward_AO_C_query_mask,
                '_forward_AO_C_query_seg':_forward_AO_C_query_seg,

                '_forward_AO_C_answer':_forward_AO_C_answer,# 分类问题，并不需要start和end


                # (7)A,C->O
                '_forward_AC_O_query':_forward_AC_O_query,
                '_forward_AC_O_query_mask':_forward_AC_O_query_mask,
                '_forward_AC_O_query_seg':_forward_AC_O_query_seg,
                '_forward_AC_O_answer_start':_forward_AC_O_answer_start,
                '_forward_AC_O_answer_end':_forward_AC_O_answer_end,

                # (8)O->A
                '_forward_O_A_query':_forward_O_A_query,
                '_forward_O_A_query_mask':_forward_O_A_query_mask,
                '_forward_O_A_query_seg':_forward_O_A_query_seg,

                '_forward_O_A_answer_start':_forward_O_A_answer_start,
                '_forward_O_A_answer_end':_forward_O_A_answer_end,

                # (9)O->C
                '_forward_O_C_query':_forward_O_C_query,
                '_forward_O_C_query_mask':_forward_O_C_query_mask,
                '_forward_O_C_query_seg':_forward_O_C_query_seg,

                '_forward_O_C_answer':_forward_O_C_answer,# 分类问题，并不需要start和end

                # (10)O,A->C
                '_forward_OA_C_query':_forward_OA_C_query,
                '_forward_OA_C_query_mask':_forward_OA_C_query_mask,
                '_forward_OA_C_query_seg':_forward_OA_C_query_seg,

                '_forward_OA_C_answer':_forward_OA_C_answer,# 分类问题，并不需要start和end

                # (11)O,C->A
                '_forward_OC_A_query':_forward_OC_A_query,
                '_forward_OC_A_query_mask':_forward_OC_A_query_mask,
                '_forward_OC_A_query_seg':_forward_OC_A_query_seg,

                '_forward_OC_A_answer_start':_forward_OC_A_answer_start,
                '_forward_OC_A_answer_end':_forward_OC_A_answer_end,

                # (12)A,O,C->P
                '_forward_AOC_P_query':_forward_AOC_P_query,
                '_forward_AOC_P_query_mask':_forward_AOC_P_query_mask,
                '_forward_AOC_P_query_seg':_forward_AOC_P_query_seg,

                '_forward_AOC_P_answer':_forward_AOC_P_answer,  # 分类问题，并不需要start和end

                # (13) C->A  #在这个的时候其实也得到C了
                '_forward_C_A_query':_forward_C_A_query,
                '_forward_C_A_query_mask':_forward_C_A_query_mask,
                '_forward_C_A_query_seg':_forward_C_A_query_seg,

                '_forward_C_A_answer_start':_forward_C_A_answer_start,
                '_forward_C_A_answer_end':_forward_C_A_answer_end,
                # (14) C->O
                '_forward_C_O_query':_forward_C_O_query,
                '_forward_C_O_query_mask':_forward_C_O_query_mask,
                '_forward_C_O_query_seg':_forward_C_O_query_seg,

                '_forward_C_O_answer_start':_forward_C_O_answer_start,
                '_forward_C_O_answer_end':_forward_C_O_answer_end,

                }
                self.QAs_r.append(result)
        else:
            pass
        print(1)

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.data_id_list)

    def __getitem__(self, index):
        return self.text_list,self.quad_list,self.QAs

class Collate():
    def __init__(self, opt):
        self.text_length_dynamic = opt.text_length_dynamic
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.word_length

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
        image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
        label = torch.LongTensor([b[2] for b in batch_data])
        text_translation_to_id = [torch.LongTensor(b[3]) for b in batch_data]
        image_augment = torch.FloatTensor([np.array(b[4]) for b in batch_data])

        data_length = [text.size(0) for text in text_to_id]
        data_translation_length = torch.LongTensor([text.size(0) for text in text_translation_to_id])

        max_length = max(data_length)
        if max_length < self.min_length:
            # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
            text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
            max_length = self.min_length

        max_translation_length = max(data_translation_length)
        if max_translation_length < self.min_length:
            # 这个地方随便选一个只要保证翻译的文本里面某一个大于设定的min_length就可以保证后续不会报错了
            text_translation_to_id[0] = torch.cat((text_translation_to_id[0], torch.LongTensor([0] * (self.min_length - text_translation_to_id[0].size(0)))))
            max_translation_length = self.min_length

        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
        text_translation_to_id = run_utils.pad_sequence(text_translation_to_id, batch_first=True, padding_value=0)

        bert_attention_mask = []
        text_image_mask = []
        for length in data_length:
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_length - length))
            bert_attention_mask.append(text_mask_cell[:])

            text_mask_cell.extend([1] * self.image_mask_num)
            text_image_mask.append(text_mask_cell[:])

        tran_bert_attention_mask = []
        tran_text_image_mask = []
        for length in data_translation_length:
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_translation_length - length))
            tran_bert_attention_mask.append(text_mask_cell[:])

            text_mask_cell.extend([1] * self.image_mask_num)
            tran_text_image_mask.append(text_mask_cell[:])

        temp_labels = [label - 0, label - 1, label - 2]
        target_labels = []
        for i in range(3):
            temp_target_labels = []
            for j in range(temp_labels[0].size(0)):
                if temp_labels[i][j] == 0:
                    temp_target_labels.append(j)
            target_labels.append(torch.LongTensor(temp_target_labels[:]))

        return text_to_id, torch.LongTensor(bert_attention_mask), image_origin, torch.LongTensor(text_image_mask), label, \
               text_translation_to_id, torch.LongTensor(tran_bert_attention_mask), image_augment, torch.LongTensor(tran_text_image_mask), target_labels


def get_resize(image_size):
    for i in range(20):
        if 2**i >= image_size:
            return 2**i
    return image_size


def data_process(opt,
                 data_path,
                 text_tokenizer,
                 data_type):

    dataset = SentenceDataset(opt, data_path, text_tokenizer, photo_path, transform_train if data_type == 1 else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), pin_memory=True if opt.cuda else False)
    return data_loader, dataset.__len__()

class OPTION():
    def __init__(self):
        self.bert_model_type='bert-base-uncased'
        #self.fuse_lr='bert-base'
# 测试数据处理相关内容

if __name__ == '__main__':
    temp_opt=OPTION()
    data_path='./data/rest16dev.json'
    data_type='rest'
    dataset_type='dev'
    wawa=SYZDataset(opt=temp_opt,data_path=data_path,data_type=data_type,dataset_type=dataset_type)