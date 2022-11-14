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

import DatasetCapsulation as Data1
import Data as Data2

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


def list_to_numpy(dic):
    _dict = {}
    for name, tensor in dic.items():
        _dict[name]=np.array(dic[name])
    return _dict




class SYZDataset(Dataset):
    # 四元组Dataset
    # 面向的是有属性缺失的数据集
    # 之后再弄一个三元组的dataset，主要的目的是复现之前的BMRC框架
    def __init__(self,
                 opt,
                 data_path,
                 data_type,
                 dataset_type,
                 task_type
                 ):
        """
        :param opt:  args得到的东西，详情见main函数
        :param data_path:  数据存储的位置  ./data
        :param data_type:  train，dev，还是test，用于构建相应的文件（夹）
        :param task_type:  ASTE、AOPE、ASQP等等
        """

        self.data_type = data_type         # ['laptop','rest16'] # for 4 yuanzu
        self.dataset_type = dataset_type  #'['train','dev','test']
        self.task_type = task_type        #['ASTE','AOPE','ASQP']
        # 读文件并保存
        if opt.bert_model_type.find('bert-base-uncased')!=-1:
            self.tokenizer=BertTokenizer.from_pretrained('../../bert/bert-base-uncased/vocab.txt')

        self.cat_to_id,self.id_to_cat=get_query_and_answer(data_type)


        self.data=[]
        if self.task_type=='ASQP':#os.path.exists('xxx/xxx/filename') is False:  # 如果没处理过，那就处理
            file_read = open(data_path+'/'+data_type+dataset_type+'_4yuanzu'+'.json', 'r', encoding='utf-8')
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

            self.standard=[]
            """
                        aspcets:[[a,b],[c,d],[e,f]]
                        opinions:[[g,h],[i,j],[k,q]]
                        as_op:[[a,b,g,h]
                        as_op_po:[[a,b,g,h,1],[c,d,i,j,0],[e,f,k,q,2]]
                        as_po:[[a,b,1],[c,d,0],[e,f,2]]
            """
            for data in self.quad_list:
                r={}
                a_o_p_target=[]
                a_target=[]
                o_target=[]
                a_o_target=[]
                a_p_target=[]
                for qua in data:
                    a_list=[int(d) for d in qua['aspect_index'].split(',')]
                    o_list=[int(d) for d in qua['opinion_index'].split(',')]
                    p_list=int(qua['polarity'])
                    a_target.append(a_list)
                    o_target.append(o_list)
                    a_o_target.append(a_list+o_list)
                    a_p_target.append(a_list+p_list)
                    a_o_p_target.append(a_list+o_list+p_list)
                r['aspcets'] = a_target
                r['opinions'] = o_target
                r['as_op'] = a_o_target
                r['as_po'] = a_p_target
                r['as_op_po'] = a_o_p_target
                self.standard.append(r)





                # 存储标注答案

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
                    l1=len(T_Q_A_O)
                    T_Q_A_O=T_CLS+T_Q_A_O+T_SEP+T_text+T_SEP
                    _forward_A_O_query.append(T_Q_A_O)
                    _A_O_query_mask_temp = [1] * len(T_Q_A_O) # 因为就1个，直接取0就行
                    _A_O_query_seg_temp = [0] * (len(T_Q_A_O) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_A_O_query_mask.append(_A_O_query_mask_temp)
                    _forward_A_O_query_seg.append(_A_O_query_seg_temp)

                    # 这里是不是会有问题？
                    _forward_A_O_answer_start_temp = [-1] * (l1+2) + A_O_ANSWER[ao_index][0]+[-1]
                    _forward_A_O_answer_end_temp = [-1] * (l1+2) + A_O_ANSWER[ao_index][1]+[-1]

                    _forward_A_O_answer_start.append(_forward_A_O_answer_start_temp)
                    _forward_A_O_answer_end.append(_forward_A_O_answer_end_temp)

                    assert len(T_Q_A_O)==len(_A_O_query_mask_temp)
                    assert len(T_Q_A_O)==len(_A_O_query_seg_temp)
                    assert len(T_Q_A_O)==len(_forward_A_O_answer_start_temp)
                    assert len(T_Q_A_O)==len(_forward_A_O_answer_end_temp)
                # 自己的填充
                A_O_max_len=max([len(l1) for l1 in _forward_A_O_query])
                for A_O_index,A_O_item in enumerate(_forward_A_O_query):
                    _forward_A_O_query[A_O_index] += [0] * (A_O_max_len - len(A_O_item))
                for A_O_index,A_O_item in enumerate(_forward_A_O_query_mask):
                    _forward_A_O_query_mask[A_O_index] += [0] * (A_O_max_len - len(A_O_item))
                for A_O_index,A_O_item in enumerate(_forward_A_O_query_seg):
                    _forward_A_O_query_seg[A_O_index] += [1] * (A_O_max_len - len(A_O_item))
                for A_O_index,A_O_item in enumerate(_forward_A_O_answer_start):
                    _forward_A_O_answer_start[A_O_index] += [-1] * (A_O_max_len - len(A_O_item))
                for A_O_index,A_O_item in enumerate(_forward_A_O_answer_end):
                    _forward_A_O_answer_end[A_O_index] += [-1] * (A_O_max_len - len(A_O_item))


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

                    _AC_O_query_mask_temp = [1] * len(_forward_AC_O_query[ac_o_index])  # 因为就1个，直接取0就行
                    _AC_O_query_seg_temp = [0] * (len(_forward_AC_O_query[ac_o_index]) - len(T_text) - 1) + [1] * (len(T_text) + 1)

                    _forward_AC_O_query_mask.append(_AC_O_query_mask_temp)
                    _forward_AC_O_query_seg.append(_AC_O_query_seg_temp)

                    _forward_AC_O_answer_start_temp = [-1] * (len(T_Q_AC_O) + 2) + AC_O_ANSWER[ac_o_index][0] + [-1]
                    _forward_AC_O_answer_end_temp = [-1] * (len(T_Q_AC_O) + 2) + AC_O_ANSWER[ac_o_index][1] + [-1]

                    _forward_AC_O_answer_start.append(_forward_AC_O_answer_start_temp)
                    _forward_AC_O_answer_end.append(_forward_AC_O_answer_end_temp)

                    T_Q_AC_O=T_CLS + T_Q_AC_O + T_SEP + T_text + T_SEP
                    assert len(T_Q_AC_O)==len(_AC_O_query_mask_temp)
                    assert len(T_Q_AC_O)==len(_AC_O_query_seg_temp)
                    assert len(T_Q_AC_O)==len(_forward_AC_O_answer_start_temp)
                    assert len(T_Q_AC_O)==len(_forward_AC_O_answer_end_temp)

                AC_O_max_len=max([len(l1) for l1 in _forward_AC_O_query])
                for AC_O_index,AC_O_item in enumerate(_forward_AC_O_query):
                    _forward_AC_O_query[AC_O_index] += [0] * (AC_O_max_len - len(AC_O_item))
                for AC_O_index,AC_O_item in enumerate(_forward_AC_O_query_mask):
                    _forward_AC_O_query_mask[AC_O_index] += [0] * (AC_O_max_len - len(AC_O_item))
                for AC_O_index,AC_O_item in enumerate(_forward_AC_O_query_seg):
                    _forward_AC_O_query_seg[AC_O_index] += [1] * (AC_O_max_len - len(AC_O_item))
                for AC_O_index,AC_O_item in enumerate(_forward_AC_O_answer_start):
                    _forward_AC_O_answer_start[AC_O_index] += [-1] * (AC_O_max_len - len(AC_O_item))
                for AC_O_index,AC_O_item in enumerate(_forward_AC_O_answer_end):
                    _forward_AC_O_answer_end[AC_O_index] += [-1] * (AC_O_max_len - len(AC_O_item))


                # (8)O->A
                for oa_index,oa_item in enumerate(O_A_QUERY):
                    T_Q_O_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in oa_item for word_ in word])
                    _forward_O_A_query.append(T_CLS+T_Q_O_A+T_SEP+T_text+T_SEP)
                    _O_A_query_mask_temp = [1] * len(_forward_O_A_query[oa_index]) # 因为就1个，直接取0就行
                    _O_A_query_seg_temp = [0] * (len(_forward_O_A_query[oa_index]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_O_A_query_mask.append(_O_A_query_mask_temp)
                    _forward_O_A_query_seg.append(_O_A_query_seg_temp)

                    _forward_O_A_answer_start_temp = [-1] * (len(T_Q_O_A)+2) + O_A_ANSWER[oa_index][0]+[-1]
                    _forward_O_A_answer_end_temp = [-1] * (len(T_Q_O_A)+2) + O_A_ANSWER[oa_index][1]+[-1]

                    _forward_O_A_answer_start.append(_forward_O_A_answer_start_temp)
                    _forward_O_A_answer_end.append(_forward_O_A_answer_end_temp)
                    T_Q_O_A = T_CLS + T_Q_O_A + T_SEP + T_text + T_SEP
                    assert len(T_Q_O_A)==len(_O_A_query_mask_temp)
                    assert len(T_Q_O_A)==len(_O_A_query_seg_temp)
                    assert len(T_Q_O_A)==len(_forward_O_A_answer_start_temp)
                    assert len(T_Q_O_A)==len(_forward_O_A_answer_end_temp)

                O_A_max_len=max([len(l1) for l1 in _forward_O_A_query])
                for O_A_index,O_A_item in enumerate(_forward_O_A_query):
                    _forward_O_A_query[O_A_index] += [0] * (O_A_max_len - len(O_A_item))
                for O_A_index,O_A_item in enumerate(_forward_O_A_query_mask):
                    _forward_O_A_query_mask[O_A_index] += [0] * (O_A_max_len - len(O_A_item))
                for O_A_index,O_A_item in enumerate(_forward_O_A_query_seg):
                    _forward_O_A_query_seg[O_A_index] += [1] * (O_A_max_len - len(O_A_item))
                for O_A_index,O_A_item in enumerate(_forward_O_A_answer_start):
                    _forward_O_A_answer_start[O_A_index] += [-1] * (O_A_max_len - len(O_A_item))
                for O_A_index,O_A_item in enumerate(_forward_O_A_answer_end):
                    _forward_O_A_answer_end[O_A_index] += [-1] * (O_A_max_len - len(O_A_item))

                # (9)O->C

                # (10)O,A->C

                # (11)O,C->A

                for oc_a_index,oc_a_item in enumerate(O_A_QUERY):
                    T_Q_OC_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in oc_a_item for word_ in word])
                    _forward_OC_A_query.append(T_CLS+T_Q_OC_A+T_SEP+T_text+T_SEP)
                    _OC_A_query_mask_temp = [1] * len(_forward_OC_A_query[oc_a_index]) # 因为就1个，直接取0就行
                    _OC_A_query_seg_temp = [0] * (len(_forward_OC_A_query[oc_a_index]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_OC_A_query_mask.append(_OC_A_query_mask_temp)
                    _forward_OC_A_query_seg.append(_OC_A_query_seg_temp)

                    _forward_OC_A_answer_start_temp = [-1] * (len(T_Q_OC_A)+2) + OC_A_ANSWER[oc_a_index][0]+[-1]
                    _forward_OC_A_answer_end_temp = [-1] * (len(T_Q_OC_A)+2) + OC_A_ANSWER[oc_a_index][1]+[-1]

                    _forward_OC_A_answer_start.append(_forward_OC_A_answer_start_temp)
                    _forward_OC_A_answer_end.append(_forward_OC_A_answer_end_temp)

                    T_Q_OC_A = T_CLS + T_Q_OC_A + T_SEP + T_text + T_SEP
                    assert len(T_Q_OC_A)==len(_OC_A_query_mask_temp)
                    assert len(T_Q_OC_A)==len(_OC_A_query_seg_temp)
                    assert len(T_Q_OC_A)==len(_forward_OC_A_answer_start_temp)
                    assert len(T_Q_OC_A)==len(_forward_OC_A_answer_end_temp)
                OC_A_max_len=max([len(l1) for l1 in _forward_OC_A_query])
                for OC_A_index,OC_A_item in enumerate(_forward_OC_A_query):
                    _forward_OC_A_query[OC_A_index] += [0] * (OC_A_max_len - len(OC_A_item))
                for OC_A_index,OC_A_item in enumerate(_forward_OC_A_query_mask):
                    _forward_OC_A_query_mask[OC_A_index] += [0] * (OC_A_max_len - len(OC_A_item))
                for OC_A_index,OC_A_item in enumerate(_forward_OC_A_query_seg):
                    _forward_OC_A_query_seg[OC_A_index] += [1] * (OC_A_max_len - len(OC_A_item))
                for OC_A_index,OC_A_item in enumerate(_forward_OC_A_answer_start):
                    _forward_OC_A_answer_start[OC_A_index] += [-1] * (OC_A_max_len - len(OC_A_item))
                for OC_A_index,OC_A_item in enumerate(_forward_OC_A_answer_end):
                    _forward_OC_A_answer_end[OC_A_index] += [-1] * (OC_A_max_len - len(OC_A_item))

                # (12)A,O,C->P
                for AOC_P_index,AOC_P_item in enumerate(AOC_P_QUERY):
                    T_Q_AOC_P=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in AOC_P_item for word_ in word])
                    _forward_AOC_P_query.append(T_CLS+T_Q_AOC_P+T_SEP+T_text+T_SEP)
                    _AOC_P_query_mask_temp = [1] * len(_forward_AOC_P_query[AOC_P_index]) # 因为就1个，直接取0就行
                    _AOC_P_query_seg_temp = [0] * (len(_forward_AOC_P_query[AOC_P_index]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_AOC_P_query_mask.append(_AOC_P_query_mask_temp)
                    _forward_AOC_P_query_seg.append(_AOC_P_query_seg_temp)

                    _forward_AOC_P_answer_temp = AOC_P_ANSWER[AOC_P_index]

                    _forward_AOC_P_answer.append(_forward_AOC_P_answer_temp)

                    T_Q_AOC_P = T_CLS + T_Q_AOC_P + T_SEP + T_text + T_SEP
                    assert len(T_Q_AOC_P)==len(_AOC_P_query_mask_temp)
                    assert len(T_Q_AOC_P)==len(_AOC_P_query_seg_temp)
                    # assert len(T_Q_AOC_P)==len(_forward_AOC_P_answer_temp)

                AOC_P_max_len = max([len(l1) for l1 in _forward_AOC_P_query])
                for AOC_P_index, AOC_P_item in enumerate(_forward_AOC_P_query):
                    _forward_AOC_P_query[AOC_P_index] += [0] * (AOC_P_max_len - len(AOC_P_item))
                for AOC_P_index, AOC_P_item in enumerate(_forward_AOC_P_query_mask):
                    _forward_AOC_P_query_mask[AOC_P_index] += [0] * (AOC_P_max_len - len(AOC_P_item))
                for AOC_P_index, AOC_P_item in enumerate(_forward_AOC_P_query_seg):
                    _forward_AOC_P_query_seg[AOC_P_index] += [1] * (AOC_P_max_len - len(AOC_P_item))



                # (13)C->A
                for C_A_index, C_A_item in enumerate(AOC_P_QUERY):
                    T_Q_C_A=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in C_A_item for word_ in word])
                    _forward_C_A_query.append(T_CLS+T_Q_C_A+T_SEP+T_text+T_SEP)
                    _C_A_query_mask_temp = [1] * len(_forward_C_A_query[C_A_index]) # 因为就1个，直接取0就行
                    _C_A_query_seg_temp = [0] * (len(_forward_C_A_query[C_A_index]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_C_A_query_mask.append(_C_A_query_mask_temp)
                    _forward_C_A_query_seg.append(_C_A_query_seg_temp)

                    _forward_C_A_answer_start_temp = [-1] * (len(T_Q_C_A)+2) + C_A_ANSWER[C_A_index][0]+[-1]
                    _forward_C_A_answer_end_temp = [-1] * (len(T_Q_C_A)+2) + C_A_ANSWER[C_A_index][1]+[-1]

                    _forward_C_A_answer_start.append(_forward_C_A_answer_start_temp)
                    _forward_C_A_answer_end.append(_forward_C_A_answer_end_temp)
                    T_Q_C_A = T_CLS + T_Q_C_A + T_SEP + T_text + T_SEP
                    assert len(T_Q_C_A)==len(_C_A_query_mask_temp)
                    assert len(T_Q_C_A)==len(_C_A_query_seg_temp)
                    assert len(T_Q_C_A)==len(_forward_C_A_answer_start_temp)
                    assert len(T_Q_C_A)==len(_forward_C_A_answer_end_temp)

                C_A_max_len=max([len(l1) for l1 in _forward_C_A_query])
                for C_A_index,C_A_item in enumerate(_forward_C_A_query):
                    _forward_C_A_query[C_A_index] += [0] * (C_A_max_len - len(C_A_item))
                for C_A_index,C_A_item in enumerate(_forward_C_A_query_mask):
                    _forward_C_A_query_mask[C_A_index] += [0] * (C_A_max_len - len(C_A_item))
                for C_A_index,C_A_item in enumerate(_forward_C_A_query_seg):
                    _forward_C_A_query_seg[C_A_index] += [1] * (C_A_max_len - len(C_A_item))
                for C_A_index,C_A_item in enumerate(_forward_C_A_answer_start):
                    _forward_C_A_answer_start[C_A_index] += [-1] * (C_A_max_len - len(C_A_item))
                for C_A_index,C_A_item in enumerate(_forward_C_A_answer_end):
                    _forward_C_A_answer_end[C_A_index] += [-1] * (C_A_max_len - len(C_A_item))

                # (14)C->O
                for C_O_index, C_O_item in enumerate(AOC_P_QUERY):
                    T_Q_C_O=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in C_O_item for word_ in word])
                    _forward_C_O_query.append(T_CLS+T_Q_C_O+T_SEP+T_text+T_SEP)
                    _C_O_query_mask_temp = [1] * len(_forward_C_O_query[C_O_index]) # 因为就1个，直接取0就行
                    _C_O_query_seg_temp = [0] * (len(_forward_C_O_query[C_O_index]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_C_O_query_mask.append(_C_O_query_mask_temp)
                    _forward_C_O_query_seg.append(_C_O_query_seg_temp)

                    _forward_C_O_answer_start_temp = [-1] * (len(T_Q_C_O)+2) + C_O_ANSWER[C_O_index][0]+[-1]
                    _forward_C_O_answer_end_temp = [-1] * (len(T_Q_C_O)+2) + C_O_ANSWER[C_O_index][1]+[-1]

                    _forward_C_O_answer_start.append(_forward_C_O_answer_start_temp)
                    _forward_C_O_answer_end.append(_forward_C_O_answer_end_temp)
                    T_Q_C_O = T_CLS + T_Q_C_O + T_SEP + T_text + T_SEP
                    assert len(T_Q_C_O)==len(_C_O_query_mask_temp)
                    assert len(T_Q_C_O)==len(_C_O_query_seg_temp)
                    assert len(T_Q_C_O)==len(_forward_C_O_answer_start_temp)
                    assert len(T_Q_C_O)==len(_forward_C_O_answer_end_temp)

                C_O_max_len=max([len(l1) for l1 in _forward_C_O_query])
                for C_O_index,C_O_item in enumerate(_forward_C_O_query):
                    _forward_C_O_query[C_O_index] += [0] * (C_O_max_len - len(C_O_item))
                for C_O_index,C_O_item in enumerate(_forward_C_O_query_mask):
                    _forward_C_O_query_mask[C_O_index] += [0] * (C_O_max_len - len(C_O_item))
                for C_O_index,C_O_item in enumerate(_forward_C_O_query_seg):
                    _forward_C_O_query_seg[C_O_index] += [1] * (C_O_max_len - len(C_O_item))
                for C_O_index,C_O_item in enumerate(_forward_C_O_answer_start):
                    _forward_C_O_answer_start[C_O_index] += [-1] * (C_O_max_len - len(C_O_item))
                for C_O_index,C_O_item in enumerate(_forward_C_O_answer_end):
                    _forward_C_O_answer_end[C_O_index] += [-1] * (C_O_max_len - len(C_O_item))
                # 统计一下目前最长的？
                # 也不知道当时我写这个干啥的哈哈哈
                # print(max([len(leng) for leng in _forward_S_A_answer_start]))


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
                #result=list_to_numpy(result)
                self.QAs_r.append(result)

        elif self.task_type=='ASTE':
            file_read = open(data_path + '/' + data_type + dataset_type + '_3yuanzu' + '.json', 'r', encoding='utf-8')
            file_content = json.load(file_read)
            file_read.close()
            self.data_id_list = []  # 保存id，方便找错
            self.text_list = []  # 保存text，方便debug
            self.tri_list = []  # 保存四元组
            self.QAs = []  # 保存QA对，实际上处理这个就行了
            self.QAs_r = []
            self.max_length = 0
            for data in file_content:
                self.data_id_list.append(data['ID'])
                self.text_list.append(data['text'])
                self.tri_list.append(data['triplet'])
                self.QAs.append(data['QAs'])

            # gold truth
            self.standard = []
            for triplet in self.tri_list:
                aspect_temp = []
                opinion_temp = []
                pair_temp = []
                triplet_temp = []
                asp_pol_temp = []
                for temp_t in triplet:
                    triplet_temp.append([temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1], temp_t[2]])
                    ap = [temp_t[0][0], temp_t[0][-1], temp_t[2]]
                    if ap not in asp_pol_temp:
                        asp_pol_temp.append(ap)
                    a = [temp_t[0][0], temp_t[0][-1]]
                    if a not in aspect_temp:
                        aspect_temp.append(a)
                    o = [temp_t[1][0], temp_t[1][-1]]
                    if o not in opinion_temp:
                        opinion_temp.append(o)
                    p = [temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1]]
                    if p not in pair_temp:
                        pair_temp.append(p)
                """
                            aspects:[[a,b],[c,d],[e,f]]
                            opinions:[[g,h],[i,j],[k,q]]
                            as_op:[[a,b,g,h]
                            as_op_po:[[a,b,g,h,1],[c,d,i,j,0],[e,f,k,q,2]]
                            as_po:[[a,b,1],[c,d,0],[e,f,2]]
                """
                self.standard.append(
                        {'aspects': aspect_temp, 'opinions': opinion_temp, 'as_op': pair_temp,
                         'as_po': asp_pol_temp, 'as_op_po': triplet_temp})

            for qa_index, qa in enumerate(self.QAs):

                text = self.text_list[qa_index]  # 原文
                # 原数据
                S_A_QUERY = qa['S_A_QUERY']
                S_A_ANSWER = qa['S_A_ANSWER']

                A_O_QUERY = qa['A_O_QUERY']
                A_O_ANSWER = qa['A_O_ANSWER']

                S_O_QUERY = qa['S_O_QUERY']
                S_O_ANSWER = qa['S_O_ANSWER']

                O_A_QUERY = qa['O_A_QUERY']
                O_A_ANSWER = qa['O_A_ANSWER']

                AO_P_QUERY=qa['AO_P_QUERY']
                AO_P_ANSWER=qa['AO_P_ANSWER']

                # 保存一些常用的,当然有些tokenizer直接加就行，这个因地制宜（其实拿hf的bert那个接口能更方便的哈哈）
                T_CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
                T_SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
                T_text = self.tokenizer.convert_tokens_to_ids([word.lower() for word in text])

                ###############
                # 一些初始化
                ###############
                ## 其中->A 或->O的为抽取式任务，构建方法相似；->C或-S为分类任务（对于本次测试来说），构建方法相似
                # （1）S->A
                _forward_S_A_query = []
                _forward_S_A_query_mask = []
                _forward_S_A_query_seg = []
                _forward_S_A_answer_start = []
                _forward_S_A_answer_end = []

                # （2）S->O
                _forward_S_O_query = []
                _forward_S_O_query_mask = []
                _forward_S_O_query_seg = []
                _forward_S_O_answer_start = []
                _forward_S_O_answer_end = []

                # (4)A->O
                _forward_A_O_query = []
                _forward_A_O_query_mask = []
                _forward_A_O_query_seg = []
                _forward_A_O_answer_start = []
                _forward_A_O_answer_end = []

                # (8)O->A
                _forward_O_A_query = []
                _forward_O_A_query_mask = []
                _forward_O_A_query_seg = []

                _forward_O_A_answer_start = []
                _forward_O_A_answer_end = []


                # (12)A,O->P
                _forward_AO_P_query = []
                _forward_AO_P_query_mask = []
                _forward_AO_P_query_seg = []

                _forward_AO_P_answer = []  # 分类问题，并不需要start和end


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

                T_Q_S_A = self.tokenizer.convert_tokens_to_ids([word_.lower() for word in S_A_QUERY for word_ in word])
                _forward_S_A_query=T_CLS + T_Q_S_A + T_SEP + T_text
                _S_A_query_mask_temp = [1] * len(_forward_S_A_query)  # 因为就1个，直接取0就行
                _S_A_query_seg_temp = [0] * (len(_forward_S_A_query) - len(T_text) ) + [1] * (len(T_text))
                _forward_S_A_query_mask=_S_A_query_mask_temp
                _forward_S_A_query_seg=_S_A_query_seg_temp

                _forward_S_A_answer_start_temp = [-1] * (len(T_Q_S_A) + 2) + S_A_ANSWER[0][0]
                _forward_S_A_answer_end_temp = [-1] * (len(T_Q_S_A) + 2) + S_A_ANSWER[0][1]

                _forward_S_A_answer_start=_forward_S_A_answer_start_temp
                _forward_S_A_answer_end=_forward_S_A_answer_end_temp
                T_Q_S_A=T_CLS + T_Q_S_A + T_SEP + T_text
                assert len(T_Q_S_A) == len(_S_A_query_mask_temp)
                assert len(T_Q_S_A) == len(_S_A_query_seg_temp)
                assert len(T_Q_S_A) == len(_forward_S_A_answer_start_temp)
                assert len(T_Q_S_A) == len(_forward_S_A_answer_end_temp)
                # （2）S->O
                T_Q_S_O = self.tokenizer.convert_tokens_to_ids([word_.lower() for word in S_O_QUERY for word_ in word])
                _forward_S_O_query= T_CLS + T_Q_S_O + T_SEP + T_text   # build_inputs_with_special_tokens等效
                _S_O_query_mask_temp = [1] * len(_forward_S_O_query)  # 因为就1个，直接取0就行
                _S_O_query_seg_temp = [0] * (len(_forward_S_O_query) - len(T_text) ) + [1] * (len(T_text))
                _forward_S_O_query_mask=_S_O_query_mask_temp
                _forward_S_O_query_seg=_S_O_query_seg_temp

                _forward_S_O_answer_start_temp = [-1] * (len(T_Q_S_O) + 2) + S_O_ANSWER[0][0]
                _forward_S_O_answer_end_temp = [-1] * (len(T_Q_S_O) + 2) + S_O_ANSWER[0][1]

                _forward_S_O_answer_start=_forward_S_O_answer_start_temp
                _forward_S_O_answer_end=_forward_S_O_answer_end_temp

                T_Q_S_O=T_CLS + T_Q_S_O + T_SEP + T_text
                assert len(T_Q_S_O) == len(_forward_S_O_query_mask)
                assert len(T_Q_S_O) == len(_forward_S_O_query_seg)
                assert len(T_Q_S_O) == len(_forward_S_O_answer_start_temp)
                assert len(T_Q_S_O) == len(_forward_S_O_answer_end_temp)
                # (4)A->O
                # 好像不需要考虑多组问题，A_O_QUERY?
                for ao_index, ao_item in enumerate(A_O_QUERY):
                    T_Q_A_O = self.tokenizer.convert_tokens_to_ids(
                        [word_.lower() for word in [ao_item] for word_ in word])
                    l1 = len(T_Q_A_O)
                    T_Q_A_O = T_CLS + T_Q_A_O + T_SEP + T_text
                    _forward_A_O_query.append(T_Q_A_O)
                    _A_O_query_mask_temp = [1] * len(T_Q_A_O)  # 因为就1个，直接取0就行
                    _A_O_query_seg_temp = [0] * (len(T_Q_A_O) - len(T_text) ) + [1] * (len(T_text) )
                    _forward_A_O_query_mask.append(_A_O_query_mask_temp)
                    _forward_A_O_query_seg.append(_A_O_query_seg_temp)

                    # 这里是不是会有问题？
                    _forward_A_O_answer_start_temp = [-1] * (l1 + 2) + A_O_ANSWER[ao_index][0]
                    _forward_A_O_answer_end_temp = [-1] * (l1 + 2) + A_O_ANSWER[ao_index][1]

                    _forward_A_O_answer_start.append(_forward_A_O_answer_start_temp)
                    _forward_A_O_answer_end.append(_forward_A_O_answer_end_temp)

                    assert len(T_Q_A_O) == len(_A_O_query_mask_temp)
                    assert len(T_Q_A_O) == len(_A_O_query_seg_temp)
                    assert len(T_Q_A_O) == len(_forward_A_O_answer_start_temp)
                    assert len(T_Q_A_O) == len(_forward_A_O_answer_end_temp)
                # 自己的填充
                A_O_max_len = max([len(l1) for l1 in _forward_A_O_query])
                for A_O_index, A_O_item in enumerate(_forward_A_O_query):
                    _forward_A_O_query[A_O_index] += [0] * (A_O_max_len - len(A_O_item))
                for A_O_index, A_O_item in enumerate(_forward_A_O_query_mask):
                    _forward_A_O_query_mask[A_O_index] += [0] * (A_O_max_len - len(A_O_item))
                for A_O_index, A_O_item in enumerate(_forward_A_O_query_seg):
                    _forward_A_O_query_seg[A_O_index] += [1] * (A_O_max_len - len(A_O_item))
                for A_O_index, A_O_item in enumerate(_forward_A_O_answer_start):
                    _forward_A_O_answer_start[A_O_index] += [-1] * (A_O_max_len - len(A_O_item))
                for A_O_index, A_O_item in enumerate(_forward_A_O_answer_end):
                    _forward_A_O_answer_end[A_O_index] += [-1] * (A_O_max_len - len(A_O_item))


                # (8)O->A
                for oa_index, oa_item in enumerate(O_A_QUERY):
                    T_Q_O_A = self.tokenizer.convert_tokens_to_ids(
                        [word_.lower() for word in oa_item for word_ in word])
                    _forward_O_A_query.append(T_CLS + T_Q_O_A + T_SEP + T_text)
                    _O_A_query_mask_temp = [1] * len(_forward_O_A_query[oa_index])  # 因为就1个，直接取0就行
                    _O_A_query_seg_temp = [0] * (len(_forward_O_A_query[oa_index]) - len(T_text) ) + [1] * (
                                len(T_text) )
                    _forward_O_A_query_mask.append(_O_A_query_mask_temp)
                    _forward_O_A_query_seg.append(_O_A_query_seg_temp)

                    _forward_O_A_answer_start_temp = [-1] * (len(T_Q_O_A) + 2) + O_A_ANSWER[oa_index][0]
                    _forward_O_A_answer_end_temp = [-1] * (len(T_Q_O_A) + 2) + O_A_ANSWER[oa_index][1]

                    _forward_O_A_answer_start.append(_forward_O_A_answer_start_temp)
                    _forward_O_A_answer_end.append(_forward_O_A_answer_end_temp)
                    T_Q_O_A = T_CLS + T_Q_O_A + T_SEP + T_text
                    assert len(T_Q_O_A) == len(_O_A_query_mask_temp)
                    assert len(T_Q_O_A) == len(_O_A_query_seg_temp)
                    assert len(T_Q_O_A) == len(_forward_O_A_answer_start_temp)
                    assert len(T_Q_O_A) == len(_forward_O_A_answer_end_temp)

                O_A_max_len = max([len(l1) for l1 in _forward_O_A_query])
                for O_A_index, O_A_item in enumerate(_forward_O_A_query):
                    _forward_O_A_query[O_A_index] += [0] * (O_A_max_len - len(O_A_item))
                for O_A_index, O_A_item in enumerate(_forward_O_A_query_mask):
                    _forward_O_A_query_mask[O_A_index] += [0] * (O_A_max_len - len(O_A_item))
                for O_A_index, O_A_item in enumerate(_forward_O_A_query_seg):
                    _forward_O_A_query_seg[O_A_index] += [1] * (O_A_max_len - len(O_A_item))
                for O_A_index, O_A_item in enumerate(_forward_O_A_answer_start):
                    _forward_O_A_answer_start[O_A_index] += [-1] * (O_A_max_len - len(O_A_item))
                for O_A_index, O_A_item in enumerate(_forward_O_A_answer_end):
                    _forward_O_A_answer_end[O_A_index] += [-1] * (O_A_max_len - len(O_A_item))


                # (12)A,O->P
                for AO_P_index, AO_P_item in enumerate(AO_P_QUERY):
                    T_Q_AO_P = self.tokenizer.convert_tokens_to_ids(
                        [word_.lower() for word in AO_P_item for word_ in word])
                    _forward_AO_P_query.append(T_CLS + T_Q_AO_P + T_SEP + T_text + T_SEP)
                    _AO_P_query_mask_temp = [1] * len(_forward_AO_P_query[AO_P_index])  # 因为就1个，直接取0就行
                    _AO_P_query_seg_temp = [0] * (len(_forward_AO_P_query[AO_P_index]) - len(T_text) ) + [1] * (
                                len(T_text) )
                    _forward_AO_P_query_mask.append(_AO_P_query_mask_temp)
                    _forward_AO_P_query_seg.append(_AO_P_query_seg_temp)

                    _forward_AO_P_answer_temp = AO_P_ANSWER[AO_P_index]

                    _forward_AO_P_answer.append(_forward_AO_P_answer_temp)

                    T_Q_AO_P = T_CLS + T_Q_AO_P + T_SEP + T_text + T_SEP
                    assert len(T_Q_AO_P) == len(_AO_P_query_mask_temp)
                    assert len(T_Q_AO_P) == len(_AO_P_query_seg_temp)
                    # assert len(T_Q_AO_P)==len(_forward_AO_P_answer_temp)

                AO_P_max_len = max([len(l1) for l1 in _forward_AO_P_query])
                for AO_P_index, AO_P_item in enumerate(_forward_AO_P_query):
                    _forward_AO_P_query[AO_P_index] += [0] * (AO_P_max_len - len(AO_P_item))
                for AO_P_index, AO_P_item in enumerate(_forward_AO_P_query_mask):
                    _forward_AO_P_query_mask[AO_P_index] += [0] * (AO_P_max_len - len(AO_P_item))
                for AO_P_index, AO_P_item in enumerate(_forward_AO_P_query_seg):
                    _forward_AO_P_query_seg[AO_P_index] += [1] * (AO_P_max_len - len(AO_P_item))


                # 统计一下目前最长的？
                # 也不知道当时我写这个干啥的哈哈哈
                # print(max([len(leng) for leng in _forward_S_A_answer_start]))

                ####################
                # 这里要加个assert 判断一下各个问题和答案的格式是不是对应的
                ####################
                result = {
                    'ID': qa_index,
                    # （1）S->A
                    '_forward_S_A_query': _forward_S_A_query,
                    '_forward_S_A_query_mask': _forward_S_A_query_mask,
                    '_forward_S_A_query_seg': _forward_S_A_query_seg,
                    '_forward_S_A_answer_start': _forward_S_A_answer_start,
                    '_forward_S_A_answer_end': _forward_S_A_answer_end,

                    # （2）S->O
                    '_forward_S_O_query': _forward_S_O_query,
                    '_forward_S_O_query_mask': _forward_S_O_query_mask,
                    '_forward_S_O_query_seg': _forward_S_O_query_seg,
                    '_forward_S_O_answer_start': _forward_S_O_answer_start,
                    '_forward_S_O_answer_end': _forward_S_O_answer_end,

                    # (4)A->O
                    '_forward_A_O_query': _forward_A_O_query,
                    '_forward_A_O_query_mask': _forward_A_O_query_mask,
                    '_forward_A_O_query_seg': _forward_A_O_query_seg,
                    '_forward_A_O_answer_start': _forward_A_O_answer_start,
                    '_forward_A_O_answer_end': _forward_A_O_answer_end,

                    # (8)O->A
                    '_forward_O_A_query': _forward_O_A_query,
                    '_forward_O_A_query_mask': _forward_O_A_query_mask,
                    '_forward_O_A_query_seg': _forward_O_A_query_seg,
                    '_forward_O_A_answer_start': _forward_O_A_answer_start,
                    '_forward_O_A_answer_end': _forward_O_A_answer_end,

                    # (12)A,O->P
                    '_forward_AO_P_query': _forward_AO_P_query,
                    '_forward_AO_P_query_mask': _forward_AO_P_query_mask,
                    '_forward_AO_P_query_seg': _forward_AO_P_query_seg,
                    '_forward_AO_P_answer': _forward_AO_P_answer,  # 分类问题，并不需要start和end
             }
                # result=list_to_numpy(result)
                self.QAs_r.append(result)
        #print(11)

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        return self.QAs_r[index]


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._forward_asp_query = pre_data['_forward_asp_query']
        self._forward_opi_query = pre_data['_forward_opi_query']
        self._forward_asp_answer_start = pre_data['_forward_asp_answer_start']
        self._forward_asp_answer_end = pre_data['_forward_asp_answer_end']
        self._forward_opi_answer_start = pre_data['_forward_opi_answer_start']
        self._forward_opi_answer_end = pre_data['_forward_opi_answer_end']
        self._forward_asp_query_mask = pre_data['_forward_asp_query_mask']
        self._forward_opi_query_mask = pre_data['_forward_opi_query_mask']
        self._forward_asp_query_seg = pre_data['_forward_asp_query_seg']
        self._forward_opi_query_seg = pre_data['_forward_opi_query_seg']

        self._backward_asp_query = pre_data['_backward_asp_query']
        self._backward_opi_query = pre_data['_backward_opi_query']
        self._backward_asp_answer_start = pre_data['_backward_asp_answer_start']
        self._backward_asp_answer_end = pre_data['_backward_asp_answer_end']
        self._backward_opi_answer_start = pre_data['_backward_opi_answer_start']
        self._backward_opi_answer_end = pre_data['_backward_opi_answer_end']
        self._backward_asp_query_mask = pre_data[
            '_backward_asp_query_mask']
        self._backward_opi_query_mask = pre_data[
            '_backward_opi_query_mask']
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']

        self._sentiment_query = pre_data['_sentiment_query']
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']

class OPTION():
    def __init__(self):
        self.bert_model_type='bert-base-uncased'
        self.acc_batch_size=1
        self.cuda=True
        self.work_nums=1
# 测试数据处理相关内容

if __name__ == '__main__':
    temp_opt=OPTION()
    data_path='./data'
    data_type='14lap'
    dataset_type='dev'
    task_type='ASTE'

    total_data = torch.load(data_path+'/14lap.pt')
    train_data = total_data['train']
    dev_data = total_data['dev']
    test_data = total_data['test']
    #dev_standard = standard_data['dev']
    #test_standard = standard_data['test']


    wawa=SYZDataset(opt=temp_opt,data_path=data_path,data_type=data_type,dataset_type=dataset_type,task_type=task_type)


    batch_generator = Data1.generate_batches(dataset=wawa, batch_size=1,gpu=False)

    train_dataset = Data2.ReviewDataset(train_data, dev_data, test_data, 'train')
    dev_dataset = Data2.ReviewDataset(train_data, dev_data, test_data, 'dev')
    test_dataset = Data2.ReviewDataset(train_data, dev_data, test_data, 'test')
    batch_generator_2 = Data2.generate_fi_batches(dataset=train_dataset, batch_size=1,
                                               ifgpu=False)


    for item1,item2 in zip(batch_generator,batch_generator_2):

        S_A_query = item1['_forward_S_A_query']
        S_A_query_mask = item1['_forward_S_A_query_mask']
        S_A_query_seg = item1['_forward_S_A_query_seg']
        S_A_answer_start_2 = item1['_forward_S_A_answer_start']
        S_A_answer_end_2 = item1['_forward_S_A_answer_end']


        S_A_query_2 = item2['forward_asp_query']
        S_A_query_mask_2 = item2['forward_asp_query_mask']
        S_A_query_seg_2 = item2['forward_asp_query_seg']
        S_A_answer_start_2 = item2['forward_asp_answer_start']
        S_A_answer_end_2= item2['forward_asp_answer_end']




        print(item1)
        print(item2)
        pass

        #print(item)
        #print(torch.from_numpy(i))