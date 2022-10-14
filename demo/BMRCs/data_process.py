# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-12
# @Do: 对BMRC相关数据的读取与处理

import os
import json
import copy


import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from transformers import RobertaTokenizer



# 四元组Dataset
class SYZDataset(Dataset):
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


        if True:#os.path.exists('xxx/xxx/filename') is False:  # 如果没处理过，那就处理
            file_read = open(data_path, 'r', encoding='utf-8')
            file_content = json.load(file_read)
            file_read.close()
            self.data_id_list = [] # 保存id，方便找错
            self.text_list = []    # 保存text，方便debug
            self.quad_list=[]      # 保存四元组
            self.QAs = []          # 保存QA对，实际上处理这个就行了
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

                AOC_S_QUERY=qa['AOC_S_QUERY']
                AOC_S_ANSWER=qa['AOC_S_ANSWER']

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

                # 保存一些常用的,当然有些tokenizer直接加就行，这个因地制宜吧
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

                # （3）S->C
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

                # (6)A,O->C

                # (7)A,C->O
                _forward_A_O_query=[]
                _forward_A_O_query_mask=[]
                _forward_A_O_query_seg=[]
                _forward_A_O_answer_start=[]
                _forward_A_O_answer_end=[]

                # (8)O->A
                _forward_A_O_query=[]
                _forward_A_O_query_mask=[]
                _forward_A_O_query_seg=[]
                _forward_A_O_answer_start=[]
                _forward_A_O_answer_end=[]

                # (9)O->C

                # (10)O,A->C

                # (11)O,C->A
                _forward_A_O_query=[]
                _forward_A_O_query_mask=[]
                _forward_A_O_query_seg=[]
                _forward_A_O_answer_start=[]
                _forward_A_O_answer_end=[]

                # (12)A,O,C->P


                ################
                # 一系列数据检查操作(待写）
                ################


                ###############
                # 构建模型输入
                ###############

                # （1）S->A
                T_Q_SA=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in S_A_QUERY for word_ in word])
                _forward_S_A_query.append(T_CLS+T_Q_SA+T_SEP+T_text+T_SEP)
                _S_A_query_mask_temp = [1] * len(_forward_S_A_query[0]) # 因为就1个，直接取0就行
                _S_A_query_seg_temp = [0] * (len(_forward_S_A_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                _forward_S_A_query_mask.append(_S_A_query_mask_temp)
                _forward_S_A_query_seg.append(_S_A_query_seg_temp)

                _forward_S_A_answer_start_temp = [-1] * (len(T_Q_SA)+2) + S_A_ANSWER[0][0]+[-1]
                _forward_S_A_answer_end_temp = [-1] * (len(T_Q_SA)+2) + S_O_ANSWER[0][1]+[-1]

                _forward_S_A_answer_start.append(_forward_S_A_answer_start_temp)
                _forward_S_A_answer_end.append(_forward_S_A_answer_end_temp)

                # （2）S->O
                T_Q_SO=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in S_O_QUERY for word_ in word])
                _forward_S_O_query.append(T_CLS+T_Q_SO+T_SEP+T_text+T_SEP)# build_inputs_with_special_tokens等效
                _S_O_query_mask_temp = [1] * len(_forward_S_O_query[0]) # 因为就1个，直接取0就行
                _S_O_query_seg_temp = [0] * (len(_forward_S_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                _forward_S_O_query_mask.append(_S_O_query_mask_temp)
                _forward_S_O_query_seg.append(_S_O_query_seg_temp)

                _forward_asp_answer_start_temp = [-1] * (len(T_Q_SO)+2) + S_O_ANSWER[0][0]+[-1]
                _forward_asp_answer_end_temp = [-1] * (len(T_Q_SO)+2) + S_O_ANSWER[0][1]+[-1]

                _forward_S_O_answer_start.append(_forward_asp_answer_start_temp)
                _forward_S_O_answer_end.append(_forward_asp_answer_end_temp)

                # (3)S->C

                # (4)A->O
                # 好像不需要考虑多组问题，A_O_QUERY
                for ao_index,ao_item in A_O_QUERY:
                    T_Q_AO=self.tokenizer.convert_tokens_to_ids([ word_.lower() for word in ao_item for word_ in word])
                    _forward_A_O_query.append(T_CLS+T_Q_SO+T_SEP+T_text+T_SEP)
                    _A_O_query_mask_temp = [1] * len(_forward_A_O_query[0]) # 因为就1个，直接取0就行
                    _A_O_query_seg_temp = [0] * (len(_forward_A_O_query[0]) - len(T_text)-1) + [1] * (len(T_text)+1)
                    _forward_A_O_query_mask.append(_A_O_query_mask_temp)
                    _forward_A_O_query_seg.append(_A_O_query_seg_temp)

                    _forward_asp_answer_start_temp = [-1] * (len(T_Q_SO)+2) + A_O_ANSWER[0][0]+[-1]
                    _forward_asp_answer_end_temp = [-1] * (len(T_Q_SO)+2) + A_O_ANSWER[0][1]+[-1]

                    _forward_A_O_answer_start.append(_forward_asp_answer_start_temp)
                    _forward_A_O_answer_end.append(_forward_asp_answer_end_temp)


                # (5)A->C

                # (6)A,O->C

                # (7)A,C->O

                # (8)O->A

                # (9)O->C

                # (10)O,A->C

                # (11)O,C->A

                # (12)A,O,C->P

                pass





        else:
            pass


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
    data_path='./data/laptopdev.json'
    data_type='laptop'
    dataset_type='dev'
    wawa=SYZDataset(opt=temp_opt,data_path=data_path,data_type=data_type,dataset_type=dataset_type)