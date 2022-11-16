import argparse
import math
import os
from datetime import datetime
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset



from BMRCs import BMRC, RoBMRC
from data_process import SYZDataset
import DatasetCapsulation as Data
import utils
import Data as Data2
from torch.utils.data import Dataset

#inference_beta = [0.90, 0.90, 0.90, 0.90]  # 推理时候的阈值



"""
    该代码的阅读要点：
        ①
        ②inference(test)和train的过程中有不同的操作
        
"""




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

##########
# debug
##########
def test4(model, t, batch_generator_1,batch_generator_2, standard, beta, logger):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_pol_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_pol_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_pol_match_num = 0

    # batch_dict: b1 ,是BMRC原始的
    for (batch_index, batch_dict),(b_2,d_2) in zip(enumerate(batch_generator_1),enumerate(batch_generator_2)):

        triplets_target = standard[batch_index]['triplet']
        asp_target = standard[batch_index]['asp_target']
        opi_target = standard[batch_index]['opi_target']
        asp_opi_target = standard[batch_index]['asp_opi_target']
        asp_pol_target = standard[batch_index]['asp_pol_target']

        # 预测三元组
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_pol_predict = []

        forward_pair_list = []
        forward_pair_prob = []
        forward_pair_ind_list = []

        backward_pair_list = []
        backward_pair_prob = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []
        final_asp_ind_list = []
        final_opi_ind_list = []
        # forward q_1

        a_dic=copy.deepcopy(batch_dict)

        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        passenge = batch_dict['forward_asp_query'][0][passenge_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 0)

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，呃呃
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1) # 获得最大的概率（softmax之后的）和其位置（0，1）
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)       # 同理是span_end的概率和位置

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        # 下面的-1 在batch_dict那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
        for i in range(f_asp_start_ind.size(0)):   # 句子的长度（这个76）
            if batch_dict['forward_asp_answer_start'][0, i] != -1:
                if f_asp_start_ind[i].item() == 1:            # 如果预测结果认为该位置是1
                    f_asp_start_index_temp.append(i)          # 那么先把i添加到start_index里
                    f_asp_start_prob_temp.append(f_asp_start_prob[i].item())        # 再把概率仍进去
                if f_asp_end_ind[i].item() == 1:
                    f_asp_end_index_temp.append(i)            # 同上
                    f_asp_end_prob_temp.append(f_asp_end_prob[i].item()) # 同上


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)
        # 第一次过滤？
        #######
        ## b2 copy:
        #######

        # passenge_index_2 = b_2['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        # passenge_2 = b_2['forward_asp_query'][0][passenge_index].squeeze(1)
        #
        # b_2['backward_opi_answer_start']= b_2['_forward_S_A_answer_start']
        # b_2['backward_opi_query'] = b_2['_forward_S_A_query']
        # b_2['backward_opi_query_mask'] = b_2['_forward_S_A_query_mask']
        # b_2['backward_opi_query_seg'] = b_2['_forward_S_A_query_seg']
        #
        #
        # f_asp_start_scores, f_asp_end_scores = model(b_2['forward_asp_query'],
        #                                              b_2['forward_asp_query_mask'],
        #                                              b_2['forward_asp_query_seg'], 0)
        #
        # f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，呃呃
        # f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        # f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1) # 获得最大的概率（softmax之后的）和其位置（0，1）
        # f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)       # 同理是span_end的概率和位置
        #
        # f_asp_start_prob_temp = []
        # f_asp_end_prob_temp = []
        # f_asp_start_index_temp = []
        # f_asp_end_index_temp = []
        #
        # # 下面的-1 在b_2那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
        # for i in range(f_asp_start_ind.size(0)):   # 句子的长度（这个76）
        #     if b_2['forward_asp_answer_start'][0, i] != -1:
        #         if f_asp_start_ind[i].item() == 1:            # 如果预测结果认为该位置是1
        #             f_asp_start_index_temp.append(i)          # 那么先把i添加到start_index里
        #             f_asp_start_prob_temp.append(f_asp_start_prob[i].item())        # 再把概率仍进去
        #         if f_asp_end_ind[i].item() == 1:
        #             f_asp_end_index_temp.append(i)            # 同上
        #             f_asp_end_prob_temp.append(f_asp_end_prob[i].item()) # 同上
        #
        #
        # f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
        #     f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)





        #####
        for i in range(len(f_asp_start_index)): # 对每一个可能是aspect的单词
            opinion_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1): # 因为是span所以要循环
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(t.convert_tokens_to_ids('?'))
            opinion_query.append(t.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query) # 这个长度是要算的

            opinion_query = torch.tensor(opinion_query).long().cuda()
            opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
            opinion_query_seg += [1] * passenge.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
            opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            # 第二次过滤？
            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)


            # pair的过滤
            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[i]-5, f_asp_end_index[i]-5] # 因为提问的长度是5，也就是说 [CLS] What aspects ? [SEP] 对应的长度是5，故减掉就是在原句中的？（说对应是因为一个单词长度不一定是1）
                opi_ind = [f_opi_start_index[idx]-f_opi_length, f_opi_end_index[idx]-f_opi_length] # 因为aspect的长度不同，所以这里是之前算的问题Q的长度，减去之后就是原位置
                temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                if asp_ind + opi_ind not in forward_pair_list: # 加法也行吧，不用dic，节省一点内存
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)


        # backward q_1
        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 0)
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)


        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for i in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, i] != -1:
                if b_opi_start_ind[i].item() == 1:
                    b_opi_start_index_temp.append(i)
                    b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                if b_opi_end_ind[i].item() == 1:
                    b_opi_end_index_temp.append(i)
                    b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)



        # backward q_2
        for i in range(len(b_opi_start_index)):
            aspect_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(t.convert_tokens_to_ids('describe'))
            aspect_query.append(t.convert_tokens_to_ids('?'))
            aspect_query.append(t.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long().cuda()
            aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
            aspect_query_seg += [1] * passenge.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
            aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx]-b_asp_length, b_asp_end_index[idx]-b_asp_length]
                opi_ind = [b_opi_start_index[i]-5, b_opi_end_index[i]-5]
                temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        # filter triplet
        # forward
        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list: # 都有就肯定是了
                if forward_pair_list[idx][0] not in final_asp_list: # 对应的aspect（list） 不在里面就加进去就行
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else: # 对应的aspect（list）在里面（这里是作者认为一个aspect可能不只是对应于一个opinion）
                    asp_index = final_asp_list.index(forward_pair_list[idx][0]) #先获取aspect的位置
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]: # 如果这个是aspect1-opinion2
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])  # 添加
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:]) # 添加
            else:
                if forward_pair_prob[idx] >= beta:  # 如果阈值大于这个就保存，这个为啥不进行一个训练呢？
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
         # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
    # sentiment
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            sentiment_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What sentiment given the aspect'.split(' ')])
            sentiment_query+=final_asp_list[idx]
            sentiment_query += t.convert_tokens_to_ids([word.lower() for word in 'and the opinion'.split(' ')])
            # # 拼接所有的opinion
            for idy in range(predict_opinion_num):
                sentiment_query+=final_opi_list[idx][idy]
                if idy < predict_opinion_num - 1:
                    sentiment_query.append(t.convert_tokens_to_ids('/'))
            sentiment_query.append(t.convert_tokens_to_ids('?'))
            sentiment_query.append(t.convert_tokens_to_ids('[SEP]'))

            sentiment_query_seg = [0] * len(sentiment_query)
            sentiment_query = torch.tensor(sentiment_query).long().cuda()
            sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
            sentiment_query_seg += [1] * passenge.size(0)
            sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
            sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

            sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
            sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

            # 每个opinion对应一个三元组
            for idy in range(predict_opinion_num):
                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                triplet_predict = asp_f + opi_f + [sentiment_predicted]
                triplets_predict.append(triplet_predict)
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_pol_predict:
                    asp_pol_predict.append(asp_f + [sentiment_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_pol_target_num += len(asp_pol_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_pol_predict_num += len(asp_pol_predict)

        for trip in triplets_target:
            for trip_ in triplets_predict:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_target:
            for trip_ in asp_predict:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_target:
            for trip_ in opi_predict:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_target:
            for trip_ in asp_opi_predict:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_pol_target:
            for trip_ in asp_pol_predict:
                if trip_ == trip:
                    asp_pol_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num+1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num+1e-6)
    f1 = 2 * precision * recall / (precision + recall+1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))


    precision_aspect = float(asp_match_num) / float(asp_predict_num+1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num+1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect+1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num+1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num+1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion+1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num+1e-6)
    recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num+1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment+1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num+1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num+1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion+1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1


def test3(model, t, batch_generator, standard, beta, logger):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_pol_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_pol_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_pol_match_num = 0

    for batch_index, batch_dict in enumerate(batch_generator):

        triplets_target = standard[batch_index]['triplet']
        asp_target = standard[batch_index]['asp_target']
        opi_target = standard[batch_index]['opi_target']
        asp_opi_target = standard[batch_index]['asp_opi_target']
        asp_pol_target = standard[batch_index]['asp_pol_target']

        # 预测三元组
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_pol_predict = []

        forward_pair_list = []
        forward_pair_prob = []
        forward_pair_ind_list = []

        backward_pair_list = []
        backward_pair_prob = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []
        final_asp_ind_list = []
        final_opi_ind_list = []
        # forward q_1

        # a_dic=copy.deepcopy(batch_dict)
        #
        # a_dic['forward_asp_answer_start']= batch_dict['_forward_S_A_answer_start']
        # a_dic['forward_asp_query']= batch_dict['_forward_S_A_query']
        # a_dic['forward_asp_query_mask']= batch_dict['_forward_S_A_query_mask']
        # a_dic['forward_asp_query_seg']= batch_dict['_forward_S_A_query_seg']

        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        passenge = batch_dict['forward_asp_query'][0][passenge_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 0)
        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，呃呃
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1) # 获得最大的概率（softmax之后的）和其位置（0，1）
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)       # 同理是span_end的概率和位置

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        # 下面的-1 在batch_dict那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
        for i in range(f_asp_start_ind.size(0)):   # 句子的长度（这个76）
            if batch_dict['forward_asp_answer_start'][0, i] != -1:
                if f_asp_start_ind[i].item() == 1:            # 如果预测结果认为该位置是1
                    f_asp_start_index_temp.append(i)          # 那么先把i添加到start_index里
                    f_asp_start_prob_temp.append(f_asp_start_prob[i].item())        # 再把概率仍进去
                if f_asp_end_ind[i].item() == 1:
                    f_asp_end_index_temp.append(i)            # 同上
                    f_asp_end_prob_temp.append(f_asp_end_prob[i].item()) # 同上


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)
        # 第一次过滤？


        for i in range(len(f_asp_start_index)): # 对每一个可能是aspect的单词
            opinion_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1): # 因为是span所以要循环
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(t.convert_tokens_to_ids('?'))
            opinion_query.append(t.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query) # 这个长度是要算的

            opinion_query = torch.tensor(opinion_query).long().cuda()
            opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
            opinion_query_seg += [1] * passenge.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
            opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            # 第二次过滤？
            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)


            # pair的过滤
            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[i]-5, f_asp_end_index[i]-5] # 因为提问的长度是5，也就是说 [CLS] What aspects ? [SEP] 对应的长度是5，故减掉就是在原句中的？（说对应是因为一个单词长度不一定是1）
                opi_ind = [f_opi_start_index[idx]-f_opi_length, f_opi_end_index[idx]-f_opi_length] # 因为aspect的长度不同，所以这里是之前算的问题Q的长度，减去之后就是原位置
                temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                if asp_ind + opi_ind not in forward_pair_list: # 加法也行吧，不用dic，节省一点内存
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        # batch_dict['backward_opi_answer_start']= batch_dict['_forward_S_A_answer_start']
        # batch_dict['backward_opi_query'] = batch_dict['_forward_S_A_query']
        # batch_dict['backward_opi_query_mask'] = batch_dict['_forward_S_A_query_mask']
        # batch_dict['backward_opi_query_seg'] = batch_dict['_forward_S_A_query_seg']

        # backward q_1
        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 0)
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)


        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for i in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, i] != -1:
                if b_opi_start_ind[i].item() == 1:
                    b_opi_start_index_temp.append(i)
                    b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                if b_opi_end_ind[i].item() == 1:
                    b_opi_end_index_temp.append(i)
                    b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)



        # backward q_2
        for i in range(len(b_opi_start_index)):
            aspect_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(t.convert_tokens_to_ids('describe'))
            aspect_query.append(t.convert_tokens_to_ids('?'))
            aspect_query.append(t.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long().cuda()
            aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
            aspect_query_seg += [1] * passenge.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
            aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx]-b_asp_length, b_asp_end_index[idx]-b_asp_length]
                opi_ind = [b_opi_start_index[i]-5, b_opi_end_index[i]-5]
                temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        # filter triplet
        # forward
        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list: # 都有就肯定是了
                if forward_pair_list[idx][0] not in final_asp_list: # 对应的aspect（list） 不在里面就加进去就行
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else: # 对应的aspect（list）在里面（这里是作者认为一个aspect可能不只是对应于一个opinion）
                    asp_index = final_asp_list.index(forward_pair_list[idx][0]) #先获取aspect的位置
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]: # 如果这个是aspect1-opinion2
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])  # 添加
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:]) # 添加
            else:
                if forward_pair_prob[idx] >= beta:  # 如果阈值大于这个就保存，这个为啥不进行一个训练呢？
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
         # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
    # sentiment
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            sentiment_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What sentiment given the aspect'.split(' ')])
            sentiment_query+=final_asp_list[idx]
            sentiment_query += t.convert_tokens_to_ids([word.lower() for word in 'and the opinion'.split(' ')])
            # # 拼接所有的opinion
            for idy in range(predict_opinion_num):
                sentiment_query+=final_opi_list[idx][idy]
                if idy < predict_opinion_num - 1:
                    sentiment_query.append(t.convert_tokens_to_ids('/'))
            sentiment_query.append(t.convert_tokens_to_ids('?'))
            sentiment_query.append(t.convert_tokens_to_ids('[SEP]'))

            sentiment_query_seg = [0] * len(sentiment_query)
            sentiment_query = torch.tensor(sentiment_query).long().cuda()
            sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
            sentiment_query_seg += [1] * passenge.size(0)
            sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
            sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

            sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
            sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

            # 每个opinion对应一个三元组
            for idy in range(predict_opinion_num):
                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                triplet_predict = asp_f + opi_f + [sentiment_predicted]
                triplets_predict.append(triplet_predict)
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_pol_predict:
                    asp_pol_predict.append(asp_f + [sentiment_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_pol_target_num += len(asp_pol_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_pol_predict_num += len(asp_pol_predict)

        for trip in triplets_target:
            for trip_ in triplets_predict:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_target:
            for trip_ in asp_predict:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_target:
            for trip_ in opi_predict:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_target:
            for trip_ in asp_opi_predict:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_pol_target:
            for trip_ in asp_pol_predict:
                if trip_ == trip:
                    asp_pol_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num+1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num+1e-6)
    f1 = 2 * precision * recall / (precision + recall+1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))


    precision_aspect = float(asp_match_num) / float(asp_predict_num+1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num+1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect+1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num+1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num+1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion+1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num+1e-6)
    recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num+1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment+1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num+1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num+1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion+1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1


def test2(model, t, batch_generator, standard, beta, logger):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_pol_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_pol_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_pol_match_num = 0

    for batch_index, batch_dict in enumerate(batch_generator):

        triplets_target = standard[batch_index]['triplet']
        asp_target = standard[batch_index]['asp_target']
        opi_target = standard[batch_index]['opi_target']
        asp_opi_target = standard[batch_index]['asp_opi_target']
        asp_pol_target = standard[batch_index]['asp_pol_target']

        # 预测三元组
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_pol_predict = []

        forward_pair_list = []
        forward_pair_prob = []
        forward_pair_ind_list = []

        backward_pair_list = []
        backward_pair_prob = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []
        final_asp_ind_list = []
        final_opi_ind_list = []
        # forward q_1
        batch_dict['forward_asp_answer_start']= batch_dict['_forward_S_A_answer_start']
        batch_dict['forward_asp_query']= batch_dict['_forward_S_A_query']
        batch_dict['forward_asp_query_mask']= batch_dict['_forward_S_A_query_mask']
        batch_dict['forward_asp_query_seg']= batch_dict['_forward_S_A_query_seg']

        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        passenge = batch_dict['forward_asp_query'][0][passenge_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 0)
        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，呃呃
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1) # 获得最大的概率（softmax之后的）和其位置（0，1）
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)       # 同理是span_end的概率和位置

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        # 下面的-1 在batch_dict那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
        for i in range(f_asp_start_ind.size(0)):   # 句子的长度（这个76）
            if batch_dict['forward_asp_answer_start'][0, i] != -1:
                if f_asp_start_ind[i].item() == 1:            # 如果预测结果认为该位置是1
                    f_asp_start_index_temp.append(i)          # 那么先把i添加到start_index里
                    f_asp_start_prob_temp.append(f_asp_start_prob[i].item())        # 再把概率仍进去
                if f_asp_end_ind[i].item() == 1:
                    f_asp_end_index_temp.append(i)            # 同上
                    f_asp_end_prob_temp.append(f_asp_end_prob[i].item()) # 同上


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)
        # 第一次过滤？


        for i in range(len(f_asp_start_index)): # 对每一个可能是aspect的单词
            opinion_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1): # 因为是span所以要循环
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(t.convert_tokens_to_ids('?'))
            opinion_query.append(t.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query) # 这个长度是要算的

            opinion_query = torch.tensor(opinion_query).long().cuda()
            opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
            opinion_query_seg += [1] * passenge.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
            opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            # 第二次过滤？
            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)


            # pair的过滤
            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[i]-5, f_asp_end_index[i]-5] # 因为提问的长度是5，也就是说 [CLS] What aspects ? [SEP] 对应的长度是5，故减掉就是在原句中的？（说对应是因为一个单词长度不一定是1）
                opi_ind = [f_opi_start_index[idx]-f_opi_length, f_opi_end_index[idx]-f_opi_length] # 因为aspect的长度不同，所以这里是之前算的问题Q的长度，减去之后就是原位置
                temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                if asp_ind + opi_ind not in forward_pair_list: # 加法也行吧，不用dic，节省一点内存
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        batch_dict['backward_opi_answer_start']= batch_dict['_forward_S_O_answer_start']
        batch_dict['backward_opi_query'] = batch_dict['_forward_S_O_query']
        batch_dict['backward_opi_query_mask'] = batch_dict['_forward_S_O_query_mask']
        batch_dict['backward_opi_query_seg'] = batch_dict['_forward_S_O_query_seg']

        # backward q_1
        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 0)
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)


        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for i in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, i] != -1:
                if b_opi_start_ind[i].item() == 1:
                    b_opi_start_index_temp.append(i)
                    b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                if b_opi_end_ind[i].item() == 1:
                    b_opi_end_index_temp.append(i)
                    b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)



        # backward q_2
        for i in range(len(b_opi_start_index)):
            aspect_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(t.convert_tokens_to_ids('describe'))
            aspect_query.append(t.convert_tokens_to_ids('?'))
            aspect_query.append(t.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long().cuda()
            aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
            aspect_query_seg += [1] * passenge.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
            aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx]-b_asp_length, b_asp_end_index[idx]-b_asp_length]
                opi_ind = [b_opi_start_index[i]-5, b_opi_end_index[i]-5]
                temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        # filter triplet
        # forward
        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list: # 都有就肯定是了
                if forward_pair_list[idx][0] not in final_asp_list: # 对应的aspect（list） 不在里面就加进去就行
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else: # 对应的aspect（list）在里面（这里是作者认为一个aspect可能不只是对应于一个opinion）
                    asp_index = final_asp_list.index(forward_pair_list[idx][0]) #先获取aspect的位置
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]: # 如果这个是aspect1-opinion2
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])  # 添加
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:]) # 添加
            else:
                if forward_pair_prob[idx] >= beta:  # 如果阈值大于这个就保存，这个为啥不进行一个训练呢？
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
         # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
    # sentiment
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            sentiment_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What sentiment given the aspect'.split(' ')])
            sentiment_query+=final_asp_list[idx]
            sentiment_query += t.convert_tokens_to_ids([word.lower() for word in 'and the opinion'.split(' ')])
            # # 拼接所有的opinion
            for idy in range(predict_opinion_num):
                sentiment_query+=final_opi_list[idx][idy]
                if idy < predict_opinion_num - 1:
                    sentiment_query.append(t.convert_tokens_to_ids('/'))
            sentiment_query.append(t.convert_tokens_to_ids('?'))
            sentiment_query.append(t.convert_tokens_to_ids('[SEP]'))

            sentiment_query_seg = [0] * len(sentiment_query)
            sentiment_query = torch.tensor(sentiment_query).long().cuda()
            sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
            sentiment_query_seg += [1] * passenge.size(0)
            sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
            sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

            sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
            sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

            # 每个opinion对应一个三元组
            for idy in range(predict_opinion_num):
                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                triplet_predict = asp_f + opi_f + [sentiment_predicted]
                triplets_predict.append(triplet_predict)
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_pol_predict:
                    asp_pol_predict.append(asp_f + [sentiment_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_pol_target_num += len(asp_pol_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_pol_predict_num += len(asp_pol_predict)

        for trip in triplets_target:
            for trip_ in triplets_predict:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_target:
            for trip_ in asp_predict:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_target:
            for trip_ in opi_predict:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_target:
            for trip_ in asp_opi_predict:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_pol_target:
            for trip_ in asp_pol_predict:
                if trip_ == trip:
                    asp_pol_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num+1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num+1e-6)
    f1 = 2 * precision * recall / (precision + recall+1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))


    precision_aspect = float(asp_match_num) / float(asp_predict_num+1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num+1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect+1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num+1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num+1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion+1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num+1e-6)
    recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num+1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment+1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num+1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num+1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion+1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1

def test(model, tokenizer, batch_generator, gold_list, beta, logger,task_type):
    """
    # 暂时先用三元组作为例子
    :param model: 用的模型
    :param tokenizer: 对应的tokenizer
    :param batch_generator: dataset对应的dataloader
    :param gold_list: 对应的标准答案：
        格式：
            主要是为了方便一次算全部的效能
            说来我一开始以为BMRC他们是针对每个都要包含不同的数据，这样一看其实和其他模型进行比较并不是特别公平
            以三元组数据为例：
            aspcets:[[a,b],[c,d],[e,f]]
            opinions:[[g,h],[i,j],[k,q]]
            as_op:[[a,b,g,h]
            as_op_po:[[a,b,g,h,1],[c,d,i,j,0],[e,f,k,q,2]]
            as_po:[[a,b,1],[c,d,0],[e,f,2]]
    :param inference_beta: 就推理值，目前来讲用最上面那个就行
    :param logger: 日志保存的
    :param gpu: 判断cda
    :param max_len: 这个emm。。
    :return:对应的F1、ACC、Recall值
    """
    # 推理过程

    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_pol_target_num = 0

    triplet_predict_num = 0
    a_predict_num = 0
    o_predict_num = 0
    a_o_predict_num = 0
    asp_pol_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_pol_match_num = 0
    if task_type=='ASQP':
        for batch_index, batch_dict in enumerate(batch_generator):

            # 对应的标准答案，应该是三元组的对？不过数据集里应该是没有这项，可能确实要进行一个新的数据处理
            # 即针对dev和test，最终还要有相应的答案对（对应的索引大概是）

            # aspcets: [[a, b], [c, d], [e, f]]
            # opinions: [[g, h], [i, j], [k, q]]
            # as_op: [[a, b, g, h]
            # as_op_po:[[a, b, g, h, 1], [c, d, i, j, 0], [e, f, k, q, 2]]
            # as_po: [[a, b, 1], [c, d, 0], [e, f, 2]]

            a_o_p_target = gold_list[batch_index]['as_op_po']
            # 提取出的aspect
            a_target = gold_list[batch_index]['aspects']
            # 提取出的opinion
            o_target = gold_list[batch_index]['opinions']
            # 对应的aspect opinion对
            a_o_target = gold_list[batch_index]['as_op']
            # 最后的（AO,P）,注意和a_o_p不一样哦
            a_p_target = gold_list[batch_index]['as_po']

            # 预测对应的值
            a_o_p_predict = []
            a_predict = []
            o_predict = []
            a_o_predict = []
            asp_pol_predict = []

            # 用来存储 （A_O）对
            A_O_pair_list = []
            # 存储相应的概率
            A_O_pair_prob = []
            # 这是干啥的我也不到
            A_O_pair_ind_list = []

            # 用来存储（O_A）对
            O_A_pair_list = []
            O_A_pair_prob = []
            O_A_pair_ind_list = []

            # 最终得到的两个组
            final_asp_list = []
            final_opi_list = []
            final_asp_ind_list = []
            final_opi_ind_list = []

            # 首先提取S_A对
            # 这里其实取[0]没什么所谓,因为batch_size是1

            passenge_index = batch_dict['_forward_S_A_answer_start'][0][0].gt(-1).float().nonzero()
            passenge = batch_dict['_forward_S_A_query'][0][0][passenge_index].squeeze(1)


            # S_A

            f_asp_start_scores, f_asp_end_scores = model(
                batch_dict['_forward_S_A_query'].view(-1, batch_dict['_forward_S_A_query'].size(-1)),
                batch_dict['_forward_S_A_query_mask'].view(-1, batch_dict['_forward_S_A_query_mask'].size(-1)),
                batch_dict['_forward_S_A_query_seg'].view(-1, batch_dict['_forward_S_A_query_seg'].size(-1)), step=0)

            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，这里之后对比学习的时候就需要进行一下相应的更改
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)  # 获得最大的概率（softmax之后的）和其位置（0，1）
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)  # 同理是span_end的概率和位置

            # 保存S_A,S_O,A_O,O_A的每一步的相关信息
            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []

            # 下面的-1 在batch_dict那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
            for i in range(f_asp_start_ind.size(0)):  # 句子的长度（这个76）
                if batch_dict['_forward_S_A_answer_start'][0][0, i] != -1:
                    if f_asp_start_ind[i].item() == 1:  # 如果预测结果认为该位置是1
                        f_asp_start_index_temp.append(i)  # 那么先把i添加到start_index里
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())  # 再把概率仍进去
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)  # 同上
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())  # 同上

            f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)
            # 推理的时候要进行过滤，把阈值不能接受的内容去掉，其中BMRC使用的是概率的乘积，RoBMRC使用的是另外一种计算方式

            # 过滤完之后就该进行A_O的查找了
            for i in range(len(f_asp_start_index)):  # 对每一个可能是aspect的单词
                opinion_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What opinion given the aspect'.split(' ')])
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):  # 因为是span所以要循环
                    opinion_query.append(batch_dict['_forward_A_O_query'][0][0][j].item())
                opinion_query.append(tokenizer.convert_tokens_to_ids('?'))
                opinion_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
                opinion_query_seg = [0] * len(opinion_query)
                f_opi_length = len(opinion_query)  # 这个长度是要算的

                opinion_query = torch.tensor(opinion_query).long().cuda()
                opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
                opinion_query_seg += [1] * passenge.size(0)
                opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
                opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp = []
                f_opi_end_prob_temp = []
                f_opi_start_index_temp = []
                f_opi_end_index_temp = []
                for k in range(f_opi_start_ind.size(0)):
                    if opinion_query_seg[0, k] == 1:
                        if f_opi_start_ind[k].item() == 1:
                            f_opi_start_index_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_ind[k].item() == 1:
                            f_opi_end_index_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                # 对aspect_opinion的过滤是同理的，概率相乘之后小于对应的阈值就把他去掉
                f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)

                # pair的过滤
                for idx in range(len(f_opi_start_index)):
                    asp = [batch_dict['_forward_A_O_query'][0][0][j].item() for j in
                           range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                    opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                    asp_ind = [f_asp_start_index[i] - 5, f_asp_end_index[
                        i] - 5]  # 因为提问的长度是5，也就是说 [CLS] What aspects ? [SEP] 对应的长度是5，故减掉就是在原句中的？（说对应是因为一个单词长度不一定是1）
                    opi_ind = [f_opi_start_index[idx] - f_opi_length,
                               f_opi_end_index[idx] - f_opi_length]  # 因为aspect的长度不同，所以这里是之前算的问题Q的长度，减去之后就是原位置
                    temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                    if asp_ind + opi_ind not in A_O_pair_list:  # 加法也行吧，不用dic，节省一点内存
                        A_O_pair_list.append([asp] + [opi])
                        A_O_pair_prob.append(temp_prob)
                        A_O_pair_ind_list.append(asp_ind + opi_ind)
                    else:
                        print('erro')
                        exit(1)
            ###########
            # S->O->A
            ##########
            b_opi_start_scores, b_opi_end_scores =  model(
                batch_dict['_forward_S_O_query'].view(-1, batch_dict['_forward_S_O_query'].size(-1)),
                batch_dict['_forward_S_O_query_mask'].view(-1, batch_dict['_forward_S_O_query_mask'].size(-1)),
                batch_dict['_forward_S_O_query_seg'].view(-1, batch_dict['_forward_S_O_query_seg'].size(-1)), step=0)
            b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
            b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
            b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
            b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

            b_opi_start_prob_temp = []
            b_opi_end_prob_temp = []
            b_opi_start_index_temp = []
            b_opi_end_index_temp = []
            for i in range(b_opi_start_ind.size(0)):
                if batch_dict['_forward_S_O_answer_start'][0][0, i] != -1:
                    if b_opi_start_ind[i].item() == 1:
                        b_opi_start_index_temp.append(i)
                        b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                    if b_opi_end_ind[i].item() == 1:
                        b_opi_end_index_temp.append(i)
                        b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

            b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
                b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)

            # O_A 过程，和上面的大同小异，不对每句话进行相关叙述了
            for i in range(len(b_opi_start_index)):
                aspect_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What aspect does the opinion'.split(' ')])
                for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                    aspect_query.append(batch_dict['_forward_O_A_query'][0][0][j].item())
                aspect_query.append(tokenizer.convert_tokens_to_ids('describe'))
                aspect_query.append(tokenizer.convert_tokens_to_ids('?'))
                aspect_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
                aspect_query_seg = [0] * len(aspect_query)
                b_asp_length = len(aspect_query)
                aspect_query = torch.tensor(aspect_query).long().cuda()
                aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
                aspect_query_seg += [1] * passenge.size(0)
                aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
                aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

                b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

                b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
                b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

                b_asp_start_prob_temp = []
                b_asp_end_prob_temp = []
                b_asp_start_index_temp = []
                b_asp_end_index_temp = []
                for k in range(b_asp_start_ind.size(0)):
                    if aspect_query_seg[0, k] == 1:
                        if b_asp_start_ind[k].item() == 1:
                            b_asp_start_index_temp.append(k)
                            b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                        if b_asp_end_ind[k].item() == 1:
                            b_asp_end_index_temp.append(k)
                            b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                    b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

                for idx in range(len(b_asp_start_index)):
                    opi = [batch_dict['_forward_O_A_query'][0][0][j].item() for j in
                           range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                    asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                    asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                    opi_ind = [b_opi_start_index[i] - 5, b_opi_end_index[i] - 5]
                    temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                    if asp_ind + opi_ind not in O_A_pair_ind_list:
                        O_A_pair_list.append([asp] + [opi])
                        O_A_pair_prob.append(temp_prob)
                        O_A_pair_ind_list.append(asp_ind + opi_ind)
                    else:
                        print('erro')
                        exit(1)
            # filter triplet
            # forward
            for idx in range(len(A_O_pair_list)):
                if A_O_pair_list[idx] in O_A_pair_list:  # 都有就肯定是了
                    if A_O_pair_list[idx][0] not in final_asp_list:  # 对应的aspect（list） 不在里面就加进去就行
                        final_asp_list.append(A_O_pair_list[idx][0])
                        final_opi_list.append([A_O_pair_list[idx][1]])
                        final_asp_ind_list.append(A_O_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([A_O_pair_ind_list[idx][2:]])
                    else:  # 对应的aspect（list）在里面（这里是作者认为一个aspect可能不只是对应于一个opinion）
                        asp_index = final_asp_list.index(A_O_pair_list[idx][0])  # 先获取aspect的位置
                        if A_O_pair_list[idx][1] not in final_opi_list[asp_index]:  # 如果这个是aspect1-opinion2
                            final_opi_list[asp_index].append(A_O_pair_list[idx][1])  # 添加
                            final_opi_ind_list[asp_index].append(A_O_pair_ind_list[idx][2:])  # 添加
                else:
                    if A_O_pair_prob[idx] >= beta:  # 如果阈值大于这个就保存，这个为啥不进行一个训练呢？
                        if A_O_pair_list[idx][0] not in final_asp_list:
                            final_asp_list.append(A_O_pair_list[idx][0])
                            final_opi_list.append([A_O_pair_list[idx][1]])
                            final_asp_ind_list.append(A_O_pair_ind_list[idx][:2])
                            final_opi_ind_list.append([A_O_pair_ind_list[idx][2:]])
                        else:
                            asp_index = final_asp_list.index(A_O_pair_list[idx][0])
                            if A_O_pair_list[idx][1] not in final_opi_list[asp_index]:
                                final_opi_list[asp_index].append(A_O_pair_list[idx][1])
                                final_opi_ind_list[asp_index].append(A_O_pair_ind_list[idx][2:])
            # backward
            for idx in range(len(O_A_pair_list)):
                if O_A_pair_list[idx] not in A_O_pair_list:
                    if O_A_pair_prob[idx] >= beta:
                        if O_A_pair_list[idx][0] not in final_asp_list:
                            final_asp_list.append(O_A_pair_list[idx][0])
                            final_opi_list.append([O_A_pair_list[idx][1]])
                            final_asp_ind_list.append(O_A_pair_ind_list[idx][:2])
                            final_opi_ind_list.append([O_A_pair_ind_list[idx][2:]])
                        else:
                            asp_index = final_asp_list.index(O_A_pair_list[idx][0])
                            if O_A_pair_list[idx][1] not in final_opi_list[asp_index]:
                                final_opi_list[asp_index].append(O_A_pair_list[idx][1])
                                final_opi_ind_list[asp_index].append(O_A_pair_ind_list[idx][2:])
            # sentiment
            for idx in range(len(final_asp_list)):
                predict_opinion_num = len(final_opi_list[idx])
                sentiment_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What sentiment given the aspect'.split(' ')])
                sentiment_query += final_asp_list[idx]
                sentiment_query += tokenizer.convert_tokens_to_ids([word.lower() for word in 'and the opinion'.split(' ')])
                # # 拼接所有的opinion
                for idy in range(predict_opinion_num):
                    sentiment_query += final_opi_list[idx][idy]
                    if idy < predict_opinion_num - 1:
                        sentiment_query.append(tokenizer.convert_tokens_to_ids('/'))
                sentiment_query.append(tokenizer.convert_tokens_to_ids('?'))
                sentiment_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long().cuda()
                sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
                sentiment_query_seg += [1] * passenge.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
                sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                # 每个opinion对应一个三元组
                for idy in range(predict_opinion_num):
                    asp_f = []
                    opi_f = []
                    asp_f.append(final_asp_ind_list[idx][0])
                    asp_f.append(final_asp_ind_list[idx][1])
                    opi_f.append(final_opi_ind_list[idx][idy][0])
                    opi_f.append(final_opi_ind_list[idx][idy][1])
                    triplet_predict = asp_f + opi_f + [sentiment_predicted]
                    a_o_p_predict.append(triplet_predict)
                    if opi_f not in o_predict:
                        o_predict.append(opi_f)
                    if asp_f + opi_f not in a_o_predict:
                        a_o_predict.append(asp_f + opi_f)
                    if asp_f + [sentiment_predicted] not in asp_pol_predict:
                        asp_pol_predict.append(asp_f + [sentiment_predicted])
                    if asp_f not in a_predict:
                        a_predict.append(asp_f)
        #

            ###
            #   这里写的比较简单，之后可以考虑职业用矩阵，应该能更好一点
            ###

            triplet_target_num += len(a_o_p_target)
            asp_target_num += len(a_target)
            opi_target_num += len(o_target)
            asp_opi_target_num += len(a_o_target)
            asp_pol_target_num += len(a_p_target)

            triplet_predict_num += len(a_o_p_predict)
            a_predict_num += len(a_predict)
            o_predict_num += len(o_predict)
            a_o_predict_num += len(a_o_predict)
            asp_pol_predict_num += len(asp_pol_predict)

            a_o_p_predict = []
            a_predict = []
            o_predict = []
            a_o_predict = []
            asp_pol_predict = []
            # 计算每一步的最终的值，包括三元组、asp、opinion、（A,O），还有极性判断，然后根据这些计算F1
            for trip in a_o_p_target:
                for trip_ in a_o_p_predict:
                    if trip_ == trip:
                        triplet_match_num += 1
            for trip in a_target:
                for trip_ in a_predict:
                    if trip_ == trip:
                        asp_match_num += 1
            for trip in o_target:
                for trip_ in o_predict:
                    if trip_ == trip:
                        opi_match_num += 1
            for trip in a_o_target:
                for trip_ in a_o_predict:
                    if trip_ == trip:
                        asp_opi_match_num += 1
            for trip in a_p_target:
                for trip_ in asp_pol_predict:
                    if trip_ == trip:
                        asp_pol_match_num += 1
        #print(1)


        precision = float(triplet_match_num) / float(triplet_predict_num + 1e-6)
        recall = float(triplet_match_num) / float(triplet_target_num + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))

        precision_aspect = float(asp_match_num) / float(a_predict_num + 1e-6)
        recall_aspect = float(asp_match_num) / float(asp_target_num + 1e-6)
        f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect + 1e-6)
        logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

        precision_opinion = float(opi_match_num) / float(o_predict_num + 1e-6)
        recall_opinion = float(opi_match_num) / float(opi_target_num + 1e-6)
        f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion + 1e-6)
        logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

        precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num + 1e-6)
        recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num + 1e-6)
        f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
                precision_aspect_sentiment + recall_aspect_sentiment + 1e-6)
        logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                                  recall_aspect_sentiment,
                                                                                  f1_aspect_sentiment))

        precision_aspect_opinion = float(asp_opi_match_num) / float(a_o_predict_num + 1e-6)
        recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num + 1e-6)
        f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
                precision_aspect_opinion + recall_aspect_opinion + 1e-6)
        logger.info(
            'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                        f1_aspect_opinion))
        return f1
    elif task_type=='ASTE':
        for batch_index, batch_dict in enumerate(batch_generator):

            # 对应的标准答案，应该是三元组的对？不过数据集里应该是没有这项，可能确实要进行一个新的数据处理
            # 即针对dev和test，最终还要有相应的答案对（对应的索引大概是）

            # aspcets: [[a, b], [c, d], [e, f]]
            # opinions: [[g, h], [i, j], [k, q]]
            # as_op: [[a, b, g, h]
            # as_op_po:[[a, b, g, h, 1], [c, d, i, j, 0], [e, f, k, q, 2]]
            # as_po: [[a, b, 1], [c, d, 0], [e, f, 2]]

            tri_index=int(batch_dict['ID'])


            """
            triplets_target = standard[batch_index]['triplet']
            asp_target = standard[batch_index]['asp_target']
            opi_target = standard[batch_index]['opi_target']
            asp_opi_target = standard[batch_index]['asp_opi_target']
            asp_pol_target = standard[batch_index]['asp_pol_target']
            
            """
            a_o_p_target = gold_list[batch_index]['as_op_po']
            # 提取出的aspect
            a_target = gold_list[batch_index]['aspects']
            # 提取出的opinion
            o_target = gold_list[batch_index]['opinions']
            # 对应的aspect opinion对
            a_o_target = gold_list[batch_index]['as_op']
            # 最后的（AO,P）,注意和a_o_p不一样哦
            a_p_target = gold_list[batch_index]['as_po']

            # 预测对应的值
            a_o_p_predict = []
            a_predict = []
            o_predict = []
            a_o_predict = []
            asp_pol_predict = []

            # 用来存储 （A_O）对
            A_O_pair_list = []
            # 存储相应的概率
            A_O_pair_prob = []
            # 这是干啥的我也不到
            A_O_pair_ind_list = []

            # 用来存储（O_A）对
            O_A_pair_list = []
            O_A_pair_prob = []
            O_A_pair_ind_list = []

            # 最终得到的两个组
            final_asp_list = []
            final_opi_list = []
            final_asp_ind_list = []
            final_opi_ind_list = []

            # 首先提取S_A对
            # 这里其实取[0]没什么所谓,因为batch_size是1

            passenge_index = batch_dict['_forward_S_A_answer_start'][0][0].gt(-1).float().nonzero()
            passenge = batch_dict['_forward_S_A_query'][0][0][passenge_index].squeeze(1)

            # S_A

            f_asp_start_scores, f_asp_end_scores = model(
                batch_dict['_forward_S_A_query'].view(-1, batch_dict['_forward_S_A_query'].size(-1)),
                batch_dict['_forward_S_A_query_mask'].view(-1, batch_dict['_forward_S_A_query_mask'].size(-1)),
                batch_dict['_forward_S_A_query_seg'].view(-1, batch_dict['_forward_S_A_query_seg'].size(-1)), step=0)

            f_asp_start_scores = F.softmax(f_asp_start_scores[0],
                                           dim=1)  # 为什么取[0]，因为test_loader的batch_size是1，这里之后对比学习的时候就需要进行一下相应的更改
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)  # 获得最大的概率（softmax之后的）和其位置（0，1）
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)  # 同理是span_end的概率和位置

            # 保存S_A,S_O,A_O,O_A的每一步的相关信息
            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []

            # 下面的-1 在batch_dict那个列表里意味着不是原句子（CLS 到第一个SEP之间，这个是问题）
            for i in range(f_asp_start_ind.size(0)):  # 句子的长度（这个76）
                if batch_dict['_forward_S_A_answer_start'][0][0, i] != -1:
                    if f_asp_start_ind[i].item() == 1:  # 如果预测结果认为该位置是1
                        f_asp_start_index_temp.append(i)  # 那么先把i添加到start_index里
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())  # 再把概率仍进去
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)  # 同上
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())  # 同上

            f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)
            # 推理的时候要进行过滤，把阈值不能接受的内容去掉，其中BMRC使用的是概率的乘积，RoBMRC使用的是另外一种计算方式

            # 过滤完之后就该进行A_O的查找了
            for i in range(len(f_asp_start_index)):  # 对每一个可能是aspect的单词
                opinion_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What opinion given the aspect'.split(' ')])
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):  # 因为是span所以要循环
                    opinion_query.append(batch_dict['_forward_A_O_query'][0][0][j].item())
                opinion_query.append(tokenizer.convert_tokens_to_ids('?'))
                opinion_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
                opinion_query_seg = [0] * len(opinion_query)
                f_opi_length = len(opinion_query)  # 这个长度是要算的

                opinion_query = torch.tensor(opinion_query).long().cuda()
                opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
                opinion_query_seg += [1] * passenge.size(0)
                opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
                opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp = []
                f_opi_end_prob_temp = []
                f_opi_start_index_temp = []
                f_opi_end_index_temp = []
                for k in range(f_opi_start_ind.size(0)):
                    if opinion_query_seg[0, k] == 1:
                        if f_opi_start_ind[k].item() == 1:
                            f_opi_start_index_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_ind[k].item() == 1:
                            f_opi_end_index_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                # 对aspect_opinion的过滤是同理的，概率相乘之后小于对应的阈值就把他去掉
                f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)

                # pair的过滤
                for idx in range(len(f_opi_start_index)):
                    asp = [batch_dict['_forward_A_O_query'][0][0][j].item() for j in
                           range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                    opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                    asp_ind = [f_asp_start_index[i] - 5, f_asp_end_index[
                        i] - 5]  # 因为提问的长度是5，也就是说 [CLS] What aspects ? [SEP] 对应的长度是5，故减掉就是在原句中的？（说对应是因为一个单词长度不一定是1）
                    opi_ind = [f_opi_start_index[idx] - f_opi_length,
                               f_opi_end_index[idx] - f_opi_length]  # 因为aspect的长度不同，所以这里是之前算的问题Q的长度，减去之后就是原位置
                    temp_prob = f_asp_prob[i] * f_opi_prob[idx]
                    if asp_ind + opi_ind not in A_O_pair_list:  # 加法也行吧，不用dic，节省一点内存
                        A_O_pair_list.append([asp] + [opi])
                        A_O_pair_prob.append(temp_prob)
                        A_O_pair_ind_list.append(asp_ind + opi_ind)
                    else:
                        print('erro')
                        exit(1)
            ###########
            # S->O->A
            ##########
            b_opi_start_scores, b_opi_end_scores = model(
                batch_dict['_forward_S_O_query'].view(-1, batch_dict['_forward_S_O_query'].size(-1)),
                batch_dict['_forward_S_O_query_mask'].view(-1, batch_dict['_forward_S_O_query_mask'].size(-1)),
                batch_dict['_forward_S_O_query_seg'].view(-1, batch_dict['_forward_S_O_query_seg'].size(-1)), step=0)
            b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
            b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
            b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
            b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

            b_opi_start_prob_temp = []
            b_opi_end_prob_temp = []
            b_opi_start_index_temp = []
            b_opi_end_index_temp = []
            for i in range(b_opi_start_ind.size(0)):
                if batch_dict['_forward_S_O_answer_start'][0][0, i] != -1:
                    if b_opi_start_ind[i].item() == 1:
                        b_opi_start_index_temp.append(i)
                        b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                    if b_opi_end_ind[i].item() == 1:
                        b_opi_end_index_temp.append(i)
                        b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

            b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
                b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)

            # O_A 过程，和上面的大同小异，不对每句话进行相关叙述了
            for i in range(len(b_opi_start_index)):
                aspect_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What aspect does the opinion'.split(' ')])
                for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                    aspect_query.append(batch_dict['_forward_O_A_query'][0][0][j].item())
                aspect_query.append(tokenizer.convert_tokens_to_ids('describe'))
                aspect_query.append(tokenizer.convert_tokens_to_ids('?'))
                aspect_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))
                aspect_query_seg = [0] * len(aspect_query)
                b_asp_length = len(aspect_query)
                aspect_query = torch.tensor(aspect_query).long().cuda()
                aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
                aspect_query_seg += [1] * passenge.size(0)
                aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
                aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

                b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 0)

                b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
                b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

                b_asp_start_prob_temp = []
                b_asp_end_prob_temp = []
                b_asp_start_index_temp = []
                b_asp_end_index_temp = []
                for k in range(b_asp_start_ind.size(0)):
                    if aspect_query_seg[0, k] == 1:
                        if b_asp_start_ind[k].item() == 1:
                            b_asp_start_index_temp.append(k)
                            b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                        if b_asp_end_ind[k].item() == 1:
                            b_asp_end_index_temp.append(k)
                            b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                    b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

                for idx in range(len(b_asp_start_index)):
                    opi = [batch_dict['_forward_O_A_query'][0][0][j].item() for j in
                           range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                    asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                    asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                    opi_ind = [b_opi_start_index[i] - 5, b_opi_end_index[i] - 5]
                    temp_prob = b_asp_prob[idx] * b_opi_prob[i]
                    if asp_ind + opi_ind not in O_A_pair_ind_list:
                        O_A_pair_list.append([asp] + [opi])
                        O_A_pair_prob.append(temp_prob)
                        O_A_pair_ind_list.append(asp_ind + opi_ind)
                    else:
                        print('erro')
                        exit(1)
            # filter triplet
            # forward
            for idx in range(len(A_O_pair_list)):
                if A_O_pair_list[idx] in O_A_pair_list:  # 都有就肯定是了
                    if A_O_pair_list[idx][0] not in final_asp_list:  # 对应的aspect（list） 不在里面就加进去就行
                        final_asp_list.append(A_O_pair_list[idx][0])
                        final_opi_list.append([A_O_pair_list[idx][1]])
                        final_asp_ind_list.append(A_O_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([A_O_pair_ind_list[idx][2:]])
                    else:  # 对应的aspect（list）在里面（这里是作者认为一个aspect可能不只是对应于一个opinion）
                        asp_index = final_asp_list.index(A_O_pair_list[idx][0])  # 先获取aspect的位置
                        if A_O_pair_list[idx][1] not in final_opi_list[asp_index]:  # 如果这个是aspect1-opinion2
                            final_opi_list[asp_index].append(A_O_pair_list[idx][1])  # 添加
                            final_opi_ind_list[asp_index].append(A_O_pair_ind_list[idx][2:])  # 添加
                else:
                    if A_O_pair_prob[idx] >= beta:  # 如果阈值大于这个就保存，这个为啥不进行一个训练呢？
                        if A_O_pair_list[idx][0] not in final_asp_list:
                            final_asp_list.append(A_O_pair_list[idx][0])
                            final_opi_list.append([A_O_pair_list[idx][1]])
                            final_asp_ind_list.append(A_O_pair_ind_list[idx][:2])
                            final_opi_ind_list.append([A_O_pair_ind_list[idx][2:]])
                        else:
                            asp_index = final_asp_list.index(A_O_pair_list[idx][0])
                            if A_O_pair_list[idx][1] not in final_opi_list[asp_index]:
                                final_opi_list[asp_index].append(A_O_pair_list[idx][1])
                                final_opi_ind_list[asp_index].append(A_O_pair_ind_list[idx][2:])
            # backward
            for idx in range(len(O_A_pair_list)):
                if O_A_pair_list[idx] not in A_O_pair_list:
                    if O_A_pair_prob[idx] >= beta:
                        if O_A_pair_list[idx][0] not in final_asp_list:
                            final_asp_list.append(O_A_pair_list[idx][0])
                            final_opi_list.append([O_A_pair_list[idx][1]])
                            final_asp_ind_list.append(O_A_pair_ind_list[idx][:2])
                            final_opi_ind_list.append([O_A_pair_ind_list[idx][2:]])
                        else:
                            asp_index = final_asp_list.index(O_A_pair_list[idx][0])
                            if O_A_pair_list[idx][1] not in final_opi_list[asp_index]:
                                final_opi_list[asp_index].append(O_A_pair_list[idx][1])
                                final_opi_ind_list[asp_index].append(O_A_pair_ind_list[idx][2:])
            # sentiment
            for idx in range(len(final_asp_list)):
                predict_opinion_num = len(final_opi_list[idx])
                sentiment_query = tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What sentiment given the aspect'.split(' ')])
                sentiment_query += final_asp_list[idx]
                sentiment_query += tokenizer.convert_tokens_to_ids(
                    [word.lower() for word in 'and the opinion'.split(' ')])
                # # 拼接所有的opinion
                for idy in range(predict_opinion_num):
                    sentiment_query += final_opi_list[idx][idy]
                    if idy < predict_opinion_num - 1:
                        sentiment_query.append(tokenizer.convert_tokens_to_ids('/'))
                sentiment_query.append(tokenizer.convert_tokens_to_ids('?'))
                sentiment_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long().cuda()
                sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
                sentiment_query_seg += [1] * passenge.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
                sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                # 每个opinion对应一个三元组
                for idy in range(predict_opinion_num):
                    asp_f = []
                    opi_f = []
                    asp_f.append(final_asp_ind_list[idx][0])
                    asp_f.append(final_asp_ind_list[idx][1])
                    opi_f.append(final_opi_ind_list[idx][idy][0])
                    opi_f.append(final_opi_ind_list[idx][idy][1])
                    triplet_predict = asp_f + opi_f + [sentiment_predicted]
                    a_o_p_predict.append(triplet_predict)
                    if opi_f not in o_predict:
                        o_predict.append(opi_f)
                    if asp_f + opi_f not in a_o_predict:
                        a_o_predict.append(asp_f + opi_f)
                    if asp_f + [sentiment_predicted] not in asp_pol_predict:
                        asp_pol_predict.append(asp_f + [sentiment_predicted])
                    if asp_f not in a_predict:
                        a_predict.append(asp_f)
            #
            ###
            #   这里写的比较简单，之后可以考虑职业用矩阵，应该能更好一点
            ###

            triplet_target_num += len(a_o_p_target)
            asp_target_num += len(a_target)
            opi_target_num += len(o_target)
            asp_opi_target_num += len(a_o_target)
            asp_pol_target_num += len(a_p_target)

            triplet_predict_num += len(a_o_p_predict)
            a_predict_num += len(a_predict)
            o_predict_num += len(o_predict)
            a_o_predict_num += len(a_o_predict)
            asp_pol_predict_num += len(asp_pol_predict)


            # 计算每一步的最终的值，包括三元组、asp、opinion、（A,O），还有极性判断，然后根据这些计算F1
            for trip in a_o_p_target:
                for trip_ in a_o_p_predict:
                    if trip_ == trip:
                        triplet_match_num += 1
            for trip in a_target:
                for trip_ in a_predict:
                    if trip_ == trip:
                        asp_match_num += 1
            for trip in o_target:
                for trip_ in o_predict:
                    if trip_ == trip:
                        opi_match_num += 1
            for trip in a_o_target:
                for trip_ in a_o_predict:
                    if trip_ == trip:
                        asp_opi_match_num += 1
            for trip in a_p_target:
                for trip_ in asp_pol_predict:
                    if trip_ == trip:
                        asp_pol_match_num += 1
        #print(1)

        precision = float(triplet_match_num) / float(triplet_predict_num + 1e-6)
        recall = float(triplet_match_num) / float(triplet_target_num + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))

        precision_aspect = float(asp_match_num) / float(a_predict_num + 1e-6)
        recall_aspect = float(asp_match_num) / float(asp_target_num + 1e-6)
        f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect + 1e-6)
        logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

        precision_opinion = float(opi_match_num) / float(o_predict_num + 1e-6)
        recall_opinion = float(opi_match_num) / float(opi_target_num + 1e-6)
        f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion + 1e-6)
        logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

        precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num + 1e-6)
        recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num + 1e-6)
        f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
                precision_aspect_sentiment + recall_aspect_sentiment + 1e-6)
        logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                                  recall_aspect_sentiment,
                                                                                  f1_aspect_sentiment))

        precision_aspect_opinion = float(asp_opi_match_num) / float(a_o_predict_num + 1e-6)
        recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num + 1e-6)
        f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
                precision_aspect_opinion + recall_aspect_opinion + 1e-6)
        logger.info(
            'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                        f1_aspect_opinion))
        return f1
def create_directory(arguments):
    if not os.path.exists(arguments.log_path ):
        os.makedirs(arguments.log_path )
    if not os.path.exists(arguments.save_model_path ):
        os.makedirs(arguments.save_model_path )
    log_path = arguments.log_path  +arguments.model_name + '.log'
    if not os.path.exists(log_path):
        log = open(log_path, 'w')
        log.close()
    return log_path

def train(args):
    """
    :param args:只说下面用到的
        args.log_path：日志保存的地址
    :return:
    """

    # init logger and tokenize
    #log_path = args.log_path  + args.data_name + args.model_name + '.log'
    logger, fh, sh = utils.get_logger(args.log_path)
    tokenizer_1 = BertTokenizer.from_pretrained(args.bert_model_type)

    # load data
    logger.info(args)
    for padding in [False, False]:
        logger.info("####################################")
        logger.info("####################################")

        logger.info('loading data......')
        #train_data_path = args.data_path + args.data_name + '.pt'
        #test_data_path = args.data_path + args.data_name + '_test.pt'

        #train_total_data = torch.load(train_data_path)
        #test_total_data = torch.load(test_data_path)


        model = BMRC(args)
        model = model.cuda()
        data_path='./data'
        data_type='14lap'
        task_type='ASTE'



        #########
        # 这里数据读取有BUG，注意一下
        #########
        #test_dataset = SYZDataset(opt=args, data_path=data_path, data_type=data_type, dataset_type='test')


        # optimizer
        logger.info('initial optimizer......')
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "bert" not in n],
             'lr': args.learning_rate, 'weight_decay': 0.01}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate, correct_bias=False)


        # load saved model, optimizer and epoch num

        start_epoch = 1
        logger.info('New model and optimizer from epoch 1')

        # scheduler

        train_dataset = SYZDataset(opt=args, data_path=data_path, data_type=data_type, dataset_type='train',task_type=task_type)
        dev_dataset = SYZDataset(opt=args, data_path=data_path, data_type=data_type, dataset_type='dev',task_type=task_type)
        test_dataset = SYZDataset(opt=args, data_path=data_path, data_type=data_type, dataset_type='test',task_type=task_type)
        standard_data = torch.load('./data/14lap_standard.pt')

        ######
        # DEBUG
        ######
        total_data = torch.load('./data/14lap.pt')

        train_data = total_data['train']
        dev_data = total_data['dev']
        test_data = total_data['test']
        train_dataset_2 = Data2.ReviewDataset(train_data, dev_data, test_data, 'train')
        dev_dataset_2= Data2.ReviewDataset(train_data, dev_data, test_data, 'dev')
        test_dataset_2= Data2.ReviewDataset(train_data, dev_data, test_data, 'test')
        #batch_generator_train = Data2.generate_fi_batches_2(dataset=train_dataset_2, batch_size=args.batch_size)

        ######




        batch_num_train =len(train_dataset) // args.batch_size
        training_steps = args.epoch_num * batch_num_train
        warmup_steps = int(training_steps * args.warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)


        ############debug

        # training
        logger.info('begin training......')
        best_f1 = 0.



        ########debug
        standard_data = torch.load('./data/14lap_standard.pt')
        dev_standard = standard_data['dev']
        test_standard = standard_data['test']

        # #######debug
        # batch_generator_dev_2 = Data2.generate_fi_batches_2(dataset=dev_dataset_2, batch_size=1, shuffle=False)
        # logger.info("dev_debug")  #
        #
        # dev_f2 = test3(model, tokenizer_1, batch_generator_dev_2, dev_standard, 0.8,logger=logger)  # ,gpu='cuda', max_len=200,task_type=task_type)
        #
        #
        # ######
        # batch_generator_dev = Data.generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False,
        #                                             gpu=args.gpu)
        # logger.info("dev")  #
        #
        # dev_f1 = test2(model, tokenizer_1, batch_generator_dev, dev_standard, 0.8,
        #                logger)  # ,gpu='cuda', max_len=200,task_type=task_type)
        #
        #
        # # test
        # batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=False,
        #                                              gpu=args.gpu)
        # logger.info("test")
        # test_f1 = test2(model, tokenizer_1, batch_generator_test, test_standard, 0.8,
        #                 logger)  # logger,gpu='cuda', max_len=200,task_type=task_type)
        #
        # #######debug
        # batch_generator_test_2 = Data2.generate_fi_batches_2(dataset=test_dataset_2, batch_size=1, shuffle=False)
        # logger.info("test_debug")
        # test_f2 = test3(model, tokenizer_1, batch_generator_test_2, test_standard, 0.8, logger)

        _dic2={}
        # flag=True
        # for item1,item2 in zip(batch_generator_train,batch_generator_train_2):
        #     if flag is True:
        #         for name, tensor in item1.items():
        #             if name!='task_type':
        #                 _dic2[name] = item1[name].long().cuda()
        #         for name, tensor in item2.items():
        #             _dic2[name] = item2[name].long().cuda()
        #         flag=False
        #     else:


        for epoch in range(start_epoch, args.epoch_num + 1):
            logger.info("train")
            print('epoch:',epoch)
            model.train()
            model.zero_grad()
            if padding == False:
                batch_generator_train = Data.generate_batches(dataset=train_dataset, shuffle=True,
                                                              batch_size=args.batch_size)
            else:
                batch_generator_train_2 = Data2.generate_fi_batches_2(dataset=train_dataset_2, shuffle=True,
                                                                  batch_size=args.batch_size)
            # for (batch_index, item1),(i_2,item2) in zip(enumerate(batch_generator_train),enumerate(batch_generator_train_2)):
            #     print(item1['ID'])
            #     A_O_query = item1['_forward_A_O_query']
            #     A_O_query_mask = item1['_forward_A_O_query_mask']
            #     A_O_query_seg = item1['_forward_A_O_query_seg']
            #     A_O_answer_start = item1['_forward_A_O_answer_start']
            #     A_O_answer_end = item1['_forward_A_O_answer_end']
            #
            #     A_O_query_2 = item2['forward_opi_query']
            #     A_O_query_mask_2 = item2['forward_opi_query_mask']
            #     A_O_query_seg_2 = item2['forward_opi_query_seg']
            #     A_O_answer_start_2 = item2['forward_opi_answer_start']
            #     A_O_answer_end_2 = item2['forward_opi_answer_end']
            #
            #     A_O_query = F.pad(A_O_query, pad=(
            #     0, len(A_O_query_2[0][0]) - len(A_O_query[0][0]), 0, len(A_O_query_2[0]) - len(A_O_query[0])),
            #                       mode='constant', value=0)
            #     A_O_query_mask = F.pad(A_O_query_mask, pad=(
            #     0, len(A_O_query_mask_2[0][0]) - len(A_O_query_mask[0][0]), 0,
            #     len(A_O_query_mask_2[0]) - len(A_O_query_mask[0])), mode='constant', value=0)
            #     # A_O_query_seg=F.pad(A_O_query_seg, pad=(0, len(A_O_query_seg_2[0][0])-len(A_O_query_seg[0][0]),0,0),mode='constant',value=1)
            #     A_O_query_seg_r = torch.zeros_like(A_O_query_seg_2)
            #     for index_1, A_O_item in enumerate(A_O_query_seg):
            #         for index_2, A_O_item_item in enumerate(A_O_item):
            #             if (A_O_item_item != 0).sum() != 0:
            #                 A_O_query_seg_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                 0, len(A_O_query_seg_2[index_1][index_2]) - len(A_O_item_item)), mode='constant',
            #                                                           value=1)
            #             else:
            #                 A_O_query_seg_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                 0, len(A_O_query_seg_2[index_1][index_2]) - len(A_O_item_item)), mode='constant', value=0)
            #
            #     A_O_answer_start_r = torch.full(A_O_answer_start_2.size(), -1)
            #     for index_1, A_O_item in enumerate(A_O_answer_start):
            #         for index_2, A_O_item_item in enumerate(A_O_item):
            #             if (A_O_item_item != 0).sum() != 0:
            #                 A_O_answer_start_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                     0, len(A_O_answer_start_2[index_1][index_2]) - len(A_O_item_item)), mode='constant',
            #                                                              value=-1)
            #             else:
            #                 A_O_answer_start_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                     0, len(A_O_answer_start_2[index_1][index_2]) - len(A_O_item_item)), mode='constant',
            #                                                              value=-1)
            #
            #     A_O_answer_end_r = torch.full(A_O_answer_end_2.size(), -1)
            #     for index_1, A_O_item in enumerate(A_O_answer_end):
            #         for index_2, A_O_item_item in enumerate(A_O_item):
            #             if (A_O_item_item != 0).sum() != 0:
            #                 A_O_answer_end_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                     0, len(A_O_answer_end_2[index_1][index_2]) - len(A_O_item_item)), mode='constant',
            #                                                            value=-1)
            #             else:
            #                 A_O_answer_end_r[index_1][index_2] = F.pad(A_O_item_item, pad=(
            #                     0, len(A_O_answer_end_2[index_1][index_2]) - len(A_O_item_item)), mode='constant',
            #                                                            value=-1)
            #     # A_O_answer_start=F.pad(A_O_answer_start, pad=(0, len(A_O_answer_start_2[0][0])-len(A_O_answer_start[0][0]),0,len(A_O_answer_start_2[0])-len(A_O_answer_start[0])), mode='constant', value=-1)
            #     # A_O_answer_end=F.pad(A_O_answer_end, pad=(0, len(A_O_answer_end_2[0][0])-len(A_O_answer_end[0][0]),0,len(A_O_answer_end_2[0])-len(A_O_answer_end[0])), mode='constant', value=-1)
            #
            #     A_O_query_seg = A_O_query_seg_r.cuda()
            #     A_O_answer_start = A_O_answer_start_r.cuda()
            #     A_O_answer_end = A_O_answer_end_r.cuda()
            #
            #     assert 0 == ((A_O_query != A_O_query_2).sum())
            #     assert 0 == ((A_O_query_mask != A_O_query_mask_2).sum())
            #     assert 0 == ((A_O_query_seg != A_O_query_seg_2).sum())
            #     assert 0 == ((A_O_answer_start != A_O_answer_start_2).sum())
            #     assert 0 == ((A_O_answer_end != A_O_answer_end_2).sum())
            # print(1)
            # #################
            # ### 有padding版本
            # #################
            if padding is True:
                for batch_index, batch_dict in enumerate(tqdm(batch_generator_train_2,total=len(train_dataset)/args.batch_size)):

                    optimizer.zero_grad()

                    # q1_a
                    f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                                                                       batch_dict['forward_asp_query_mask'],
                                                                       batch_dict['forward_asp_query_seg'], 0)
                    f_asp_loss = utils.calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                                             batch_dict['forward_asp_answer_start'],
                                                             batch_dict['forward_asp_answer_end'])
                    # q1_b
                    b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                                 batch_dict['backward_opi_query_mask'],
                                                                 batch_dict['backward_opi_query_seg'], 0)
                    b_opi_loss = utils.calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                                             batch_dict['backward_opi_answer_start'],
                                                             batch_dict['backward_opi_answer_end'])
                    # q2_a
                    f_opi_start_scores, f_opi_end_scores = model(
                        batch_dict['forward_opi_query'].view(-1, batch_dict['forward_opi_query'].size(-1)),
                        batch_dict['forward_opi_query_mask'].view(-1, batch_dict['forward_opi_query_mask'].size(-1)),
                        batch_dict['forward_opi_query_seg'].view(-1, batch_dict['forward_opi_query_seg'].size(-1)),
                        0)
                    f_opi_loss = utils.calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                                                             batch_dict['forward_opi_answer_start'].view(-1, batch_dict[
                                                                 'forward_opi_answer_start'].size(-1)),
                                                             batch_dict['forward_opi_answer_end'].view(-1, batch_dict[
                                                                 'forward_opi_answer_end'].size(-1)))
                    # q2_b
                    b_asp_start_scores, b_asp_end_scores = model(
                        batch_dict['backward_asp_query'].view(-1, batch_dict['backward_asp_query'].size(-1)),
                        batch_dict['backward_asp_query_mask'].view(-1, batch_dict['backward_asp_query_mask'].size(-1)),
                        batch_dict['backward_asp_query_seg'].view(-1, batch_dict['backward_asp_query_seg'].size(-1)),
                        0)
                    b_asp_loss = utils.calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                                                             batch_dict['backward_asp_answer_start'].view(-1,
                                                                                                          batch_dict[
                                                                                                              'backward_asp_answer_start'].size(
                                                                                                              -1)),
                                                             batch_dict['backward_asp_answer_end'].view(-1, batch_dict[
                                                                 'backward_asp_answer_end'].size(-1)))
                    # q_3
                    sentiment_scores = model(
                        batch_dict['sentiment_query'].view(-1, batch_dict['sentiment_query'].size(-1)),
                        batch_dict['sentiment_query_mask'].view(-1, batch_dict['sentiment_query_mask'].size(-1)),
                        batch_dict['sentiment_query_seg'].view(-1, batch_dict['sentiment_query_seg'].size(-1)),
                        1)

                    sentiment_loss = utils.calculate_sentiment_loss(sentiment_scores,
                                                                    batch_dict['sentiment_answer'].view(-1))

                    # loss
                    loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + args.beta * sentiment_loss
                    loss_sum.backward()
                    optimizer.step()
                    scheduler.step()

                    # train logger
                    if batch_index % 100 == 0:
                        logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                    'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{}'.
                                    format(epoch, args.epoch_num, batch_index, batch_num_train,
                                           round(loss_sum.item(), 4),
                                           round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                                           round(b_asp_loss.item(), 4), round(b_opi_loss.item(), 4),
                                           round(sentiment_loss.item(), 4)))

        ########################
        ########  无padding 版本
        ########################
            else:
                for batch_index, batch_dict in enumerate(tqdm(batch_generator_train,total=len(train_dataset)/args.batch_size)):
                    # print(batch_index)
                    optimizer.zero_grad()
                    ###############
                    ##  不要有的加view有的不加，提前先写好数据处理（待写
                    ##############

                    S_A_start_scores, S_A_end_scores = model(batch_dict['_forward_S_A_query'],
                                                                       batch_dict['_forward_S_A_query_mask'],
                                                                       batch_dict['_forward_S_A_query_seg'], step=0)

                    S_A_loss = utils.calculate_entity_loss(S_A_start_scores, S_A_end_scores,
                                                             batch_dict['_forward_S_A_answer_start'],
                                                             batch_dict['_forward_S_A_answer_end'])

                    # S_O
                    S_O_start_scores, S_O_end_scores = model(batch_dict['_forward_S_O_query'],
                                                                       batch_dict['_forward_S_O_query_mask'],
                                                                       batch_dict['_forward_S_O_query_seg'], step=0)

                    S_O_loss = utils.calculate_entity_loss(S_O_start_scores, S_O_end_scores,
                                                             batch_dict['_forward_S_O_answer_end'], batch_dict['_forward_S_O_answer_end'])

                    # A_O
                    A_O_start_scores, A_O_end_scores = model(batch_dict['_forward_A_O_query'].view(-1, batch_dict['_forward_A_O_query'].size(-1)),
                                                                       batch_dict['_forward_A_O_query_mask'].view(-1, batch_dict['_forward_A_O_query_mask'].size(-1)),
                                                                       batch_dict['_forward_A_O_query_seg'].view(-1, batch_dict['_forward_A_O_query_seg'].size(-1)), step=0)

                    A_O_loss = utils.calculate_entity_loss(A_O_start_scores, A_O_end_scores,
                                                             batch_dict['_forward_A_O_answer_start'].view(-1, batch_dict['_forward_A_O_answer_start'].size(-1)),
                                                             batch_dict['_forward_A_O_answer_end'].view(-1, batch_dict['_forward_A_O_answer_end'].size(-1)))

                    # O_A

                    O_A_start_scores, O_A_end_scores = model(batch_dict['_forward_O_A_query'].view(-1, batch_dict['_forward_O_A_query'].size(-1)),
                                                                       batch_dict['_forward_O_A_query_mask'].view(-1, batch_dict['_forward_O_A_query_mask'].size(-1)),
                                                                       batch_dict['_forward_O_A_query_seg'].view(-1, batch_dict['_forward_O_A_query_seg'].size(-1)), step=0)

                    O_A_loss = utils.calculate_entity_loss(O_A_start_scores, O_A_end_scores,
                                                             batch_dict['_forward_O_A_answer_start'].view(-1, batch_dict['_forward_O_A_answer_start'].size(-1)),
                                                             batch_dict['_forward_O_A_answer_end'].view(-1, batch_dict['_forward_O_A_answer_end'].size(-1)))

                    # AO_P
                    AO_P_scores = model(batch_dict['_forward_AO_P_query'].view(-1, batch_dict['_forward_AO_P_query'].size(-1)),
                                                                       batch_dict['_forward_AO_P_query_mask'].view(-1, batch_dict['_forward_AO_P_query_mask'].size(-1)),
                                                                       batch_dict['_forward_AO_P_query_seg'].view(-1, batch_dict['_forward_AO_P_query_seg'].size(-1)), step='P')


                    AO_P_loss = utils.calculate_sentiment_loss(
                        AO_P_scores, batch_dict['_forward_AO_P_answer'].view(-1))
                    # loss
                    loss_sum = S_A_loss + S_O_loss + O_A_loss + A_O_loss +  args.beta * AO_P_loss
                    # 这里现在少了几个loss
                    loss_sum.backward()
                    optimizer.step()
                    scheduler.step()
                    #
                    # train logger
                    if (batch_index + 1) % 100 == 0:
                        logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                    'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{}\t'.
                                    format(epoch, args.epoch_num, batch_index + 1, batch_num_train,
                                           round(loss_sum.item(), 4),
                                           round(S_A_loss.item(), 4), round(S_O_loss.item(), 4),
                                           round(A_O_loss.item(), 4), round(O_A_loss.item(), 4),
                                           round(AO_P_loss.item(), 4)))

    ########
    #debug
    ########
                # q1_a
                # f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                #                                                    batch_dict['forward_asp_query_mask'],
                #                                                    batch_dict['forward_asp_query_seg'], 0)
                # f_asp_loss = utils.calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                #                                          batch_dict['forward_asp_answer_start'],
                #                                          batch_dict['forward_asp_answer_end'])
                # # q1_b
                # b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                #                                              batch_dict['backward_opi_query_mask'],
                #                                              batch_dict['backward_opi_query_seg'], 0)
                # b_opi_loss = utils.calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                #                                          batch_dict['backward_opi_answer_start'],
                #                                          batch_dict['backward_opi_answer_end'])
                # # q2_a
                # f_opi_start_scores, f_opi_end_scores = model(
                #     batch_dict['forward_opi_query'].view(-1, batch_dict['forward_opi_query'].size(-1)),
                #     batch_dict['forward_opi_query_mask'].view(-1, batch_dict['forward_opi_query_mask'].size(-1)),
                #     batch_dict['forward_opi_query_seg'].view(-1, batch_dict['forward_opi_query_seg'].size(-1)),
                #     0)
                # f_opi_loss = utils.calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                #                                          batch_dict['forward_opi_answer_start'].view(-1, batch_dict[
                #                                              'forward_opi_answer_start'].size(-1)),
                #                                          batch_dict['forward_opi_answer_end'].view(-1, batch_dict[
                #                                              'forward_opi_answer_end'].size(-1)))
                # # q2_b
                # b_asp_start_scores, b_asp_end_scores = model(
                #     batch_dict['backward_asp_query'].view(-1, batch_dict['backward_asp_query'].size(-1)),
                #     batch_dict['backward_asp_query_mask'].view(-1, batch_dict['backward_asp_query_mask'].size(-1)),
                #     batch_dict['backward_asp_query_seg'].view(-1, batch_dict['backward_asp_query_seg'].size(-1)),
                #     0)
                # b_asp_loss = utils.calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                #                                          batch_dict['backward_asp_answer_start'].view(-1, batch_dict[
                #                                              'backward_asp_answer_start'].size(-1)),
                #                                          batch_dict['backward_asp_answer_end'].view(-1, batch_dict[
                #                                              'backward_asp_answer_end'].size(-1)))
                # # q_3
                # sentiment_scores = model(batch_dict['sentiment_query'].view(-1, batch_dict['sentiment_query'].size(-1)),
                #                          batch_dict['sentiment_query_mask'].view(-1,
                #                                                                  batch_dict['sentiment_query_mask'].size(
                #                                                                      -1)),
                #                          batch_dict['sentiment_query_seg'].view(-1,
                #                                                                 batch_dict['sentiment_query_seg'].size(-1)),
                #                          1)
                #
                # sentiment_loss = utils.calculate_sentiment_loss(sentiment_scores, batch_dict['sentiment_answer'].view(-1))
                # # loss
                # loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + args.beta * sentiment_loss
                # loss_sum.backward()
                # optimizer.step()
                # scheduler.step()
                #
                # # train logger
                # if batch_index % 100 == 0:
                #     logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                #                 'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{}'.
                #                 format(epoch, args.epoch_num, batch_index, batch_num_train,
                #                        round(loss_sum.item(), 4),
                #                        round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                #                        round(b_asp_loss.item(), 4), round(b_opi_loss.item(), 4),
                #                        round(sentiment_loss.item(), 4)))
                ##################
                #################
                ###############
    #     ##########debug end






            # # validation& test
            #
            if True:

                batch_generator_dev = Data.generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False,
                                                            gpu=args.gpu)
                logger.info("dev") #

                dev_f1 = test2(model, tokenizer_1, batch_generator_dev,dev_standard, 0.8, logger)#,gpu='cuda', max_len=200,task_type=task_type)


                # # #######debug
                # if padding==True:
                #     batch_generator_dev_2 = Data2.generate_fi_batches_2(dataset=dev_dataset_2, batch_size=1, shuffle=False)
                #     logger.info("dev_debug") #
                #
                #     dev_f2 = test3(model, tokenizer_1, batch_generator_dev_2,dev_standard, 0.8, logger)#,gpu='cuda', max_len=200,task_type=task_type)
                # #############
                # test
                batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                             gpu=args.gpu)
                logger.info("test")
                test_f1=test2(model, tokenizer_1, batch_generator_test, test_standard, 0.8,logger)# logger,gpu='cuda', max_len=200,task_type=task_type)

                # # #######debug
                # if epoch % 4 == 0:
                #     batch_generator_test_2 = Data2.generate_fi_batches_2(dataset=test_dataset_2, batch_size=1, shuffle=False)
                #     logger.info("test_debug")
                #     test_f2 = test3(model, tokenizer_1, batch_generator_test_2, test_standard, 0.8,logger)

                # save model and optimizer
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    logger.info('Model saved after epoch {}'.format(epoch))
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    #torch.save(state, args.save_model_path)

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMRCs')

    parser.add_argument('--task_type', type=str, default='ASTE',choices=[\
                                                                         'ATE','ACD','OTE','ASC'\
                                                                         'AOPE','AESC','ACSA','ASTE'\
                                                                         'ACSD','ASQP'\
                                                                         ],help='要对哪个任务进行实验')
    parser.add_argument('--dataset_type', type=str, default='ASTE',choices=['lap14','rest14','rest15','rest16'],help='要对哪个任务进行实验')
    parser.add_argument('--data_path', type=str, default="./data",help='数据的父目录')

    # test用
    #parser.add_argument('--data_name', type=str, default='./data/rest16dev.json')
    parser.add_argument('--log_path', type=str, default="./log", help='日志保存的地点')
    parser.add_argument('--save_model_path', type=str, default="./checkpoint", help='训练的模型保存的地点')
    parser.add_argument('--model_name', type=str, default="BMRC", choices=["BMRC", "ROBMRC", "ATBMRC"], help="选择使用的模型")
    parser.add_argument('--work_nums', type=int, default=1, help="同时工作的最大工作数")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--checkpoint_path', type=str, default="./model/final_2.pth")
    # 暂时用不到的一些东西
    #parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"],help='我也不知道是啥')
    #parser.add_argument('--max_aspect_num', type=str, default="max_aspect_num", choices=["max_aspect_num"])
    #parser.add_argument('--reload', type=bool, default=False)


    parser.add_argument('--bert_model_type', type=str, default="../../bert/bert-base-uncased",help='要选用的预训练模型位置')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.80,help='对应的推理的阈值')

    # 训练过程的超参数
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--epoch_num', type=int, default=30, help='训练的次数')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--add_note', type=str, default='')# 日志的名字是否要特殊一点

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    opt = parser.parse_args()


    ########################
    ####   检验输入的各个参数是否符合规定（待写，之后再列出一个相应的表格
    ########################


    ################
    # 加载日志（待优化）
    ################
    dt = datetime.now()
    logger = utils.get_logger(opt.log_path,time=dt.strftime('%Y-%m-%d-%H-%M-%S'))
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    # 进行训练或进行预测
    if opt.mode == 'train':
        same_seeds(77)
        train(opt)
    elif opt.mode == 'test':
        logger, fh, sh = utils.get_logger(opt.log_path)

        same_seeds(77)
        logger.info('start testing......')
        # load checkpoint
        logger.info('loading checkpoint......')
        data_path = './data'
        data_type = '14lap'
        task_type = 'ASTE'
        test_dataset = SYZDataset(opt=opt, data_path=data_path, data_type=data_type, dataset_type='test',
                                  task_type=task_type)
        standard_data = torch.load('./data/14lap_standard.pt')
        dev_standard = standard_data['dev']
        test_standard = standard_data['test']
        tokenizer_1 = BertTokenizer.from_pretrained(opt.bert_model_type)

        checkpoint = torch.load(opt.checkpoint_path)
        model = BMRC(opt)
        model = model.cuda()
        model.load_state_dict(checkpoint['net'])
        model.eval()

        batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=True,
                                                        gpu=True)
        # eval
        logger.info('evaluating......')
        f1 = test2(model, tokenizer_1, batch_generator_test, test_standard, opt.beta, logger)



