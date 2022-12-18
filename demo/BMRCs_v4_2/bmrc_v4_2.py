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


from BMRCs import BMRC, ROBMRC
from data_process import SYZDataset
import DatasetCapsulation as Data
import utils

def test_ROBMRC(model, tokenize, batch_generator, test_data, beta, logger, gpu, max_len):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_sent_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_sent_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_sent_match_num = 0

    for batch_index, batch_dict in enumerate(tqdm(batch_generator,total=len(test_data))):

        triplets_target = test_data[batch_index]['as_op_po']
        asp_target = test_data[batch_index]['aspects']
        opi_target = test_data[batch_index]['opinions']
        asp_opi_target = test_data[batch_index]['as_op']
        asp_sent_target = test_data[batch_index]['as_po']

        # 预测三元组
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_sent_predict = []

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


        batch_dict['forward_asp_answer_start']= batch_dict['_forward_S_A_answer_start'].cuda()
        batch_dict['forward_asp_query']= batch_dict['_forward_S_A_query'].cuda()
        batch_dict['forward_asp_query_mask']= batch_dict['_forward_S_A_query_mask'].cuda()
        batch_dict['forward_asp_query_seg']= batch_dict['_forward_S_A_query_seg'].cuda()

        ok_start_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero(as_tuple=False)

        ok_start_tokens = batch_dict['forward_asp_query'][0][ok_start_index].squeeze(1)




        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 'S_A')

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)

        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []
        for start_index in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, start_index] != -1:
                if f_asp_start_ind[start_index].item() == 1:
                    f_asp_start_index_temp.append(start_index)
                    f_asp_start_prob_temp.append(f_asp_start_prob[start_index].item())
                if f_asp_end_ind[start_index].item() == 1:
                    f_asp_end_index_temp.append(start_index)
                    f_asp_end_prob_temp.append(f_asp_end_prob[start_index].item())

        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired_robmrc(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, max_len)

        # 根据预测到的切面，生成查询
        for start_index in range(len(f_asp_start_index)):
            opinion_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1):
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(tokenize.convert_tokens_to_ids('?'))
            opinion_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query)

            opinion_query = torch.tensor(opinion_query).long()
            if gpu:
                opinion_query = opinion_query.cuda()
            opinion_query = torch.cat([opinion_query, ok_start_tokens], -1).unsqueeze(0)
            opinion_query_seg += [1] * ok_start_tokens.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().unsqueeze(0)
            if gpu:
                opinion_query_mask = opinion_query_mask.cuda()
            opinion_query_seg = torch.tensor(opinion_query_seg).long().unsqueeze(0)
            if gpu:
                opinion_query_seg = opinion_query_seg.cuda()

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 'A_O')

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

            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired_robmrc(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, max_len)

            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in
                       range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[start_index] - 5, f_asp_end_index[start_index] - 5]
                opi_ind = [f_opi_start_index[idx] - f_opi_length, f_opi_end_index[idx] - f_opi_length]
                temp_prob = math.sqrt(f_asp_prob[start_index] * f_opi_prob[idx])
                if asp_ind + opi_ind not in forward_pair_list:
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)



        batch_dict['backward_opi_answer_start']= batch_dict['_forward_S_O_answer_start'].cuda()
        batch_dict['backward_opi_query'] = batch_dict['_forward_S_O_query'].cuda()
        batch_dict['backward_opi_query_mask'] = batch_dict['_forward_S_O_query_mask'].cuda()
        batch_dict['backward_opi_query_seg'] = batch_dict['_forward_S_O_query_seg'].cuda()



        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 'S_O')
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for start_index in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, start_index] != -1:
                if b_opi_start_ind[start_index].item() == 1:
                    b_opi_start_index_temp.append(start_index)
                    b_opi_start_prob_temp.append(b_opi_start_prob[start_index].item())
                if b_opi_end_ind[start_index].item() == 1:
                    b_opi_end_index_temp.append(start_index)
                    b_opi_end_prob_temp.append(b_opi_end_prob[start_index].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired_robmrc(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp, max_len)

        for start_index in range(len(b_opi_start_index)):
            aspect_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(tokenize.convert_tokens_to_ids('describe'))
            aspect_query.append(tokenize.convert_tokens_to_ids('?'))
            aspect_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long()
            if gpu:
                aspect_query = aspect_query.cuda()
            aspect_query = torch.cat([aspect_query, ok_start_tokens], -1).unsqueeze(0)
            aspect_query_seg += [1] * ok_start_tokens.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().unsqueeze(0)
            if gpu:
                aspect_query_mask = aspect_query_mask.cuda()
            aspect_query_seg = torch.tensor(aspect_query_seg).long().unsqueeze(0)
            if gpu:
                aspect_query_seg = aspect_query_seg.cuda()

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 'O_A')

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

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired_robmrc(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, max_len)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                opi_ind = [b_opi_start_index[start_index] - 5, b_opi_end_index[start_index] - 5]
                temp_prob = math.sqrt(b_asp_prob[idx] * b_opi_prob[start_index])
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list or forward_pair_prob[idx] >= beta:
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
            for idy in range(predict_opinion_num):
                sentiment_query = tokenize.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What sentiment given the aspect'.split(' ')])
                sentiment_query += final_asp_list[idx]
                sentiment_query += tokenize.convert_tokens_to_ids(
                    [word.lower() for word in 'and the opinion'.split(' ')])
                sentiment_query += final_opi_list[idx][idy]
                sentiment_query.append(tokenize.convert_tokens_to_ids('?'))
                sentiment_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long()
                if gpu:
                    sentiment_query = sentiment_query.cuda()
                sentiment_query = torch.cat([sentiment_query, ok_start_tokens], -1).unsqueeze(0)
                sentiment_query_seg += [1] * ok_start_tokens.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().unsqueeze(0)
                if gpu:
                    sentiment_query_mask = sentiment_query_mask.cuda()
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().unsqueeze(0)
                if gpu:
                    sentiment_query_seg = sentiment_query_seg.cuda()

                sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 'AO_P')
                sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                triplet_predict = asp_f + opi_f + [sentiment_predicted]
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_sent_predict:
                    asp_sent_predict.append(asp_f + [sentiment_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)
                if triplet_predict not in triplets_predict:
                    triplets_predict.append(triplet_predict)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_sent_target_num += len(asp_sent_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_sent_predict_num += len(asp_sent_predict)

        for trip in triplets_predict:
            for trip_ in triplets_target:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_predict:
            for trip_ in asp_target:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_predict:
            for trip_ in opi_target:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_predict:
            for trip_ in asp_opi_target:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_sent_predict:
            for trip_ in asp_sent_target:
                if trip_ == trip:
                    asp_sent_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num + 1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))

    precision_aspect = float(asp_match_num) / float(asp_predict_num + 1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num + 1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect + 1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num + 1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num + 1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion + 1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_sent_match_num) / float(asp_sent_predict_num + 1e-6)
    recall_aspect_sentiment = float(asp_sent_match_num) / float(asp_sent_target_num + 1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment + 1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num + 1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num + 1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion + 1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1

def test_ROBMRC_2(model, t, batch_generator, standard, beta, logger):
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
        batch_dict['forward_asp_answer_start']= batch_dict['_forward_S_A_answer_start'].cuda()
        batch_dict['forward_asp_query']= batch_dict['_forward_S_A_query'].cuda()
        batch_dict['forward_asp_query_mask']= batch_dict['_forward_S_A_query_mask'].cuda()
        batch_dict['forward_asp_query_seg']= batch_dict['_forward_S_A_query_seg'].cuda()

        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero(as_tuple=False)
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


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired_robmrc_ROBMRC(f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, max_len=200)

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
            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired_robmrc_ROBMRC(f_opi_start_prob_temp,
                                                                                          f_opi_end_prob_temp,
                                                                                          f_opi_start_index_temp,
                                                                                          f_opi_end_index_temp,
                                                                                          max_len=200)

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
        batch_dict['backward_opi_answer_start']= batch_dict['_forward_S_O_answer_start'].cuda()
        batch_dict['backward_opi_query'] = batch_dict['_forward_S_O_query'].cuda()
        batch_dict['backward_opi_query_mask'] = batch_dict['_forward_S_O_query_mask'].cuda()
        batch_dict['backward_opi_query_seg'] = batch_dict['_forward_S_O_query_seg'].cuda()

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

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired_robmrc_ROBMRC(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,max_len=200)

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

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired_robmrc_ROBMRC(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, max_len=200)

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

def test_bmrc(model, t, batch_generator, standard, beta, logger):
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

        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero(as_tuple=False)
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


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired_robmrc(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,max_len=200)
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
            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired_robmrc(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp,max_len=200)


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

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired_robmrc(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,max_len=200)

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

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired_robmrc(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp,max_len=200)

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
        ####################
        # re_think
        ###################

        ################
        ###### 统计 最终结果
        ################

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
    logger.info("####################################")
    logger.info("####################################")
    logger.info('loading data......')

    if opt.model_name=='BMRC':
        model = BMRC(args)
    elif opt.model_name=='ROBMRC':
        model= ROBMRC(args)
    else:
        raise KeyError('Wrong model name.')


    if opt.gpu is True:
        model = model.cuda()

    data_path='./data'
    data_type='14lap'
    task_type='ASTE'

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

    batch_num_train =len(train_dataset) // args.batch_size
    training_steps = args.epoch_num * batch_num_train
    warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)


    # training
    logger.info('begin training......')
    best_f1 = 0.



    ########debug
    standard_data = torch.load('./data/14lap_standard.pt')
    dev_standard = standard_data['dev']
    test_standard = standard_data['test']

    _dic2={}


    BMRC_step=[0,0,0,0,'P']
    ROBMRC_step=['S_A','A_O','S_O','O_A','AO_P']

    if opt.model_name=='BMRC':
        used_step=BMRC_step
    elif opt.model_name=='ROBMRC':
        used_step=ROBMRC_step
    else:
        print('Wrong model_name')
        exit(1)

    # batch_generator_dev = Data.generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False, gpu=args.gpu)
    # logger.info("dev")  #
    # if opt.model_name == 'BMRC':
    #     dev_f1 = test_bmrc(model, tokenizer_1, batch_generator_dev, dev_standard, 0.8,
    #                        logger)  # ,gpu='cuda', max_len=200,task_type=task_type)
    # elif opt.model_name == 'ROBMRC':
    #     dev_f1 = test_ROBMRC(model, tokenizer_1, batch_generator_dev, dev_dataset.standard_2, 0.9, logger, gpu=True,
    #                          max_len=200)  # task_type=task_type)

    for epoch in range(start_epoch, args.epoch_num + 1):
        logger.info("train")
        print('epoch:',epoch)
        model.train()
        model.zero_grad()

        batch_generator_train = Data.generate_batches(dataset=train_dataset, shuffle=True,
                                                          batch_size=args.batch_size)


        for batch_index, batch_dict in enumerate(tqdm(batch_generator_train,total=len(train_dataset)/args.batch_size)):
            # print(batch_index)
            optimizer.zero_grad()
            ###############
            ##  不要有的加view有的不加，提前先写好数据处理（待写
            ##############
            S_A_start_scores, S_A_end_scores = model(batch_dict['_forward_S_A_query'],
                                                               batch_dict['_forward_S_A_query_mask'],
                                                               batch_dict['_forward_S_A_query_seg'], step=used_step[0])

            S_A_loss = utils.calculate_entity_loss(S_A_start_scores, S_A_end_scores,
                                                     batch_dict['_forward_S_A_answer_start'],
                                                     batch_dict['_forward_S_A_answer_end'])/6

            # S_O
            S_O_start_scores, S_O_end_scores = model(batch_dict['_forward_S_O_query'],
                                                               batch_dict['_forward_S_O_query_mask'],
                                                               batch_dict['_forward_S_O_query_seg'], step=used_step[1])

            S_O_loss = utils.calculate_entity_loss(S_O_start_scores, S_O_end_scores,
                                                     batch_dict['_forward_S_O_answer_start'], batch_dict['_forward_S_O_answer_end'])/6

            # A_O
            A_O_start_scores, A_O_end_scores = model(batch_dict['_forward_A_O_query'].view(-1, batch_dict['_forward_A_O_query'].size(-1)),
                                                               batch_dict['_forward_A_O_query_mask'].view(-1, batch_dict['_forward_A_O_query_mask'].size(-1)),
                                                               batch_dict['_forward_A_O_query_seg'].view(-1, batch_dict['_forward_A_O_query_seg'].size(-1)), step=used_step[2])

            A_O_loss = utils.calculate_entity_loss(A_O_start_scores, A_O_end_scores,
                                                     batch_dict['_forward_A_O_answer_start'].view(-1, batch_dict['_forward_A_O_answer_start'].size(-1)),
                                                     batch_dict['_forward_A_O_answer_end'].view(-1, batch_dict['_forward_A_O_answer_end'].size(-1)))/6

            # O_A

            O_A_start_scores, O_A_end_scores = model(batch_dict['_forward_O_A_query'].view(-1, batch_dict['_forward_O_A_query'].size(-1)),
                                                               batch_dict['_forward_O_A_query_mask'].view(-1, batch_dict['_forward_O_A_query_mask'].size(-1)),
                                                               batch_dict['_forward_O_A_query_seg'].view(-1, batch_dict['_forward_O_A_query_seg'].size(-1)), step=used_step[3])

            O_A_loss = utils.calculate_entity_loss(O_A_start_scores, O_A_end_scores,
                                                     batch_dict['_forward_O_A_answer_start'].view(-1, batch_dict['_forward_O_A_answer_start'].size(-1)),
                                                     batch_dict['_forward_O_A_answer_end'].view(-1, batch_dict['_forward_O_A_answer_end'].size(-1)))/6

            # AO_P
            if opt.gat is False:
                AO_P_scores = model(batch_dict['_forward_AO_P_query'].view(-1, batch_dict['_forward_AO_P_query'].size(-1)),
                                                                   batch_dict['_forward_AO_P_query_mask'].view(-1, batch_dict['_forward_AO_P_query_mask'].size(-1)),
                                                                   batch_dict['_forward_AO_P_query_seg'].view(-1, batch_dict['_forward_AO_P_query_seg'].size(-1)), step=used_step[4])


                AO_P_loss = utils.calculate_sentiment_loss(
                    AO_P_scores, batch_dict['_forward_AO_P_answer'].view(-1))/6
            else:
                pass

            if opt.train_rt is True:
                # P_A
                # P_A_start_scores, P_A_end_scores = model(batch_dict['_forward_P_A_query'].view(-1, batch_dict['_forward_P_A_query'].size(-1)),
                #                                                    batch_dict['_forward_P_A_query_mask'].view(-1, batch_dict['_forward_P_A_query_mask'].size(-1)),
                #                                                    batch_dict['_forward_P_A_query_seg'].view(-1, batch_dict['_forward_P_A_query_seg'].size(-1)), step=used_step[2])
                #
                # P_A_loss = utils.calculate_entity_loss(P_A_start_scores, P_A_end_scores,
                #                                          batch_dict['_forward_P_A_answer_start'].view(-1, batch_dict['_forward_P_A_answer_start'].size(-1)),
                #                                          batch_dict['_forward_P_A_answer_end'].view(-1, batch_dict['_forward_P_A_answer_end'].size(-1)))
                #
                # # P_O
                # P_O_start_scores, P_O_end_scores = model(batch_dict['_forward_P_O_query'].view(-1, batch_dict['_forward_P_O_query'].size(-1)),
                #                                                    batch_dict['_forward_P_O_query_mask'].view(-1, batch_dict['_forward_P_O_query_mask'].size(-1)),
                #                                                    batch_dict['_forward_P_O_query_seg'].view(-1, batch_dict['_forward_P_O_query_seg'].size(-1)), step=used_step[2])
                #
                # P_O_loss = utils.calculate_entity_loss(P_O_start_scores, P_O_end_scores,
                #                                          batch_dict['_forward_P_O_answer_start'].view(-1, batch_dict['_forward_P_O_answer_start'].size(-1)),
                #                                          batch_dict['_forward_P_O_answer_end'].view(-1, batch_dict['_forward_P_O_answer_end'].size(-1)))

                # # PA_O
                PA_O_start_scores, PA_O_end_scores = model(batch_dict['_forward_PA_O_query'].view(-1, batch_dict['_forward_PA_O_query'].size(-1)),
                                                                   batch_dict['_forward_PA_O_query_mask'].view(-1, batch_dict['_forward_PA_O_query_mask'].size(-1)),
                                                                   batch_dict['_forward_PA_O_query_seg'].view(-1, batch_dict['_forward_PA_O_query_seg'].size(-1)), step=used_step[2])

                PA_O_loss = utils.calculate_entity_loss(PA_O_start_scores, PA_O_end_scores,
                                                         batch_dict['_forward_PA_O_answer_start'].view(-1, batch_dict['_forward_PA_O_answer_start'].size(-1)),
                                                         batch_dict['_forward_PA_O_answer_end'].view(-1, batch_dict['_forward_PA_O_answer_end'].size(-1)))/6
                # # PO_A
                PO_A_start_scores, PO_A_end_scores = model(batch_dict['_forward_PO_A_query'].view(-1, batch_dict['_forward_PO_A_query'].size(-1)),
                                                                   batch_dict['_forward_PO_A_query_mask'].view(-1, batch_dict['_forward_PO_A_query_mask'].size(-1)),
                                                                   batch_dict['_forward_PO_A_query_seg'].view(-1, batch_dict['_forward_PO_A_query_seg'].size(-1)), step=used_step[2])

                PO_A_loss = utils.calculate_entity_loss(PO_A_start_scores, PO_A_end_scores,
                                                         batch_dict['_forward_PO_A_answer_start'].view(-1, batch_dict['_forward_PO_A_answer_start'].size(-1)),
                                                         batch_dict['_forward_PO_A_answer_end'].view(-1, batch_dict['_forward_PO_A_answer_end'].size(-1)))/6


                # loss
                loss_sum = S_A_loss + S_O_loss + O_A_loss + A_O_loss + PA_O_loss + PO_A_loss+ args.beta * AO_P_loss
                #loss_sum = S_A_loss + S_O_loss + O_A_loss + A_O_loss + P_A_loss + P_O_loss + PA_O_loss+ PO_A_loss + args.beta * AO_P_loss
                loss_sum.backward()
                optimizer.step()
                scheduler.step()
                #
                # train logger
                if (batch_index + 1) % 10 == 0:
                    logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                'S_A Loss:{};S_O Loss:{}\t A_O Loss:{};O_A Loss:{}\t AO_P Loss:{}\t;\t PA_O Loss:{}\t;\t PO_A Loss:{}\t;'.
                                format(epoch, args.epoch_num, batch_index + 1, batch_num_train,
                                       round(loss_sum.item(), 4),
                                       round(S_A_loss.item(), 4), round(S_O_loss.item(), 4),
                                       round(A_O_loss.item(), 4), round(O_A_loss.item(), 4),
                                       round(AO_P_loss.item(), 4), round(PA_O_loss.item(), 4),
                                       round(PO_A_loss.item(), 4)))
            else:
                loss_sum = S_A_loss + S_O_loss + O_A_loss + A_O_loss + args.beta * AO_P_loss
                loss_sum.backward()
                optimizer.step()
                scheduler.step()
                #
                # train logger
                if (batch_index + 1) % 10 == 0:
                    logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{}\t'.
                                format(epoch, args.epoch_num, batch_index + 1, batch_num_train,
                                       round(loss_sum.item(), 4),
                                       round(S_A_loss.item(), 4), round(S_O_loss.item(), 4),
                                       round(A_O_loss.item(), 4), round(O_A_loss.item(), 4),
                                       round(AO_P_loss.item(), 4)))



        if True:
            batch_generator_dev = Data.generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False,gpu=args.gpu)
            logger.info("dev") #
            if opt.model_name=='BMRC':
                dev_f1 = test_bmrc(model, tokenizer_1, batch_generator_dev,dev_standard, 0.8, logger)#,gpu='cuda', max_len=200,task_type=task_type)
            elif opt.model_name=='ROBMRC':
                dev_f1 = test_ROBMRC(model, tokenizer_1, batch_generator_dev,dev_dataset.standard_2, 0.9, logger,gpu=True, max_len=200)#task_type=task_type)


            # test
            batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                         gpu=args.gpu)
            logger.info("test")
            if opt.model_name=='BMRC':
                test_f1 = test_bmrc(model, tokenizer_1, batch_generator_test, test_standard, 0.8,
                                    logger)  # logger,gpu='cuda', max_len=200,task_type=task_type)
            elif opt.model_name=='ROBMRC':
                test_f1 = test_ROBMRC(model, tokenizer_1, batch_generator_test, test_dataset.standard, 0.9,
                                    logger,gpu=True, max_len=200)#,task_type=task_type)

            # save model and optimizer
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                logger.info('Model saved after epoch {}'.format(epoch))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, args.save_model_path)


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


#########################
####### 主函数
########################
if __name__ == '__main__':

    ##################
    ##### 参数设置
    ##################
    parser = argparse.ArgumentParser(description='BMRCs')
    parser.add_argument('--task_type', type=str, default='ASTE',choices=[\
                                                                         'ATE','ACD','OTE','ASC'\
                                                                         'AOPE','AESC','ACSA','ASTE'\
                                                                         'ACSD','ASQP'\
                                                                         ],help='要对哪个任务进行实验')
    parser.add_argument('--dataset_type', type=str, default='ASTE',choices=['lap14','rest14','rest15','rest16'],help='要对哪个任务进行实验')
    parser.add_argument('--data_path', type=str, default="./data",help='数据的父目录')
    #parser.add_argument('--data_name', type=str, default='./data/rest16dev.json')
    parser.add_argument('--log_path', type=str, default="./log", help='日志保存的地点')
    parser.add_argument('--save_model_path', type=str, default="./checkpoint", help='训练的模型保存的地点')
    parser.add_argument('--model_name', type=str, default="BMRC", choices=["BMRC", "ROBMRC", "ATBMRC"], help="选择使用的模型")
    parser.add_argument('--work_nums', type=int, default=1, help="同时工作的最大工作数")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--checkpoint_path', type=str, default="./model/final_2.pth")
    parser.add_argument('--zhongzi', type=list, default=[5,77,89,32,66])# 5个种子
    # 暂时用不到的一些东西
    #parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"],help='我也不知道是啥')
    #parser.add_argument('--max_aspect_num', type=str, default="max_aspect_num", choices=["max_aspect_num"])
    #parser.add_argument('--reload', type=bool, default=False)


    parser.add_argument('--bert_model_type', type=str, default="../../bert/bert-base-uncased",help='要选用的预训练模型位置')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.80,help='对应的推理的阈值')

    # 训练过程的超参数
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--epoch_num', type=int, default=40, help='训练的次数')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--add_note', type=str, default='')# 日志的名字是否要特殊一点

    # some setting
    parser.add_argument('--nizhuan', type=bool, default=False)    # [CLS] + context +[SEP] +question FOR TRUE WHILE CLS] + question +[SEP] +context for False
    parser.add_argument('--gat', type=bool, default=False)       # 是否用gat对bert的输出做处理
    parser.add_argument('--train_rt', type=bool, default=False)  # 训练过程的rethinking
    parser.add_argument('--infer_rt', type=bool, default=False)  # 推理过程的rethinking

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0) # 设置GPU
    opt = parser.parse_args()

    ###################
    #####debug setting
    ##################
    opt.train_rt=True



    ########################XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ####   检验输入的各个参数是否符合规定（待写，之后再列出一个相应的表格
    ########################XXXXXXXXXXXXXXXXXXXXXXXXXXX




    ################XXXXXXXXXXXXX
    # 加载日志（待优化）
    ################XXXXXXXXXXXXX



    #########################
    ##### 进行训练或预测
    #######################
    if opt.mode == 'train':
        for seed in opt.zhongzi:
            dt = datetime.now()
            logger = utils.get_logger(opt.log_path, time=dt.strftime('%Y-%m-%d-%H-%M-%S'))
            opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
                '%Y-%m-%d-%H-%M-%S') + '-'
            if opt.add_note != '':
                opt.save_model_path += opt.add_note + '-'
            print('\n', opt.save_model_path, '\n')
            same_seeds(seed)
            train(opt)
    elif opt.mode == 'test':
        dt = datetime.now()
        logger = utils.get_logger(opt.log_path, time=dt.strftime('%Y-%m-%d-%H-%M-%S'))
        opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
            '%Y-%m-%d-%H-%M-%S') + '-'
        if opt.add_note != '':
            opt.save_model_path += opt.add_note + '-'
        print('\n', opt.save_model_path, '\n')

        logger, fh, sh = utils.get_logger(opt.log_path)
        same_seeds(77)
        logger.info('start testing......')
        # load checkpoint
        logger.info('loading checkpoint......')
        data_path = './data'
        data_type = '14lap'
        task_type = 'ASTE'
        test_dataset = SYZDataset(opt=opt, data_path=data_path, data_type=data_type, dataset_type='test',task_type=task_type)
        standard_data = torch.load('./data/14lap_standard.pt')
        dev_standard = standard_data['dev']
        test_standard = standard_data['test']
        tokenizer_1 = BertTokenizer.from_pretrained(opt.bert_model_type)

        checkpoint = torch.load(opt.checkpoint_path)
        model = BMRC(opt)
        model = model.cuda()
        model.load_state_dict(checkpoint['net'])
        model.eval()

        batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=True,gpu=True)
        # eval
        logger.info('evaluating......')
        f1 = test_bmrc(model, tokenizer_1, batch_generator_test, test_standard, opt.beta, logger)



