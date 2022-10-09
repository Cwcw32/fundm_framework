# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4
# https://github.com/NKU-IIPLab/BMRC

from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn


class BMRC(nn.Module):
    """

    """
    def __init__(self,
                 args):
        hidden_size = args.hidden_size

        super(BMRC, self).__init__()

        if args.bert_model_type == 'bert-base-uncased':# 只是使用BERT模型
            self._bert = BertModel.from_pretrained(args.bert_model_type)
            self._tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
            print('Bertbase model loaded')
        else:
            raise KeyError('Config.args.bert_model_type should be bert-based-uncased. ')

        self.classifier_start = nn.Linear(hidden_size, 2)

        self.classifier_end = nn.Linear(hidden_size, 2)

        self._classifier_sentiment = nn.Linear(hidden_size, 3)

    def forward(self,
                query_tensor,
                query_mask,
                query_seg,
                step):
        """

        :param query_tensor:就输入BERT的就行，本任务来讲是[CLS]Question（根据情况要更改）[SEP]原句子[PADDING]...[PADDING]
        :param query_mask:
        :param query_seg:
        :param step: 用来区分是span提取还是情感分类
        :return:
        """

        hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        if step == 0:  # 预测实体（即aspect或opinion）
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        else:  # 预测情感（即sentiment）
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            return cls_hidden_scores

class RoBMRC(nn.Module):
    r"""
        可以看到这个模型和上一个主要区别就是每个任务都是使用一个独立分类器来进行判断
    """
    def __init__(self, hidden_size, bert_model_type):

        super(RoBMRC, self).__init__()

        # BERT模型
        if bert_model_type == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained(bert_model_type)
            print('bert-base-uncased model loaded')

        else:
            raise KeyError('bert_model_type should be bert-based-uncased.')

        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        self.classifier_sentiment = nn.Linear(hidden_size, 3)

    def forward(self, query_tensor, query_mask, query_seg, step):

        hidden_states = self.bert(query_tensor.long(), attention_mask=query_mask.long(), token_type_ids=query_seg.long())[0]
        if step == 'A':
            predict_start = self.classifier_a_start(hidden_states)
            predict_end = self.classifier_a_end(hidden_states)
            return predict_start, predict_end
        elif step == 'O':
            predict_start = self.classifier_o_start(hidden_states)
            predict_end = self.classifier_o_end(hidden_states)
            return predict_start, predict_end
        elif step == 'AO':
            predict_start = self.classifier_ao_start(hidden_states)
            predict_end = self.classifier_ao_end(hidden_states)
            return predict_start, predict_end
        elif step == 'OA':
            predict_start = self.classifier_oa_start(hidden_states)
            predict_end = self.classifier_oa_end(hidden_states)
            return predict_start, predict_end
        elif step == 'S':
            sentiment_hidden_states = hidden_states[:, 0, :]
            sentiment_scores = self.classifier_sentiment(sentiment_hidden_states)
            return sentiment_scores
        else:
            raise KeyError('step error.')

"""
  for batch_index, batch_dict in enumerate(batch_generator):
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
                                                         batch_dict['forward_opi_answer_start'].view(-1, batch_dict['forward_opi_answer_start'].size(-1)),
                                                         batch_dict['forward_opi_answer_end'].view(-1, batch_dict['forward_opi_answer_end'].size(-1)))
                # q2_b
                b_asp_start_scores, b_asp_end_scores = model(
                    batch_dict['backward_asp_query'].view(-1, batch_dict['backward_asp_query'].size(-1)),
                    batch_dict['backward_asp_query_mask'].view(-1, batch_dict['backward_asp_query_mask'].size(-1)),
                    batch_dict['backward_asp_query_seg'].view(-1, batch_dict['backward_asp_query_seg'].size(-1)),
                    0)
                b_asp_loss = utils.calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                                                         batch_dict['backward_asp_answer_start'].view(-1, batch_dict['backward_asp_answer_start'].size(-1)),
                                                         batch_dict['backward_asp_answer_end'].view(-1, batch_dict['backward_asp_answer_end'].size(-1)))
                # q_3
                sentiment_scores = model(batch_dict['sentiment_query'].view(-1, batch_dict['sentiment_query'].size(-1)),
                                         batch_dict['sentiment_query_mask'].view(-1, batch_dict['sentiment_query_mask'].size(-1)),
                                         batch_dict['sentiment_query_seg'].view(-1, batch_dict['sentiment_query_seg'].size(-1)),
                                         1)
                sentiment_loss = utils.calculate_sentiment_loss(sentiment_scores, batch_dict['sentiment_answer'].view(-1))

                # loss
                loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + args.beta*sentiment_loss
"""