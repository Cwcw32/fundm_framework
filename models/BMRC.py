# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn


class BMRC(nn.Module):
    def __init__(self, args):
        hidden_size = args.hidden_size

        super(BMRC, self).__init__()

        # BERT模型
        if args.bert_model_type == 'bert-base-uncased':
            self._bert = BertModel.from_pretrained(args.bert_model_type)
            self._tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
            print('Bertbase model loaded')
        else:
            raise KeyError('Config.args.bert_model_type should be bert-based-uncased. ')

        self.classifier_start = nn.Linear(hidden_size, 2)

        self.classifier_end = nn.Linear(hidden_size, 2)

        self._classifier_sentiment = nn.Linear(hidden_size, 3)

    def forward(self, query_tensor, query_mask, query_seg, step):

        hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        if step == 0:  # predict entity
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        else:  # predict sentiment
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            return cls_hidden_scores
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