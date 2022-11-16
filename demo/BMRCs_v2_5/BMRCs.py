# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4
# https://github.com/NKU-IIPLab/BMRC

from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
import torch

class BMRC_test_2_1(nn.Module):
    """

    """
    def __init__(self,
                 args):
        hidden_size = args.hidden_size

        super(BMRC, self).__init__()

        if args.bert_model_type.find('bert-base-uncased')!=-1:# 只是使用BERT模型
            self._bert = BertModel.from_pretrained('../../bert/bert-base-uncased')
            self._tokenizer = BertTokenizer.from_pretrained('../../bert/bert-base-uncased')
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

        # #for hook
        # bert_outputs = self.context_encoder(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)
        #
        # bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        #
        # bert_out = self.layer_drop(bert_out)
        #
        # # rm [CLS]
        # bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        # bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float()
        # word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        # bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        #
        # # average
        # wnt = word_mapback_one_hot.sum(dim=-1)
        # wnt.masked_fill_(wnt == 0, 1)
        # bert_out = bert_out / wnt.unsqueeze(dim=-1)


        if step == 0:  # 预测实体（即aspect或opinion）
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        else:  # 预测情感（即sentiment）
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            return cls_hidden_scores

class BMRC(nn.Module):
    """

    """
    def __init__(self,
                 args):
        hidden_size = args.hidden_size

        super(BMRC, self).__init__()

        if args.bert_model_type.find('bert-base-uncased')!=-1:# 只是使用BERT模型
            self._bert = BertModel.from_pretrained('../../bert/bert-base-uncased')
            self._tokenizer = BertTokenizer.from_pretrained('../../bert/bert-base-uncased')
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

class ROBMRC(nn.Module):
    r"""
        可以看到这个模型实际上和上一个主要区别就是每个任务都是使用一个独立分类器来进行判断
        但这种解耦可能会带来一些错误？不妨对错误进行一些研究
    """
    def __init__(self, args):

        super(ROBMRC, self).__init__()
        hidden_size = args.hidden_size
        bert_model_type=args.bert_model_type
        # BERT模型
        if bert_model_type.find('bert-base-uncased')!=-1:# 只是使用BERT模型
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
        if step == 'S_A':
            predict_start = self.classifier_a_start(hidden_states)
            predict_end = self.classifier_a_end(hidden_states)
            return predict_start, predict_end
        elif step == 'S_O':
            predict_start = self.classifier_o_start(hidden_states)
            predict_end = self.classifier_o_end(hidden_states)
            return predict_start, predict_end
        elif step == 'A_O':
            predict_start = self.classifier_ao_start(hidden_states)
            predict_end = self.classifier_ao_end(hidden_states)
            return predict_start, predict_end
        elif step == 'O_A':
            predict_start = self.classifier_oa_start(hidden_states)
            predict_end = self.classifier_oa_end(hidden_states)
            return predict_start, predict_end
        elif step == 'AO_P':
            sentiment_hidden_states = hidden_states[:, 0, :]
            sentiment_scores = self.classifier_sentiment(sentiment_hidden_states)
            return sentiment_scores
        else:
            raise KeyError('step error.')
