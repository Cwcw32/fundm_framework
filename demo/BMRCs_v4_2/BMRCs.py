# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4
# https://github.com/NKU-IIPLab/BMRC
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from  models import H_TransformerEncoder
import numpy
import torch
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F

class BMRC(nn.Module):
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

class BMRC_w_gat(nn.Module):
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

        # for gat

        self.dropout = nn.Dropout(0.1)
        args.hidden_dim=768
        self.dense = nn.Linear(args.hidden_size, args.hidden_dim)
        self.dep_embed=nn.Embedding(27, 300)

        self.graph_encoder = H_TransformerEncoder(
            d_model = args.hidden_dim,
            nhead = args.attn_head,
            num_encoder_layers = args.num_encoder_layer,
            inner_encoder_layers = args.max_num_spans,
            dropout = args.layer_dropout,
            dim_feedforward = args.bert_hidden_dim,
            activation = 'relu',
            layer_norm_eps = 1e-5
        )

    def forward(self,
                inputs,
                step):
        """
        :param inputs:
                :param query_tensor:就输入BERT的就行，本任务来讲是[CLS]Question（根据情况要更改）[SEP]原句子[PADDING]...[PADDING]
                :param query_mask: 输入的mask
                :param query_seg: 输入的seg
                :param dependency_graph 基于ASPECT/OPINION的graph
                :param bert_lengths: BERT输出之后的长度（包括了）
                :param word_mapback: BERT处理后对应的原索引
                :param bert_sequence: bert处理之后的结果
                :param dependency_graph: 基于ASPECT/OPINION的graph
                :param dependency_graph: 基于ASPECT/OPINION的graph


        :param step: 用来区分是span提取还是情感分类
        :return:最后的一个相应的分类数据
        """
        if step !='AO_P':  # 预测实体（即aspect或opinion）
            query_tensor,query_mask,query_seg=inputs
            hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        else:  # 预测情感（即sentiment）


            length, bert_lengths, word_mapback, bert_sequence,bert_mask, bert_segments_ids,aspect_indi,dependency_graph= inputs

            #dep_tag=dep_tags[:,0]

            #dep_feature = self.dep_embed(dep_tag)

            ###############################################################
            # 1. contextual encoder
            bert_outputs = self._bert(bert_sequence, attention_mask=bert_mask, token_type_ids=bert_segments_ids)

            bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output

            bert_out = self.layer_drop(bert_out)

            # rm [CLS]
            bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
            bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float()
            word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
            bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))

            # average
            wnt = word_mapback_one_hot.sum(dim=-1)
            wnt.masked_fill_(wnt == 0, 1)
            bert_out = bert_out / wnt.unsqueeze(dim=-1)


            key_padding_mask = sequence_mask(length)
            span_matrix=dependency_graph.bool()

            graph_out = self.graph_encoder(bert_out,mask=span_matrix, src_key_padding_mask=key_padding_mask)

            asp_wn = aspect_indi.sum(dim=1).unsqueeze(-1)  # aspect words num
            mask = aspect_indi.unsqueeze(-1).repeat(1, 1, self.args.hidden_size)  # mask for h
            graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn

            bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn
            as_features = torch.cat([graph_enc_outputs+bert_enc_outputs, bert_pooler_out],-1)#,dep_out], -1)

            as_features = self.dropout(as_features)
            output = self._classifier_sentiment(as_features)
            return output





def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))

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