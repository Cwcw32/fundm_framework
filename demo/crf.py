import copy

import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import Dataset
import torch.nn.utils.rnn as run_utils
from transformers import AutoTokenizer, BertModel, BertForPreTraining, BertConfig
from transformers import BertTokenizer, BertModel
#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
import torch
from torchcrf import CRF
from torch.utils.data import DataLoader


# 这里我们使用中文RoBERT 进行embedding
# 数据文件选择中文数据集的
# 预处理方法详情说明请参考对应目录的proceess.py文件或其目录下对应的说明文件

class BERT_CRF_Model(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass




def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


class BertModel(nn.Module):
    def __init__(self, opt):
        super(BertModel, self).__init__()
        abl_path = '../bert/'

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'trf/hfl/rbt3/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'trf/hfl/rbt3', config=self.config)
            self.bert = self.model.bert
        self.layer_drop = nn.Dropout(0.0)
        for param in self.model.parameters():
            param.requires_grad = True
        self.crf = CRF(3, batch_first=True)
        #self.output_dim = self.model.encoder.layer[11].output.dense.out_features

        self.dense = nn.Linear(768, 3)

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask,token_list,words_ids,bert_lengths,labels=None,):
        output = self.bert(input, attention_mask=attention_mask)
        text_cls = output.pooler_output
        text_encoder = output.last_hidden_state
        bert_out, bert_pooler_out = output.last_hidden_state, output.pooler_output

        bert_out = self.layer_drop(bert_out)

        # rm [CLS]
        bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(words_ids).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        # average
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)
        outputs = (bert_out,)

        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(bert_out, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs



class CommonDataset(Dataset):
    def __init__(self,data_path='../data/uni/China/mooc/',process_type=None,file_type='.txt'):
        self.tokenizer = AutoTokenizer.from_pretrained("../bert/trf/hfl/rbt3")#,cache_dir='../bert/trf')

        self.data_path = data_path
        if process_type=='train':
            self.data_path=self.data_path+'train'+file_type
        elif process_type =='dev':
            self.data_path=self.data_path+'dev'+file_type
        elif process_type =='test':
            self.data_path=self.data_path+'test'+file_type
        else:
            assert('Wrong process_type,please make it as \'train\',\'dev\',or\'test\'')
        #self.content={}
        if file_type=='.txt':
            fin = open('../data/uni/China/mooc/train.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            item_token=[]
            item_label=[]
            self.token_list_to_id=[]
            self.list_attention=[]
            self.text_list=[]
            self.label_list=[]
            self.token_list=[]      # 分词后的结果，因为用的是autotokenizer，所以是加上了[CLS]和[SEP]，这里详情请参考BERT中的内容，这里会有[CLS]和[SEP],下面处理的时候将他去掉了，如果需要请将对应内容注释
            self.words_id=[]        # 对应的是bert后的序列到原分词结果(上面的token_list)的索引
            self.bert_length=[]     # 对应的是bert后的长度（不包括CLS和其他内容）

            dic={'B-ASP':0,'I-ASP':1,'O':2}
            for n,item in enumerate(lines):
                if item=='\n':
                    max_len = 256 #bert截取的最大长度
                    token_result=self.tokenizer(item_token, truncation=True, is_split_into_words=True, max_length=max_len)
                    text_to_id=token_result['input_ids']
                    offsets=token_result.encodings[0].offsets
                    balala=max([int(b[1]) for b in offsets])
                    word_ids=token_result.encodings[0].word_ids
                    token_list=token_result.encodings[0].tokens
                    text_att=token_result['attention_mask']

                    word_ids=[b for b in word_ids if b is not None]
                    bert_length=len(word_ids) # 实际上就是bert处理之后的的长度，不包括CLS
                    token_list=token_list[1:-1] # 这里把CLS和SEP删掉，如果有[SEP]XXX[SEP]格式的话请另行处理

                    # if len(item_label)!=len(word_ids):
                    #     print(1)

                    self.text_list.append(item_token[:max_len])
                    self.label_list.append(item_label[:max_len])
                    self.token_list_to_id.append(text_to_id[:max_len])
                    self.list_attention.append(text_att[:max_len])
                    self.words_id.append(word_ids[:max_len])
                    self.token_list.append(token_list[:max_len])
                    self.bert_length.append(bert_length)

                    item_token=[]
                    item_label=[]
                    continue
                else:
                    item=item.split()
                    item_token.append(item[0])
                    item_label.append(dic[item[1]])

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        token_to_id=self.token_list_to_id[index]
        text=self.text_list[index]
        att=self.list_attention[index]
        label=self.label_list[index]
        token_list=self.token_list[index]
        words_id=self.words_id[index]
        bert_length=self.bert_length[index]
        return text,\
               token_to_id,\
               att,\
               label,\
               token_list,\
               words_id,\
               bert_length

class Collate():
    def __init__(self, opt):
        self.text_length_dynamic = 3
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.word_length
        pass
    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        #a=batch_data
        #print(1)
        text_to_id = [torch.LongTensor(b[1]) for b in batch_data]

        token_list = [b[4] for b in batch_data]
        words_ids = [torch.LongTensor(b[5]) for b in batch_data]
        bert_length = [b[6] for b in batch_data]

        label = [torch.LongTensor(b[3]) for b in batch_data]

        data_length = [text.size(0) for text in text_to_id]


        max_length = max(data_length)
        # if max_length < self.min_length:
        #     # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
        #     text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
        #     max_length = self.min_length

        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
        words_ids = run_utils.pad_sequence(words_ids, batch_first=True, padding_value=0)
        label = run_utils.pad_sequence(label, batch_first=True, padding_value=-1)

        # if label.shape[1]!=words_ids.shape[1]:
        #     print(1)

        bert_attention_mask = []
        for length in data_length:
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_length - length))
            bert_attention_mask.append(text_mask_cell[:])


        #print(1)
        return text_to_id,\
               torch.LongTensor(bert_attention_mask),\
               torch.LongTensor(label),\
               token_list,\
               words_ids,\
               torch.LongTensor(bert_length)

class OPTION():
    def __init__(self):
        self.acc_batch_size=8
        self.cuda=True
        self.text_model='bert-base'
        self.fuse_lr=0.1
        self.optim='adamw'
        self.optim_b1=0.9
        self.optim_b2=0.999
        #self.fuse_lr='bert-base'

if __name__ == '__main__':
    opt = OPTION()
    dataset_t = CommonDataset(process_type='train')

    opt.cuda=True
    data_type=1
    data_loader_t = DataLoader(dataset_t, batch_size=opt.acc_batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=1, collate_fn=Collate(opt), pin_memory=True if opt.cuda else False)

    model=BertModel(opt)

    pre_train_model_param = [name for name, param in model.named_parameters() if 'text_model' in name]

    # model.named_parameters(): [bert, classifier, crf]
    bert_optimizer = list(model.bert.named_parameters())
    classifier_optimizer = list(model.dense.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': 3e-5 * 5, 'weight_decay': 0.01},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'lr': 3e-5 * 5, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': 3e-5 * 5}
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    for i in range(0,5):
        train_losses=0
        model.train()
        for index, data in enumerate(data_loader_t):
            text_to_id, bert_attention_mask, label, token_list,\
                   words_ids,\
                   bert_length=data
            loss =model(text_to_id,bert_attention_mask,token_list,words_ids,bert_length,label)
            train_losses  = loss[0]
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            train_losses.backward()
            # gradient clipping
            #nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            #print(1)
            print(train_losses)
