import copy

import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import Dataset
import torch.nn.utils.rnn as run_utils
from transformers import AutoTokenizer, BertModel, BertForPreTraining, BertConfig
from transformers import BertTokenizer, BertModel
#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
import torch
from torchcrf import CRF

# 这里我们使用中文RoBERT 进行embedding
# 数据文件选择中文数据集的
# 预处理方法详情说明请参考对应目录的proceess.py文件或其目录下对应的说明文件

class BERT_CRF_Model(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass





class BertModel(nn.Module):
    def __init__(self, opt):
        super(BertModel, self).__init__()
        abl_path = './bert/'

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'trf/hfl/rbt3/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'trf/hfl/rbt3', config=self.config)
            self.bert = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask,labels=None):
        output = self.bert(input, attention_mask=attention_mask)
        text_cls = output.pooler_output
        text_encoder = output.last_hidden_state
        if labels is not None:
            loss_mask = labels.gt(-1)
            # loss = self.crf(logits, labels, loss_mask) * (-1)
            # outputs = (loss,) + outputs
        return output



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
            dic={'B-ASP':0,'I-ASP':1,'O':2}
            for n,item in enumerate(lines):
                if item=='\n':
                    max_len = 64 #bert截取的最大长度
                    token_result=self.tokenizer(item_token, truncation=True, is_split_into_words=True, max_length=max_len)
                    text_to_id=token_result['input_ids']
                    text_att=token_result['attention_mask']
                    self.text_list.append(item_token)
                    self.label_list.append(item_label)
                    self.token_list_to_id.append(text_to_id)
                    self.list_attention.append(text_att)
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
        token=self.token_list_to_id[index]
        text=self.text_list[index]
        att=self.list_attention[index]
        label=self.label_list[index]

        return text,token,att,label

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


        label = [torch.LongTensor(b[3]) for b in batch_data]

        data_length = [text.size(0) for text in text_to_id]


        max_length = max(data_length)
        # if max_length < self.min_length:
        #     # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
        #     text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
        #     max_length = self.min_length

        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
        bert_attention_mask = []
        for length in data_length:
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_length - length))
            bert_attention_mask.append(text_mask_cell[:])

        return text_to_id, torch.LongTensor(bert_attention_mask), label



from torch.utils.data import DataLoader
class OPTION():
    def __init__(self):
        self.acc_batch_size=2
        self.cuda=True
        self.text_model='bert-base'
        #self.fuse_lr='bert-base'
        self.optim='adamw'
        self.fuse_lr=0.9
        self.fuse_lr=0.999
        self.fuse_lr='bert-base'

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
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in pre_train_model_param],
            "lr": 0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in pre_train_model_param],
            "lr": opt.fuse_lr,
        },
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)



    for index, data in enumerate(data_loader_t):
        text_to_id, bert_attention_mask, label=data
        output=model(text_to_id,bert_attention_mask,label)

        print(1)
