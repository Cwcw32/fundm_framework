import copy
import re

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections.abc
string_classes = (str, bytes)
int_classes = int
container_abcs = collections.abc

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

class QueryAndAnswer:
    def __init__(self, line, forward_asp_query, forward_opi_query,
                 forward_asp_query_mask, forward_asp_query_seg,
                 forward_opi_query_mask, forward_opi_query_seg,
                 forward_asp_answer_start, forward_asp_answer_end,
                 forward_opi_answer_start, forward_opi_answer_end,
                 backward_asp_query, backward_opi_query,
                 backward_asp_answer_start, backward_asp_answer_end,
                 backward_asp_query_mask, backward_asp_query_seg,
                 backward_opi_query_mask, backward_opi_query_seg,
                 backward_opi_answer_start, backward_opi_answer_end,
                 sentiment_query, sentiment_answer,
                 sentiment_query_mask, sentiment_query_seg):
        self.line = line
        self.forward_asp_query = forward_asp_query
        self.forward_opi_query = forward_opi_query
        self.forward_asp_query_mask = forward_asp_query_mask
        self.forward_asp_query_seg = forward_asp_query_seg
        self.forward_opi_query_mask = forward_opi_query_mask
        self.forward_opi_query_seg = forward_opi_query_seg
        self.forward_asp_answer_start = forward_asp_answer_start
        self.forward_asp_answer_end = forward_asp_answer_end
        self.forward_opi_answer_start = forward_opi_answer_start
        self.forward_opi_answer_end = forward_opi_answer_end
        self.backward_asp_query = backward_asp_query
        self.backward_opi_query = backward_opi_query
        self.backward_asp_query_mask = backward_asp_query_mask
        self.backward_asp_query_seg = backward_asp_query_seg
        self.backward_opi_query_mask = backward_opi_query_mask
        self.backward_opi_query_seg = backward_opi_query_seg
        self.backward_asp_answer_start = backward_asp_answer_start
        self.backward_asp_answer_end = backward_asp_answer_end
        self.backward_opi_answer_start = backward_opi_answer_start
        self.backward_opi_answer_end = backward_opi_answer_end
        self.sentiment_query = sentiment_query
        self.sentiment_answer = sentiment_answer
        self.sentiment_query_mask = sentiment_query_mask
        self.sentiment_query_seg = sentiment_query_seg


class TestDataset:
    def __init__(self, line, aspect_list, opinion_list, asp_opi_list, asp_sent_list, triplet_list):
        self.line = line
        self.aspect_list = aspect_list
        self.opinion_list = opinion_list
        self.asp_opi_list = asp_opi_list
        self.asp_sent_list = asp_sent_list
        self.triplet_list = triplet_list


class ReviewDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dataset_to_numpy_array = {
            'line': self.dataset[item].line,
            'forward_asp_query': np.array(self.dataset[item].forward_asp_query),
            'forward_opi_query': np.array(self.dataset[item].forward_opi_query),
            'forward_asp_query_mask': np.array(self.dataset[item].forward_asp_query_mask),
            'forward_opi_query_mask': np.array(self.dataset[item].forward_opi_query_mask),
            'forward_asp_query_seg': np.array(self.dataset[item].forward_asp_query_seg),
            'forward_opi_query_seg': np.array(self.dataset[item].forward_opi_query_seg),
            'forward_asp_answer_start': np.array(self.dataset[item].forward_asp_answer_start),
            'forward_asp_answer_end': np.array(self.dataset[item].forward_asp_answer_end),
            'forward_opi_answer_start': np.array(self.dataset[item].forward_opi_answer_start),
            'forward_opi_answer_end': np.array(self.dataset[item].forward_opi_answer_end),
            'backward_asp_query': np.array(self.dataset[item].backward_asp_query),
            'backward_opi_query': np.array(self.dataset[item].backward_opi_query),
            'backward_asp_query_mask': np.array(self.dataset[item].backward_asp_query_mask),
            'backward_opi_query_mask': np.array(self.dataset[item].backward_opi_query_mask),
            'backward_asp_query_seg': np.array(self.dataset[item].backward_asp_query_seg),
            'backward_opi_query_seg': np.array(self.dataset[item].backward_opi_query_seg),
            'backward_asp_answer_start': np.array(self.dataset[item].backward_asp_answer_start),
            'backward_asp_answer_end': np.array(self.dataset[item].backward_asp_answer_end),
            'backward_opi_answer_start': np.array(self.dataset[item].backward_opi_answer_start),
            'backward_opi_answer_end': np.array(self.dataset[item].backward_opi_answer_end),
            'sentiment_query': np.array(self.dataset[item].sentiment_query),
            'sentiment_answer': np.array(self.dataset[item].sentiment_answer),
            'sentiment_query_mask': np.array(self.dataset[item].sentiment_query_mask),
            'sentiment_query_seg': np.array(self.dataset[item].sentiment_query_seg)
        }
        return dataset_to_numpy_array

    def get_batch_num(self, batch_size):
        if len(self.dataset) % batch_size == 0:
            return len(self.dataset) / batch_size
        return int(len(self.dataset) / batch_size) + 1

class Collate3(): #
    def __init__(self, opt=None):

        pass
    def __call__(self, batch_data):
        # 把所有大小统一一下
        _batch_data=[]
        for num in range(len(batch_data)):
            _batch_data.append(copy.deepcopy({}))
        for key in batch_data[0]:
            temp_data=[{key:d[key]} for d in batch_data]
            max_len_1=0
            max_len_2=0
            len_dic=[0,0]
            for tem_n,temp_dic in enumerate(temp_data):
                temp_item=temp_dic[key]
                #print(1)
                if isinstance(temp_item, int_classes):
                    pass
                    #print('ID card:',temp_item)
                elif len(temp_item)==0:
                    # 为空的就先不考虑了
                    continue
                elif key.find('answer')!=-1 and key.find('answer_start')==-1 and key.find('answer_end')==-1: # answer不是提取任务，是分类任务，即对应的list是[a,b,c]不是[[a,b,c],[a,b,c]]
                    for ii in temp_item :
                        if ii <0:
                            print('1')
                    max_len_2=max(max_len_2,len(temp_item))
                    #print(max_len_2)
                elif key.find('_S_O_')==-1 and key.find('_S_A_')==-1:  # s_o和s_a第一个是那啥的
                    len_1=len(temp_item)
                    max_len_1=max(max_len_1,len_1)
                    len_2=max([len(j) for j in temp_item])
                    max_len_2=max(max_len_2,len_2)
                    #print(max_len_1)
                    #print(max_len_2)
                    len_dic[0]=max_len_1
                    len_dic[1]=max_len_2
                else:
                    max_len_2=max(max_len_2,len(temp_item))
                    # if key=='_forward_S_A_answer_start':
                    #     print(1)
            for temp_n, temp_dic in enumerate(temp_data):
                temp_item=temp_dic[key]
                #print(1)
                if isinstance(temp_item, int_classes):                  # 是int，本题对应的是ID
                    f = copy.deepcopy(temp_data[temp_n][key])
                    _batch_data[temp_n][key] = f
                    #print('ID card:',temp_item)
                elif len(temp_item)==0:                                 # 跳过空内容
                    # 为空的就先不考虑了
                    continue
                elif key.find('_S_O_')!=-1 or key.find('_S_A_')!=-1:  # 这两个大小是[B,L]
                    res=copy.deepcopy(temp_data[temp_n][key])
                    if key.find('query')!=-1 and key.find('mask')==-1 and key.find('seg')==-1:  #是query而不是query_mask_seg
                        res += [0] * (max_len_2 - len(temp_item))
                    elif key.find('mask')!=-1:
                        res += [0] * (max_len_2 - len(temp_item))
                    elif key.find('seg')!=-1:
                        res += [1] * (max_len_2 - len(temp_item))
                    elif key.find('answer_start')!=-1 or key.find('answer_end')!=-1:
                        res += [-1] * (max_len_2 - len(temp_item))
                    else :
                        raise TypeError('data_process过程有问题')
                    f=copy.deepcopy(res)
                    _batch_data[temp_n][key]=np.array(f)
                else:       # [B,L1,L2]
                    res = copy.deepcopy(temp_item)
                    if key.find('query')!=-1 and key.find('mask')==-1 and key.find('seg')==-1:  #是query而不是query_mask_seg
                        # 最里面维度扩充
                        for item_n,item_item in enumerate(temp_item):
                            res[item_n] += [0] * (max_len_2 - len(item_item))
                        # 最外面维度扩充
                        res=res+[[0]*max_len_2]*(max_len_1-len(temp_item))
                    elif key.find('mask')!=-1:
                        for item_n,item_item in enumerate(temp_item):
                            res[item_n] += [0] * (max_len_2 - len(item_item))
                        # 最外面维度扩充
                        res=res+[[0]*max_len_2]*(max_len_1-len(res))
                    elif key.find('seg')!=-1:
                        for item_n,item_item in enumerate(temp_item):
                            res[item_n] += [1] * (max_len_2 - len(item_item))
                        # 最外面维度扩充
                        res=res+[[0]*max_len_2]*(max_len_1-len(temp_item))
                    elif key.find('answer_start')!=-1 or key.find('answer_end')!=-1:
                        for item_n,item_item in enumerate(temp_item):
                            res[item_n] += [-1] * (max_len_2 - len(item_item))
                        # 最外面维度扩充
                        res=res+[[-1]*max_len_2]*(max_len_1-len(temp_item))
                    elif key.find('answer')!=-1:
                        res += [-1] * (max_len_2 - len(temp_item))
                        # 没有外面的维度
                    else :
                        raise TypeError('data_process过程有问题')
                    f=copy.deepcopy(res)
                    _batch_data[temp_n][key]=np.array(f)
        return self.default_collate(_batch_data)

    def default_collate(self,batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)


        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, gpu=True,collate_fn=None):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for n,data_dict in enumerate(dataloader):
        #print(n)
        _dict = {}
        for name, tensor in data_dict.items():
            if gpu is True and 'line' not in name:
                _dict[name] = data_dict[name].long().cuda()
            else:
                _dict[name] = data_dict[name]
        #_dict['task_type']=dataset.task_type
        yield _dict
