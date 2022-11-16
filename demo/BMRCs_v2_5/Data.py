# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

from torch.utils.data import Dataset, DataLoader
import numpy as np
from DatasetCapsulation import Collate3

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
        self._backward_asp_query_mask = pre_data['_backward_asp_query_mask']
        self._backward_opi_query_mask = pre_data['_backward_opi_query_mask']
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']

        self._sentiment_query = pre_data['_sentiment_query']
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']


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

def generate_fi_batches_2(dataset, batch_size, shuffle=False, drop_last=True, ifgpu=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            if ifgpu:
                out_dict[name] = data_dict[name].long().cuda()
            else:
                out_dict[name] = data_dict[name]
        yield out_dict

def generate_fi_batches(dataset, batch_size, shuffle=True, drop_last=True, ifgpu=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for n,data_dict in enumerate(dataloader):
        _dict = {}
        for name, tensor in data_dict.items():
            _dict[name] = data_dict[name]
        yield _dict