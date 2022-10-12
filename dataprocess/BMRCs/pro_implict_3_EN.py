# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4
# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-05
# @Do:     增加注释
import pickle
import os

#import torch

class dual_sample(object):
    """
        保存对应的QA对
    """
    def __init__(self,
                 original_sample,
                 text,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers):
        """
        :param original_sample:
        :param text:    原文本
        :param forward_querys:  A->O的问题  固定有一个”What aspects?“  之后是"What", "opinion", "given", "the", "aspect" XXX ? 由于第一步不一定得到多少个aspects，所以这个长度也不一定多长
        :param forward_answers: 对应的答案，其中（1）len(answers)应该等于len(querys)（2）格式为[0 0 0 0 1 1 0 ...] 就是mask啦
        :param backward_querys: O->A的问题  固定有一个”What opinions?“  然后是一堆 what aspect does the opinion good describe?
        :param backward_answers: 对应的答案
        :param sentiment_querys: 最后SA 的问题 ["What", "sentiment", "given", "the", "aspect"] + aspect + ["and", "the", "opinion"] opinion?
        :param sentiment_answers: 对应的答案
        """
        self.original_sample = original_sample  #
        self.text = text  #
        self.forward_querys=forward_querys
        self.forward_answers=forward_answers
        self.backward_querys=backward_querys
        self.backward_answers=backward_answers
        self.sentiment_querys=sentiment_querys
        self.sentiment_answers=sentiment_answers


def get_text(lines):
    """
    :param lines: f.readlines()
    :return: text_list apsect_list 和opinion_list ；后面俩都用O来表示mask，其他如P表示opinion等等
    """
    # 样本示例：
    # It is always reliable , never bugged and responds well .####It=O is=O always=O reliable=O ,=O never=O bugged=O and=O responds=T-POS well=O .=O####It=O is=O always=O reliable=O ,=O never=O bugged=O and=O responds=O well=S .=O
    # 即’####‘分割开了  text ####papers=O aspect=T-POS(等) papers=O #### papers=O opinion=P papers=O
    text_list = []
    aspect_list = []
    opinion_list = []
    for f in lines:
        temp = f.split("####")
        assert len(temp) == 3
        word_list = temp[0].split()
        aspect_label_list = [t.split("=")[-1] for t in temp[1].split()]
        opinion_label_list = [t.split("=")[-1] for t in temp[2].split()]
        falg=False
        for i in opinion_label_list:
            if i!='O':
                flag=True
            print(1)

        if flag is False:
            print(1)

        assert len(word_list) == len(aspect_label_list) == len(opinion_label_list)
        text_list.append(word_list)
        aspect_list.append(aspect_label_list)
        opinion_list.append(opinion_label_list)
    return text_list, aspect_list, opinion_list


def valid_data(triplet, aspect, opinion):# 保证aspect和opinion处都不是O
    for t in triplet[0][0]:
        assert aspect[t] != ["O"]
    for t in triplet[0][1]:
        assert opinion[t] != ["O"]


def fusion_dual_triplet(triplet):#
    triplet_aspect = []
    triplet_opinion = []
    triplet_sentiment = []
    dual_opinion = []
    dual_aspect = []
    for t in triplet:
        if t[0] not in triplet_aspect: # 这个aspect之前没有
            triplet_aspect.append(t[0]) # aspect放进去
            triplet_opinion.append([t[1]]) # opinion放进去
            triplet_sentiment.append(t[2]) # sent放进去
        else:                          # 这个aspect之前有
            idx = triplet_aspect.index(t[0])#获得对应的索引
            triplet_opinion[idx].append(t[1]) #对应的里面加 比如[[1],[3,4]]代表triplet_aspect（2个aspect） 那么[[9],[2]]是之前对应的opinion，现在新的triplet是([3,4],[7,8],[1]),那么就变成[[9],[[2],[7,8]]]
            assert triplet_sentiment[idx] == t[2] # 同一个aspect就应该只有一个情感
        if t[1] not in dual_opinion:    # 如果opinion之前没有
            dual_opinion.append(t[1])
            dual_aspect.append([t[0]])
        else:                           # 有
            idx = dual_opinion.index(t[1])
            dual_aspect[idx].append(t[0])

    # triplet_aspect:
    return triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect


if __name__ == '__main__':

    lap_path='../../data/uni/semeval_implict_3_EN/laptops'
    rest_path='../../data/uni/semeval_implict_3_EN/restaurants'

    data_name='Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
    if(os.path.isfile(lap_path+'/'+data_name)):

        # 读文件
        f = open(lap_path+'/'+data_name, 'rb')
        du_data = pickle.load(f)
        f.close()

    else:#额外的数据处理，参考其他文件
        pass



