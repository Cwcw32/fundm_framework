# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4
# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-05
# @Do:     增加注释
import pickle
import torch
import json
from tqdm import tqdm
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
            #print(1)

        if flag is False:
            pass
            #print(1)

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
    home_path = "../../data/uni/semeval1456_3yuanzu_EN/original/"
    dataset_name_list = ["14lap", "14rest", "15rest", "16rest"]
    dataset_type_list = ["train", "test", "dev"]
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            output_path = "../../data/uni/semeval1456_3yuanzu_EN/preprocess/" + dataset_name + "_" + dataset_type + ".pt"
            # 读三元组
            f = open(home_path + dataset_name + "/" + dataset_name + "_pair/" + dataset_type + "_pair.pkl", "rb")
            triple_data = pickle.load(f)#格式： [(aspect_index,opinion_index,sentiment)]
            f.close()
            # 读具体标注（为啥还要读三元组呢？）
            f = open(home_path + dataset_name + "/" + dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = f.readlines()  # 格式见get_text()的样本示例
            f.close()
            # 获得text,aspect,opinion
            text_list, aspect_list, opinion_list = get_text(text_lines)
            sample_list = []
            for k in tqdm(range(len(text_list))): # 不处理单aspect 单opinion的情况？
                ID=k
                triplet = triple_data[k]
                text = text_list[k]
                valid_data(triplet, aspect_list[k], opinion_list[k]) # 验证一下aspect、opinion处是不是O
                triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect = fusion_dual_triplet(triplet)
                forward_query_list = []
                backward_query_list = []
                sentiment_query_list = []
                forward_answer_list = []
                backward_answer_list = []
                sentiment_answer_list = []
                forward_query_list.append(["What", "aspects", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for ta in triplet_aspect:
                    assert ta[0]<=ta[-1]
                    start[ta[0]] = 1
                    end[ta[-1]] = 1
                forward_answer_list.append([start, end])

                backward_query_list.append(["What", "opinions", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for to in dual_opinion:
                    start[to[0]] = 1
                    end[to[-1]] = 1
                backward_answer_list.append([start, end])

                for idx in range(len(triplet_aspect)):
                    ta = triplet_aspect[idx]
                    # opinion query
                    query = ["What", "opinion", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["?"]
                    forward_query_list.append(query)
                    start = [0] * len(text)
                    end = [0] * len(text)
                    for to in triplet_opinion[idx]:
                        start[to[0]] = 1
                        end[to[-1]] = 1
                    forward_answer_list.append([start, end])
                    # sentiment query
                    query = ["What", "sentiment", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["and", "the",
                                                                                                        "opinion"]
                    for idy in range(len(triplet_opinion[idx]) - 1):
                        to = triplet_opinion[idx][idy]
                        query += text[to[0]:to[-1] + 1] + ["/"]
                    to = triplet_opinion[idx][-1]
                    query += text[to[0]:to[-1] + 1] + ["?"]
                    sentiment_query_list.append(query)
                    sentiment_answer_list.append(triplet_sentiment[idx])
                for idx in range(len(dual_opinion)):
                    ta = dual_opinion[idx]
                    # opinion query
                    query = ["What", "aspect", "does", "the", "opinion"] + text[ta[0]:ta[-1] + 1] + ["describe", "?"]
                    backward_query_list.append(query)
                    start = [0] * len(text)
                    end = [0] * len(text)
                    for to in dual_aspect[idx]:
                        start[to[0]] = 1
                        end[to[-1]] = 1
                    backward_answer_list.append([start, end])

#################################################
                # (
                # text_lines[k], text, forward_query_list, forward_answer_list, backward_query_list, backward_answer_list,
                # sentiment_query_list, sentiment_answer_list)
                sample_dic = {
                    'ID': ID,
                    'text': text,
                    'triplet': triplet,
                    'QAs': {
                        'S_A_QUERY': [forward_query_list[0]],
                        'S_A_ANSWER': [forward_answer_list[0]],

                        'A_O_QUERY': forward_query_list[1:],
                        'A_O_ANSWER': forward_answer_list[1:],

                        'S_O_QUERY': [backward_query_list[0]],
                        'S_O_ANSWER': [backward_answer_list[0]],
                        'O_A_QUERY': backward_query_list[1:],
                        'O_A_ANSWER': backward_answer_list[1:],


                        'AO_P_QUERY': sentiment_query_list,
                        'AO_P_ANSWER': sentiment_answer_list,
                    }
                }
                ###############
                # 这里麻烦统计一下各种问题的信息
                #############
                sample_list.append(
                    sample_dic
                )
            with open(dataset_name + dataset_type +'_3yuanzu'+'.json', 'w+') as file:
                json.dump(sample_list, file)


