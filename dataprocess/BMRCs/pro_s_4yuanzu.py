# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-05
# @Do: 针对4元组数据集的数据预处理

import csv
import pickle
import sys

import torch

"""
forward_aspect_query_template = ["[CLS]", "what", "aspects", "?", "[SEP]"]
forward_opinion_query_template = ["[CLS]", "what", "opinion", "given", "the", "aspect", "?", "[SEP]"]
backward_opinion_query_template = ["[CLS]", "what", "opinions", "?", "[SEP]"]
backward_aspect_query_template = ["[CLS]", "what", "aspect", "does", "the", "opinion", "describe", "?", "[SEP]"]
sentiment_query_template = ["[CLS]", "what", "sentiment", "given", "the", "aspect", "and", "the", "opinion", "?",]
"""



class QSAAS(object):
    """
        QuestionS And AnswerS
        保存对应的QA对
    """
    def __init__(self,
                 original_sample,
                 text,
                 forward_querys,
                 forward_answers,
                 category_querys,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers,
                 ):
        """
        :param original_sample:
        :param text:    原文本
        :param forward_querys:  A->O的问题  固定有一个”What aspects?“  之后是"What", "opinion", "given", "the", "aspect" XXX ?
                                由于第一步不一定得到多少个aspects，所以这个长度也不一定多长
        :param forward_answers: 对应的答案，其中（1）len(answers)应该等于len(querys)（2）格式为[0 0 0 0 1 1 0 ...] 就是mask啦
        :param category_querys: 实际上这个可以和aspect、sentiment、opinion进行结合
                                比如 what is the aspect XXX 's category?
                                比如  ["What", "sentiment", "given", "the", "aspect"（改成category或加上）] + aspect + ["and", "the", "opinion"] opinion?
                                emmmm好像可以考虑的组合有点多，甚至有些不知所措（作为插件或单独训练一个分类器？）
        :param backward_querys: O->A的问题  固定有一个”What opinions?“  然后是一堆 what aspect does the opinion good describe?
        :param backward_answers: 对应的答案
        :param sentiment_querys: 最后SA 的问题 ["What", "sentiment", "given", "the", "aspect"] + aspect + ["and", "the", "opinion"] opinion?
        :param sentiment_answers: 对应的答案
        """
        self.original_sample = original_sample  #
        self.text = text  #
        self.forward_querys=forward_querys
        self.forward_answers=forward_answers
        self.category_querys=category_querys
        self.backward_querys=backward_querys
        self.backward_answers=backward_answers
        self.sentiment_querys=sentiment_querys
        self.sentiment_answers=sentiment_answers


def get_text(lines):# 通过输入数据集的origin lines得到一种稍微标准化的输出
    """
    :param lines:
        格式为:
        [
            [
                'this unit is ` ` pretty ` ` and stylish , so my high school daughter was attracted to it for that reason .',
                '1,2 LAPTOP#DESIGN_FEATURES 2 5,6',
                '1,2 LAPTOP#DESIGN_FEATURES 2 9,10'
            ],[...],...,[...]
        ]
    :return:格式为：
    [
        {
            'text':
                ['The',
                   'apple',
                    'is',
                      'delicious'
                       'and'
                        'pretty'
                ],
         'quad':
                [
                    {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '0~2',
                   'opinion_index': '3,4',
                   'aspect': ['apple'],
                   'opinion': ['delicious']
                   },
                  {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '2',
                   'opinion_index': '5,6',
                   'aspect': ['apple'],
                   'opinion': ['pretty']
                   }
               ]
        }
        ,{...},...,{...}
    ]
    """
    all_data = []
    for item in lines:
        #temp = {}
        text = item[0].split()
        #temp['text'] = text
        quad_ = item[1:]
        quad = []
        for q in quad_:
            quad_temp = {}
            quad_temp['aspect_index'], quad_temp['category'] \
                , quad_temp['polarity'], quad_temp['opinion_index'] = q.split()
            i1, j1 = [int(asp_index) for asp_index in quad_temp['aspect_index'].split(',')]
            # print(i1,'   ',j1)
            # (text[1:3])
            quad_temp['aspect'] = text[i1:j1]
            i2, j2 = [int(opi_index) for opi_index in quad_temp['opinion_index'].split(',')]
            quad_temp['opinion'] = text[i2:j2]
            quad.append(quad_temp)

        # print(quad_temp['aspect'])
        temp['quad'] = quad
        temp={
          'text':text,
          'quad_':quad
        }
        all_data.append(temp)
    #return all
    return all_data

def get_quad(quad_list):
    """
    :param quad_list: 格式如下
            [
                    {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '0~2',
                   'opinion_index': '3,4',
                   'aspect': ['apple'],
                   'opinion': ['delicious']
                   },
                  {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '2',
                   'opinion_index': '5,6',
                   'aspect': ['apple'],
                   'opinion': ['pretty']
                   }
            ]
    :return:
        [[(1,2),(3,4),polarity,category],[...],...,[...]]
        [aspect_index,opinion_index,polarity,category]
    """
    quad_return=[]
    for item in quad_list:
        temp=[]
        a1,a2=(item['aspect_index'].split(','))
        temp.append((a1,a2))
        o1,o2=(item['opinion_index'].split(','))
        temp.append((o1,o2))
        temp.append(item['polarity'])
        temp.append(item['category'])
        quad_return.append(temp)
    print(quad_return)

def fusion_quad(quad):# 由于这个预处理本身也很快，数据量也不大，这里就不写哈希表加速了,也不考虑什么代码复用、模块化的内容了
    """
    :param quad:
            [
                    {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '0~2',
                   'opinion_index': '3,4',
                   'aspect': ['apple'],
                   'opinion': ['delicious']
                   },
                  {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '2',
                   'opinion_index': '5,6',
                   'aspect': ['apple'],
                   'opinion': ['pretty']
                   }
            ]
    :return:
        as_guanlian:
        格式为
            [
                {
                'aspect': [xxxx],
                'as_op':[yyyy,cccc,...],
                ‘aspect_index’=[i,j],     # 这个是key value
                ‘op_index’=[[a,b],[c,d],...],
                'category':zzzzzz
                },
                {...}
                ...,
                {...}
            ]
        op_guanlian:
            格式为：
            [
                {
                'opinion':[yyyy],
                'op_as':[xxxx],
                'as_index'=[i,j],
                'op_index'=[a,b],       # 这个是key value
                }
                ,...
            ]
    """
    as_guanlian=[]
    op_guanlian=[]
    for item in quad:
        aspect=item['aspect']
        temp_as_guanlian={}
        temp_op_guanlian={}
        as_index=item['aspect_index'].split(',')
        temp_as_guanlian['aspect']=aspect
        temp_as_guanlian['aspect_index']=as_index# key value
        opinion=item['opinion']
        op_index=item['opinion_index'].split(',')
        temp_as_guanlian['as_op']=[opinion] # 初始化的化是要让他变成list的
        temp_as_guanlian['op_index']=op_index

        temp_op_guanlian['aspect']=[aspect]
        temp_op_guanlian['aspect_index']=as_index
        temp_op_guanlian['as_op']=opinion
        temp_op_guanlian['op_index']=op_index  #key value

        if len(as_guanlian)==0:
            as_guanlian.append(temp_as_guanlian)
        else:
            flag=True
            for n,item_as in enumerate(as_guanlian):
                if item_as['aspect_index']==as_index:
                    as_guanlian[n]['op_as'].append(opinion)
                    as_guanlian[n]['op_index'].append(op_index)
                    flag=False
                    break
            if flag is True:
                as_guanlian.append(temp_as_guanlian)
        if len(op_guanlian)==0:
            op_guanlian.append(temp_op_guanlian)
        else:
            flag=True
            for n,item_op in enumerate(op_guanlian):
                if item_as['aspect_index'] == as_index:
                    op_guanlian[n]['op_as'].append([aspect]) # op-as 对
                    op_guanlian[n]['op_index'].append(as_index)
                    flag = False
                    break
            if flag is True:
                op_guanlian.append(temp_op_guanlian)
    #print(quad_return)
    return as_guanlian, op_guanlian


if __name__ == '__main__':
    home_path = "../data/uni/semeval_4yuanzu_EN/"
    dataset_name_list = ['laptop','rest16']
    dataset_type_list = ["train", "test", "dev"]
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            output_path = "../data/uni/semeval_4yuanzu_EN/preprocess/" + dataset_name + "_" + dataset_type + "_QAS1.pt"
            # 读取原数据（CSV格式）
            filenameTSV1 = '../data/uni/semeval_4yuanzu_EN/laptop/+laptop_quad_dev.tsv'
            with open(filenameTSV1, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")  # , quotechar=quotechar)
                lines = []
                for line in reader:
                    """"
                    if sys.version_info[0] == 2:# 给python2用的，不用在意
                        line = list(unicode(cell,'UTF-8') for cell in line)
                    """
                    lines.append(line)
                print(lines)
            sample_list = []
            text_list=get_text(lines)
            for k in range(len(text_list)):
                text= text_list[k]['text']
                quad=text_list[k]['quad']
                quad_l=get_quad(quad)

                forward_aspect_query_list=[] # 从aspect开始的一系列问题
                forward_aspect_answer_list=[]# 与上面这个对应的相应的答案，0为不是答案，1为是答案
                forward_aspect_query_list.append(["What","aspects","?"]) # 第一个问题，aspect有哪些
                start=[0]*len(text)
                end=[0]*len(text)
                for qu in quad_l:# 这样从直觉上来说似乎有些不妥？
                    start[qu[0][0]]=1# start
                    end[qu[0][1]]=1# end
                forward_aspect_answer_list.append([start,end])

                forward_opinion_query_list=[] # 从opinion 开始的一系列问题
                forward_opinion_answer_list=[] # 对应的答案，分两行start=[0 0 0 0 1 0 0];end =[0 0 0 0 0 0 1]
                forward_opinion_query_list.append(["What", "opinions", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for to in quad_l:
                    start[qu[1][0]] = 1
                    end[qu[1][1]] = 1
                forward_opinion_answer_list.append([start, end])

                for qu in quad: # 这方法看起来好蠢，这得问多少轮啊
                    asp_quad=qu['aspect']# 'xxxx'
                    asp_index_quad=qu['aspect_index'].split(',')# '1,2'
                    opi_quad=qu['opinion']#'xxxx'
                    opi_index_quad=qu['opinion_index'].split(',')#
                    pol_quad=qu['polarity']  #
                    cat_quad=qu['category']  #
                    """
                    'quad':
                [
                    {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '0~2',
                   'opinion_index': '3,4',
                   'aspect': ['apple'],
                   'opinion': ['delicious']
                   },
                  {'aspect_index': '1,2',
                   'category': 'XXXX#YYYYYY',
                   'polarity': '2',
                   'opinion_index': '5,6',
                   'aspect': ['apple'],
                   'opinion': ['pretty']
                   }
               ]
               """
                    f_asp_q=["What", "opinion", "given", "the", "aspect"] +asp_quad + ["?"]
                    forward_aspect_query_list.append(f_asp_q)
                    start=[0]*len(f_asp_q)
                    end=[0]*len(f_asp_q)
                    # 不行还是得弄出来对应的那种字典




                temp_sample = dual_sample(text_lines[k], text, forward_query_list, forward_answer_list,
                                          backward_query_list, backward_answer_list, sentiment_query_list,
                                          sentiment_answer_list)
                sample_list.append(temp_sample)