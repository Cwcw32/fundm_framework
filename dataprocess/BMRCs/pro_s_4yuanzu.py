# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-05
# @Do: 针对4元组数据集的数据预处理

import csv
import operator
import pickle
import sys
import json

from tqdm import tqdm

#import torch

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

    def get_labels(self, domain_type):
        """See base class."""
        l = None
        sentiment = None
        if domain_type.startswith('rest'):
            l = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
        elif domain_type == 'laptop':
            l = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE', 'CPU#OPERATION_PERFORMANCE',
            'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE', 'POWER_SUPPLY#CONNECTIVITY', 'SOFTWARE#USABILITY',
            'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY', 'FANS&COOLING#DESIGN_FEATURES',
            'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY', 'POWER_SUPPLY#GENERAL', 'PORTS#QUALITY',
            'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY', 'MOUSE#GENERAL', 'KEYBOARD#MISCELLANEOUS',
            'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS', 'SOFTWARE#PRICE', 'FANS&COOLING#OPERATION_PERFORMANCE',
            'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL', 'DISPLAY#OPERATION_PERFORMANCE',
            'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY', 'KEYBOARD#PRICE', 'SUPPORT#OPERATION_PERFORMANCE',
            'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY', 'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES',
            'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL', 'PORTS#USABILITY', 'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY',
            'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'HARDWARE#DESIGN_FEATURES', 'MEMORY#USABILITY',
            'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY', 'OS#PRICE', 'SUPPORT#QUALITY',
            'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL', 'COMPANY#OPERATION_PERFORMANCE',
            'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES', 'Out_Of_Scope#OPERATION_PERFORMANCE',
            'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY', 'DISPLAY#USABILITY', 'POWER_SUPPLY#QUALITY',
            'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY', 'HARDWARE#GENERAL', 'COMPANY#PRICE',
            'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'SOFTWARE#PORTABILITY', 'HARD_DISC#OPERATION_PERFORMANCE',
            'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE', 'OS#USABILITY',
            'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE', 'LAPTOP#PRICE', 'OS#GENERAL', 'HARDWARE#PRICE',
            'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'FANS&COOLING#QUALITY', 'BATTERY#OPERATION_PERFORMANCE',
            'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE', 'KEYBOARD#GENERAL', 'SOFTWARE#QUALITY',
            'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'WARRANTY#QUALITY', 'HARD_DISC#QUALITY',
            'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
        sentiment = ['0', '1', '2']
        label_list = []
        # label_list.append(l)
        # label_list.append(sentiment)
        cate_senti = []
        for cate in l:
            for senti in sentiment:
                cate_senti.append(cate+'#'+senti)
        label_list.append(cate_senti)
        return label_list

def get_query_and_answer(domain_type):
    """
    目的是模拟NER中的阅读理解任务第一版（BERT-MRF）
    :param domain_type:
        LAPTOP
        RESTAURANT
    :return:
        forward_category_query_list：
            包含本领域的category
            list的长度是category的数量
        forward_category_answer_list:
            上面的query对应的答案
    """
    if domain_type=='LAPTOP':
        l = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE',
         'CPU#OPERATION_PERFORMANCE',
         'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE', 'POWER_SUPPLY#CONNECTIVITY',
         'SOFTWARE#USABILITY',
         'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY',
         'FANS&COOLING#DESIGN_FEATURES',
         'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY', 'POWER_SUPPLY#GENERAL',
         'PORTS#QUALITY',
         'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY', 'MOUSE#GENERAL',
         'KEYBOARD#MISCELLANEOUS',
         'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS', 'SOFTWARE#PRICE',
         'FANS&COOLING#OPERATION_PERFORMANCE',
         'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL',
         'DISPLAY#OPERATION_PERFORMANCE',
         'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY', 'KEYBOARD#PRICE',
         'SUPPORT#OPERATION_PERFORMANCE',
         'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY',
         'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES',
         'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL', 'PORTS#USABILITY',
         'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY',
         'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'HARDWARE#DESIGN_FEATURES',
         'MEMORY#USABILITY',
         'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY', 'OS#PRICE',
         'SUPPORT#QUALITY',
         'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL',
         'COMPANY#OPERATION_PERFORMANCE',
         'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES', 'Out_Of_Scope#OPERATION_PERFORMANCE',
         'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY', 'DISPLAY#USABILITY',
         'POWER_SUPPLY#QUALITY',
         'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY', 'HARDWARE#GENERAL',
         'COMPANY#PRICE',
         'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'SOFTWARE#PORTABILITY',
         'HARD_DISC#OPERATION_PERFORMANCE',
         'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE',
         'OS#USABILITY',
         'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE', 'LAPTOP#PRICE',
         'OS#GENERAL', 'HARDWARE#PRICE',
         'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'FANS&COOLING#QUALITY',
         'BATTERY#OPERATION_PERFORMANCE',
         'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE', 'KEYBOARD#GENERAL',
         'SOFTWARE#QUALITY',
         'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'WARRANTY#QUALITY',
         'HARD_DISC#QUALITY',
         'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
    else:
        l = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
    cat_to_id={}
    id_to_cat={}
    num=0
    for item in l:
        cat_to_id[item]=num
        id_to_cat[id]=item
    return cat_to_id,id_to_cat

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
        #temp['quad'] = quad
        temp={
          'text':text,
          'quad':quad
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
    #print(quad_return)
def fusion_quad(quad,text=None):# 由于这个预处理本身也很快，数据量也不大，这里就不写哈希表加速了,也不考虑什么代码复用、模块化的内容了
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

                'aspect': [aspect words],
                ‘aspect_index’=[i,j],     # 这个是key value

                'as_op':[o1,cccc,...],
                ‘op_index’=[[a,b],[c,d],...],

                'category':[c1,c2,...,cn],
                'polarity':[p1,p2,...,pn],
                },
                {...}
                ...,
                {...}
            ]
        op_guanlian:
            格式为：
            [
                {
                'opinion':[opinion],
                'op_index'=[a,b],       # 这个是key value

                'op_as':[a1,a2,a3,...,an],
                'as_index'=[i,j],

                'category':[c1,c2,c3,...,cn],
                'polarity':[p1,p2,p3,...,pn],
                }
                ,...
            ]
    """
    num=0
    as_guanlian=[]
    op_guanlian=[]
    for n1,item in enumerate(quad):
        temp_as_guanlian={}
        temp_op_guanlian={}

        aspect=item['aspect']
        as_index=item['aspect_index'].split(',')
        opinion=item['opinion']
        op_index=item['opinion_index'].split(',')
        category=item['category']
        polarity=item['polarity']

        temp_as_guanlian['aspect']=aspect
        temp_as_guanlian['aspect_index']=as_index# key value,加[]后续处理会很麻烦哦
        temp_as_guanlian['as_op']=[opinion] # 初始化的化是要让他变成list的
        temp_as_guanlian['op_index']=[op_index]
        temp_as_guanlian['category']=[category]
        temp_as_guanlian['polarity']=[polarity]

        temp_op_guanlian['opinion']=opinion
        temp_op_guanlian['opinion_index']=op_index  #key value
        temp_op_guanlian['op_as']=[aspect]
        temp_op_guanlian['as_index']=[as_index]
        temp_op_guanlian['category']=[category]
        temp_op_guanlian['polarity']=[polarity]

        ##############
        ## 待更改
        ##############
        if aspect=='' and opinion=='':#没有aspect的情况和opinion的情况，暂时放一放，等category研究完的
            continue

        if len(as_guanlian)==0:
            as_guanlian.append(temp_as_guanlian)
        else:
            flag=True
            for n2,item_as in enumerate(as_guanlian):
                if operator.eq(item_as['aspect_index'],as_index):#as_index是主键，假如新的四元组也针对这个aspect
                    # 判断1 category是否是一样的
                    if as_guanlian[n2]['category'].count(category)!=0: # as一样，category也一样，那只能是观点词不一样了
                        #continue # 先跳过这里吧
                        index=as_guanlian[n2]['category'].index(category)# 获得索引
                        #assert op_guanlian[n2]['polarity'][index]!=polarity # 对应的情感、观点词一样就跳过
                        if  as_guanlian[n2]['polarity'][index]==polarity and as_guanlian[n2]['as_op'][index]==opinion: # 对应的情感、观点词都一样就跳过
                            continue
                        #assert op_guanlian[n2]['opinion'][index]!=opinion# 相当于重复句子
                        #if as_guanlian[n2]['as_op'][index]!=opinion:# 相当于重复句子,这里就先不管那啥了
                            #continue
                    as_guanlian[n2]['as_op'].append(opinion)
                    as_guanlian[n2]['op_index'].append(op_index)
                    as_guanlian[n2]['category'].append(category)
                    as_guanlian[n2]['polarity'].append(polarity)
                    flag=False
                    num+=1
                    """
                    if opinion==[]:
                        print('ASASASAS: ',as_guanlian)
                        print('TEXT:',text)
                        print('QUAD:',quad)
                        print('\n')
                    #break
                    """
            if flag is True:
                as_guanlian.append(temp_as_guanlian)

        if len(op_guanlian)==0:
            op_guanlian.append(temp_op_guanlian)
        else:
            flag=True

            for n3,item_op in enumerate(op_guanlian):
                if operator.eq(item_op['opinion_index'],op_index):#op_index是主键
                    if op_guanlian[n3]['category'].count(category)!=0: # as一样，category也一样
                        #continue # 先跳过这里吧
                        index=op_guanlian[n3]['category'].index(category)# 获得对应的索引
                        ################
                        ##感觉写的有点不对
                        ################
                        #assert op_guanlian[n2]['polarity'][index]!=polarity # 对应的情感、观点词一样就跳过
                        if  op_guanlian[n3]['polarity'][index]==polarity and op_guanlian[n3]['op_as'][index]==opinion: # 情感和aspect都一样就跳过吧，说明是重复句子
                            continue
                        #assert op_guanlian[n2]['opinion'][index]!=opinion# 相当于重复句子
                        #if op_guanlian[n3]['op_as'][index]!=opinion:# 相当于重复句子,就跳过吧
                        #    continue

                    op_guanlian[n3]['op_as'].append(aspect) # op-as 对
                    op_guanlian[n3]['as_index'].append(as_index)
                    op_guanlian[n3]['category'].append(category)
                    op_guanlian[n3]['polarity'].append(polarity)
                    flag = False
                    num+=1
                    # if aspect==[]:
                    #     print('OPOPOP： ',op_guanlian)
                    #     print('text:',text)
                    #     print(quad)
                    #     print('\n')
                    #break

            if flag is True:
                op_guanlian.append(temp_op_guanlian)
    return as_guanlian, op_guanlian,num


if __name__ == '__main__':
    home_path = "../../data/uni/semeval_4yuanzu_EN/"
    dataset_name_list = ['laptop']#,'rest16']
    dataset_type_list = [ "dev","train", "test"]

    ID=0

    for dataset_name in dataset_name_list:#: tqdm(train_loader, desc='Train Iteration:')
        for dataset_type in tqdm(dataset_type_list,desc=dataset_name+'_'):
            output_path = "../../data/uni/semeval_4yuanzu_EN/preprocess/" + dataset_name + "_" + dataset_type + "_QAS1.pt"
            # 读取原数据（CSV格式）
            # filenameTSV1='../../data/uni/semeval_4yuanzu_EN/laptop/laptop_quad_dev.tsv'
            filenameTSV1 = '../../data/uni/semeval_4yuanzu_EN/'+dataset_name+'/'+dataset_name+'_quad_'\
                           +dataset_type+'.tsv'
            with open(filenameTSV1, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")  # , quotechar=quotechar)
                lines = []
                for line in reader:
                    """"
                    if sys.version_info[0] == 2:# 给python2用的，不用在意
                        line = list(unicode(cell,'UTF-8') for cell in line)
                    """
                    lines.append(line)
                #print(lines)
            sample_list = []
            text_list=get_text(lines)
            for k in range(len(text_list)):
                ID+=1
                text= text_list[k]['text']
                quad=text_list[k]['quad']

                as_guanlian, op_guanlian, num=fusion_quad(quad)
                quad_l=get_quad(quad)

                """
                        as_guanlian:
                        格式为
                        [
                            {
            
                            'aspect': [aspect words1],
                            ‘aspect_index’=[i,j],     # 这个是key value
            
                            'as_op':[o1,cccc,...],
                            ‘op_index’=[[a,b],[c,d],...],
            
                            'category':[c1,c2,...,cn],
                            'polarity':[p1,p2,...,pn],
                            },
                            {...}
                            ...,
                            {...}
                        ]
                """
                forward_aspect_query_list = []  # S->A
                forward_aspect_answer_list = []
                forward_aspect_opinion_query_list = [] # A->O
                forward_aspect_opinion_answer_list = []
                forward_aspect_category_query_list=[]  # A->C
                forward_aspect_category_answer_list=[]
                forward_aspect_opinion_category_query_list=[]  #(A,O)->C
                forward_aspect_opinion_category_answer_list=[]
                forward_aspect_category_opinion_query_list=[] #(A,C)->O
                forward_aspect_category_opinion_answer_list=[]




                forward_pol_query_list = [] #(A,O,C)->P
                forward_pol_answer_list = []

                forward_opinion_query_list = []  # S->O
                forward_opinion_answer_list = []  # 与上面这个对应的相应的答案，0为不是答案，1为是答案
                forward_opinion_aspect_query_list = [] # O->A
                forward_opinion_aspect_answer_list = []
                forward_opinion_category_query_list = [] # O->C
                forward_opinion_category_answer_list = []
                forward_opinion_aspect_category_query_list = [] # (O,A)->C
                forward_opinion_aspect_category_answer_list = []
                forward_opinion_category_aspect_query_list = []# (O,C)->A
                forward_opinion_category_aspect_answer_list = []


                forward_category_query_list=[]  # 从category开始的一系列问题,由于之前已经覆盖了一些这里就不写全了
                forward_category_answer_list=[]

                # 目的是为了训练从sentence到A的能力
                start_as = [0] * len(text) # 这里放外面是因为aspect提取希望是一起提出来
                end_as = [0] * len(text)
                forward_aspect_query_list.append(["What", "aspects", "?"])  # 第一个问题，aspect有哪些
                forward_aspect_answer_list.append([start_as,end_as])

                # 目的是为了训练从sentence到O的能力
                start_op = [0] * len(text) # 这里放外面是因为aspect提取希望是一起提出来
                end_op = [0] * len(text)
                forward_opinion_query_list.append(["What", "opinions", "?"])  # 第一个问题，aspect有哪些
                forward_opinion_answer_list.append([start_op,end_op])

                # 目的是为了训练从sentence到C的能力
                forward_category_query_list.append(["What","categorys","?"]) # 改成一个二分类的问题，对应的答案是对应位置向量为1！
                forward_category_answer_list.append(0) # 改成一个二分类的问题，对应的答案是对应位置向量为1！


                """
                以rest为例，这里先写一个，对应的明天再写
                ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
                'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
                另外由于fusion_quad那还没改，所以a方向和o方向可以随便选一个作为c方向
                """


                for as_index,as_item in enumerate(as_guanlian):

                    aspect=as_item['aspect']
                    aspect_index=[int(i) for i in as_item['aspect_index']]
                    as_op=as_item['as_op']
                    op_index=[[int(i),int(j)] for i,j in as_item['op_index']]
                    category=as_item['category']
                    polarity=[int(i) for i in as_item['polarity']]

                    start_as[aspect_index[0]]=1
                    end_as[aspect_index[1]-1]=1
                    forward_aspect_answer_list[0]=[start_as,end_as]

                    # 目的是为了训练从A到C的能力
                    ask_sen_as_ca = ["What", "category", "given", "the", "aspect"] + aspect + ["?"]
                    forward_aspect_category_query_list.append(ask_sen_as_ca)
                    forward_aspect_category_answer_list.append(1111111111111111111111111111111)

                    ###############################
                    # 目的是为了训练从（A,sentence,C）到（A,O,C）的能力
                    ##############################

                    if aspect != []:
                        # 从A到O
                        ask_sen_as_op = ["What", "opinion", "given", "the", "aspect"] + aspect + ["?"]
                    else:
                        # 从(A,C)到O
                        ask_sen_as_op = ["What", "opinion", "describe", "the", "category"] + category + [ "with","no", "aspect","?"]

                    start_as_op = [0] * len(text)
                    end_as_op = [0] * len(text)
                    forward_aspect_opinion_query_list.append(ask_sen_as_op)

                    for as_op_n,as_op_item in enumerate(as_op):

                        start_as_op[op_index[as_op_n][0]]=1
                        end_as_op[op_index[as_op_n][1]-1]=1
                        if len(forward_aspect_opinion_answer_list)<as_index: # 还没有初始化过
                            forward_aspect_opinion_answer_list[aspect_index] = [start_as_op, end_as_op]# 把新的opinion加到答案里面
                        else:
                            forward_aspect_opinion_answer_list.append([start_as_op, end_as_op])

                        # 从(A,O)到C  []情况暂时没考虑，很复杂的问题
                        ask_sen_as_op_ca=["What", "category", "given", "the", "aspect"] +aspect + ["and","the","opinion"]+as_op_item+["?"]
                        forward_aspect_opinion_category_query_list.append(ask_sen_as_op_ca)
                        forward_aspect_opinion_category_answer_list.append([0])

                        # 从(A,C)到O
                        ask_sen_as_ca_op=["What","opinion","given","the","aspect"]+aspect+["and","the","category"]+category+["?"]
                        forward_aspect_category_opinion_query_list.append(ask_sen_as_ca_op)
                        forward_aspect_category_opinion_answer_list.append([0])

                    ##############
                    ## 待更改
                    ##############
                    # 这里单独拿出来是因为任务最终的目标实际上是为了提取四元组，而这个过程是训练的过程，所以在有gold truth的情况下可以这样做，注意的点是推理和训练的过程并不一样
                    # 一个也许的考虑是后面加上 (for the category xxxxx)
                    # 一个aspect就应该对应一个category
                    ##############################
                    # 目的是为了训练(A,O,C)得到S的能力
                    ##############################
                    if as_op !=[[]] and aspect!=[]:
                        # 有aspect也有opinion，采用BMRC的标准方法
                        pol_query = ["What", "sentiment", "given", "the", "aspect"] + aspect +["from","the","category"]+[category[0]]+ ["and", "the", "opinion"]+[sff for as_op_ii in as_op for sff in as_op_ii ]+["?"]
                        forward_pol_query_list.append(pol_query)
                        forward_pol_answer_list.append(polarity[0])
                    elif as_op==[[]] and aspect!=[]:
                        # 有aspect 却没有opinion
                        pol_query = ["What", "sentiment", "given", "the", "aspect"] + aspect + ["without", "opinion","for","the","category"]+[category[0]]+["?"]
                        forward_pol_query_list.append(pol_query)
                        forward_pol_answer_list.append(polarity[0])
                    elif aspect==[] and as_op!=[[]]:
                        # 有opinion没有aspect的情况
                        pol_query = ["What", "sentiment", "given", "the", "opinion"]+[sff for as_op_ii in as_op for sff in as_op_ii]+ ["for","the","category"]+[category[0]]+["without","aspect","?"]

                        forward_pol_query_list.append(pol_query)
                        forward_pol_answer_list.append(polarity[0])
                    else:
                        # 既没有aspect也没有opinion的情况
                        pol_query = ["What", "sentiment", "given","no","opinion"]+ ["for","the","category"]+[category[0]]+["without","aspect","?"]
                        forward_pol_query_list.append(pol_query)
                        forward_pol_answer_list.append(polarity[0])

                    """
                    op_guanlian:
                    格式为：
                    [
                        {
                            'opinion': [opinion],
                            'op_index' = [a, b],  # 这个是key value
                            'op_as':[a1, a2, a3, ..., an],
                            'as_index' = [i, j],
                            'category':[c1, c2, c3, ..., cn],
                            'polarity': [p1, p2, p3, ..., pn],
                    }
                    , ...
                    ]
                    """



                for op_item in op_guanlian:

                    opinion=op_item['opinion']
                    opinion_index=[int(i) for i in op_item['opinion_index']]

                    # 目的是为了训练从O到C的能力
                    ask_sen_op_ca = ["What", "category", "does", "the", "opinion"] + opinion + ["describe", "?"]
                    forward_opinion_category_query_list.append(ask_sen_op_ca)
                    forward_opinion_category_answer_list.append([0])

                    op_as=op_item['op_as']
                    op_index=[[int(i),int(j)] for i,j in op_item['as_index']]
                    category=op_item['category']
                    polarity=[int(i) for i in op_item['polarity']]

                    # 每次都要更新一下，因为我们希望第一次就把所有opinion提取完成
                    start_op[opinion_index[0]]=1
                    end_op[opinion_index[1]-1]=1
                    forward_opinion_answer_list[0]=[start_op,end_op]

                    for op_as_n,op_as_item in enumerate(op_as):
                        start_op_as = [0] * len(text)
                        end_op_as = [0] * len(text)

                        # 目的是为了训练从O到A的能力
                        if opinion!=[]:
                            ask_sen_op_as= ["What", "aspect", "does", "the", "opinion"] + opinion+ ["describe", "?"]
#                            ["What", "is", "the", "aspect"] +aspect + ["?"]
                        else:
                            ask_sen_op_as= ["Which", "aspect", "has" ,"no","opinions"+"?"]

                        forward_opinion_aspect_query_list.append(ask_sen_op_as)
                        start_op_as[op_index[op_as_n][0]]=1
                        end_op_as[op_index[op_as_n][1]-1]=1
                        forward_opinion_aspect_answer_list.append([start_op_as,end_op_as])

                        # 从O到C的能力
                    # 从(O,A)到C  []情况暂时没考虑，很复杂的问题
                    ask_sen_op_as_ca = ["What", "category", "given", "the", "aspect"] + [sff for op_as_ii in op_as for sff in op_as_ii] + ["and", "the","opinion"] + opinion + [ "?"]
                    forward_opinion_aspect_category_query_list.append(ask_sen_op_as_ca)
                    forward_opinion_aspect_category_answer_list.append(0)

                    # 从(O,C)到A
                    ask_sen_op_ca_as = ["What", "aspect", "given", "the", "opinion"] + opinion + ["and", "the","category"] + category+["?"]
                    forward_opinion_category_aspect_query_list.append(ask_sen_op_ca_as)
                    forward_opinion_category_aspect_answer_list.append(0)


                sample_dic={
                    'ID':ID,
                    'text':text,
                    'quad':quad,
                     'QAs': {
                    'S_A_QUERY':forward_aspect_query_list,
                    'S_A_ANSWER':forward_aspect_answer_list,
                    'A_O_QUERY':forward_aspect_opinion_query_list,
                    'A_O_ANSWER':forward_aspect_opinion_answer_list,
                    'AO_C_QUERY':forward_aspect_opinion_category_query_list,
                    'AO_C_ANSWER':forward_aspect_opinion_category_answer_list,
                    'A_C_QUERY':forward_aspect_category_query_list,
                    'A_C_ANSWER':forward_aspect_category_answer_list,
                    'AC_O_QUERY':forward_aspect_category_opinion_query_list,
                    'AC_O_ANSWER':forward_aspect_category_opinion_answer_list,

                    'AOC_S_QUERY':forward_pol_query_list,
                    'AOC_S_ANSWER':forward_pol_answer_list,

                    'S_O_QUERY':forward_opinion_query_list,
                    'S_O_ANSWER':forward_opinion_answer_list,
                    'O_A_QUERY':forward_opinion_aspect_query_list,
                    'O_A_ANSWER':forward_opinion_aspect_answer_list,
                    'OA_C_QUERY': forward_opinion_aspect_category_query_list,
                    'OA_C_ANSWER': forward_opinion_aspect_category_answer_list,
                    'O_C_QUERY':forward_opinion_category_query_list,
                    'O_C_ANSWER':forward_opinion_category_answer_list,
                    'OC_A_QUERY': forward_opinion_category_aspect_query_list,
                    'OC_A_ANSWER': forward_opinion_category_aspect_answer_list,

                    'S_C_QUERY':forward_category_query_list,
                    'S_C_ANSWER':forward_category_answer_list,
                        }
                     }
                sample_list.append(
                    sample_dic
                    )
            with open(dataset_name+dataset_type+'.json', 'w+') as file:
                json.dump(sample_list, file, indent=2, separators=(',', ': '))
#                print(1)

