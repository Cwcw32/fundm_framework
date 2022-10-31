# @Author: Bufan Xu,     Contact: 294594605@qq.com
# @Date:   2022-10-05
# @Do: 针对4元组数据集的数据预处理，部分代码来自ASQP源文件

import csv
import operator
import pickle
import sys
import json

from tqdm import tqdm

# import torch

"""
forward_aspect_query_template = ["[CLS]", "what", "aspects", "?", "[SEP]"]
forward_opinion_query_template = ["[CLS]", "what", "opinion", "given", "the", "aspect", "?", "[SEP]"]
backward_opinion_query_template = ["[CLS]", "what", "opinions", "?", "[SEP]"]
backward_aspect_query_template = ["[CLS]", "what", "aspect", "does", "the", "opinion", "describe", "?", "[SEP]"]
sentiment_query_template = ["[CLS]", "what", "sentiment", "given", "the", "aspect", "and", "the", "opinion", "?",]
"""

"""
数据样例
keep up the good work .####[['NULL', 'restaurant general', 'positive', 'good']]

"""

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity
            c = senttag2opinion[tri[2]]           # 'POS' -> 'good'

            one_tri = f"It is {c} because {a} is {b}"
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets


def get_para_tasd_targets(sents, labels):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets


def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    task = 'asqp'
    if task == 'aste':
        targets = get_para_aste_targets(sents, labels)
    elif task == 'tasd':
        targets = get_para_tasd_targets(sents, labels)
    elif task == 'asqp':
        targets = get_para_asqp_targets(sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets

if __name__ == '__main__':
    home_path = "../../data/uni/semeval_4yuanzu_EN/"
    dataset_name_list = ['r15_asqp','r16_asqp']  # ,'rest16']
    dataset_type_list = ["dev", "train", "test"]
    ID = 0
    for dataset_name in dataset_name_list:  #: tqdm(train_loader, desc='Train Iteration:')
        for dataset_type in tqdm(dataset_type_list, desc=dataset_name + '_'):
            # 保存为pt格式
            output_path = "../../data/uni/semeval_4yuanzu_EN/preprocess/" + dataset_name + "_" + dataset_type + "_QAS1.pt"
            # 读取原数据（CSV格式）
            # filenameTSV1='../../data/uni/semeval_4yuanzu_EN/laptop/laptop_quad_dev.tsv'
            filenameTSV1 = '../../data/uni/semeval_4yuanzu_EN/' + dataset_name + '/' + dataset_type + '.txt'
            with open(filenameTSV1, "r", encoding="utf-8") as f:
                inputs, targets = get_transformed_io(filenameTSV1)      # inputs 就是对应的text原句 ,target是用[SSEP]分割开的标准句子（... IS ... BEACUASE ... IS ...)这种形式

                for i in range(len(inputs)):
                    # change input and target to two strings
                    input = ' '.join(inputs[i])
                    target = targets[i]
                    print(1)
                    # tokenized_input = self.tokenizer.batch_encode_plus(
                    #     [input], max_length=self.max_len, padding="max_length",
                    #     truncation=True, return_tensors="pt"
                    # )
                    # tokenized_target = self.tokenizer.batch_encode_plus(
                    #     [target], max_length=self.max_len, padding="max_length",
                    #     truncation=True, return_tensors="pt"
                    # )
                    #
                    # self.inputs.append(tokenized_input)
                    # self.targets.append(tokenized_target)



#             with open(dataset_name + dataset_type + '.json', 'w+') as file:
#                 json.dump(sample_list, file, indent=2, separators=(',', ': '))
# #                print(1)

