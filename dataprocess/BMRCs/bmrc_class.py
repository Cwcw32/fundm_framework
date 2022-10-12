import os


# 本class为通用版本
class QueryAndAnswer:
    def __init__(self,

                 text=None,

                 forward_aspect_query_list=None,
                 forward_aspect_answer_list=None,
                 forward_aspect_opinion_query_list=None,
                 forward_aspect_opinion_answer_list=None,

                 forward_pol_query_list=None,
                 forward_pol_answer_list=None,

                 forward_opinion_query_list=None,
                 forward_opinion_answer_list=None,
                 forward_opinion_aspect_query_list=None,
                 forward_opinion_aspect_answer_list=None

                 ):
        self.text=text
        self.forward_aspect_query_list = []  # 从aspect开始的一系列问题
        self.forward_aspect_answer_list = []  # 与上面这个对应的相应的答案，0为不是答案，1为是答案
        self.forward_aspect_opinion_query_list = []
        self.forward_aspect_opinion_answer_list = []
        self.forward_pol_query_list = []
        self.forward_pol_answer_list = []
        self.forward_opinion_query_list = []  # 从opinion开始的一系列问题
        self.forward_opinion_answer_list = []  # 与上面这个对应的相应的答案，0为不是答案，1为是答案
        self.forward_opinion_aspect_query_list = []
        self.forward_opinion_aspect_answer_list = []
