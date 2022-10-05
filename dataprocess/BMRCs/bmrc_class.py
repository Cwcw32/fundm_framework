import os
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