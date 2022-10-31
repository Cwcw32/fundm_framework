# 计算熵
from math import log
from collections import Counter

def entropy(y_label):
    """
    :param y_label:
    :return:
    """
    counter = Counter(y_label)
    ent = 0.0
    for num in counter.values():
        p = num / len(y_label)
        ent += -p * log(p)
    return ent
