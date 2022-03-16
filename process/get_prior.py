import numpy as np


def get_prior(train_y, class_num: int):
    prior = np.zeros(class_num)
    train_amount = train_y.shape[0]
    for now_focus_class in range(class_num):
        prior[now_focus_class] = np.sum(train_y == now_focus_class) / train_amount
    return prior