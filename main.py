import numpy as np

from config.constant import FILE_PATH
from dataprocess.dataset import DataSet
from process.get_proba_matrix import get_prob_matrix_discrete, get_prob_matrix_continuous

toggle_option = input("Enter toggle option(0: discrete/1:continuous):")

dataset = DataSet(FILE_PATH)

# get training/testing data (pics, labels)
train_x, train_y = dataset.get_training_data()
test_x, test_y = dataset.get_testing_data()

if toggle_option == "0":
    # discrete
    # TODO: Get probability matrix
    prob_matrix_discrete = get_prob_matrix_discrete(train_x, train_y)
    # TODO: Get prior
    # TODO: Get and print posterior
    # TODO: print the pic based on likelihood
    pass
else:
    # continuous
    # TODO: Get probability matrix
    prob_matrix_continuous = get_prob_matrix_continuous(train_x, train_y)
    # TODO: Get prior
    # TODO: Get and print posterior
    # TODO: print the pic based on likelihood
    pass

