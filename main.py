import numpy as np

from config.constant import FILE_PATH
from dataprocess.dataset import DataSet
from process.get_imagination import get_imagination_discrete, get_imagination_continuous
from process.get_posterior import get_posterior_discrete, get_posterior_continuous
from process.get_prior import get_prior
from process.get_proba_matrix import get_prob_matrix_discrete, get_prob_matrix_continuous

toggle_option = input("Enter toggle option(0: discrete/1:continuous):")

dataset = DataSet(FILE_PATH)

# get training/testing data (pics, labels)
train_x, train_y = dataset.get_training_data()
test_x, test_y = dataset.get_testing_data()

if toggle_option == "0":
    # discrete
    # TODO: Get probability matrix
    prob_matrix_discrete = get_prob_matrix_discrete(train_x, train_y, (28,28), 10, 32)
    # TODO: Get prior
    prior = get_prior(train_y, 10)
    # TODO: Get and print posterior
    # prob = get_posterior_discrete(test_x, test_y, prior, prob_matrix_discrete, 32)
    # TODO: print the pic based on likelihood
    get_imagination_discrete(prob_matrix_discrete, class_num=10, pic_size=(28, 28))
    print()
else:
    # continuous
    # TODO: Get probability matrix
    prob_matrix_continuous = get_prob_matrix_continuous(train_x, train_y, (28,28), 10, 256)
    prior = get_prior(train_y, 10)
    prob = get_posterior_continuous(test_x, test_y, prior, prob_matrix_continuous)
    get_imagination_continuous(prob_matrix_continuous, class_num=10, pic_size=(28, 28))
    print()
    # TODO: Get prior
    # TODO: Get and print posterior
    # TODO: print the pic based on likelihood


