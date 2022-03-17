import numpy as np
from tqdm import trange


def get_posterior_discrete(test_x, test_y, prior, prob_matrix, bins_number: int):
    test_amount = test_x.shape[0]
    class_num = prior.shape[0]
    pic_size = test_x.shape[1]
    base_bin = (256 // bins_number)

    temp_prob = np.zeros((test_amount, class_num))

    epoch = trange(0, test_amount, dynamic_ncols=True)
    for pic_index in epoch:
        for now_focus_class in range(class_num):
            for now_focus_pixel in range(pic_size):
                to_bin = int(test_x[pic_index, now_focus_pixel])//base_bin
                prob_value = prob_matrix[now_focus_class, now_focus_pixel, to_bin]
                if prob_value < 0.00001:
                    # print("prob_value < 0.0001")
                    prob_value = 0.00001
                temp_prob[pic_index, now_focus_class] += np.log(prob_value)

    epoch = trange(0, test_amount, dynamic_ncols=True)
    error = 0
    for pic_index in epoch:
        temp_prob[pic_index, :] = temp_prob[pic_index, :] + np.log(prior)
        marginal = np.sum(temp_prob[pic_index, :])
        temp_prob[pic_index, :] = temp_prob[pic_index, :] / marginal
        print("Posterior (in log scale):")
        for now_focus_class in range(class_num):
            print(f"{now_focus_class}: {temp_prob[pic_index, now_focus_class]}")
        predict = np.argmin(temp_prob[pic_index])
        gt = test_y[pic_index]
        print(f"Prediction:{predict},", end="")
        print(f"Ans: {gt}")
        print()
        if predict != gt:
            error += 1
    print(f"error rate:{error/test_amount}")

    return temp_prob





