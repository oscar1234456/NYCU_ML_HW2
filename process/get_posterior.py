import numpy as np
from tqdm import trange


# [[[]*class_num]*test_amount]
def get_posterior_discrete(test_x, test_y, prior, prob_matrix, bins_number: int):
    # p(c=0|測圖1) = [p(測圖1|c=0)*p(c=0)] / p(測圖1)
    # p(測圖1|c=0) = p(測圖1之pixel1|c=0) * p(測圖1之pixel2|c=0)* .... * p(測圖1之pixel784|c=0)
    # p(c=0): prior
    # p(測圖1): p(測圖1|c=0)*p(c=0) = p(測圖1, c=0), sum up w.r.t. c
    test_amount = test_x.shape[0]
    class_num = prior.shape[0]
    pic_size = test_x.shape[1]
    base_bin = (256 // bins_number)

    temp_prob = np.zeros((test_amount, class_num))

    epoch = trange(0, test_amount, dynamic_ncols=True)
    for pic_index in epoch:
        for now_focus_class in range(class_num):
            for now_focus_pixel in range(pic_size):
                # get the bin of this pixel in this testing pic
                to_bin = int(test_x[pic_index, now_focus_pixel]) // base_bin
                # count likelihood
                prob_value = prob_matrix[now_focus_class, now_focus_pixel, to_bin]
                # avoid getting 0 value
                if prob_value < 0.00001:
                    # print("prob_value < 0.0001")
                    prob_value = 0.00001
                # make log transformation (easy to multiply all prob)
                temp_prob[pic_index, now_focus_class] += np.log(prob_value)
                # notice: after log trans, all value become negative

    epoch = trange(0, test_amount, dynamic_ncols=True)
    error = 0
    for pic_index in epoch:
        # prior:[[]*class_number]
        # likelihood * prior
        temp_prob[pic_index, :] = temp_prob[pic_index, :] + np.log(prior)

        # marginal constant: sum up w.r.t. c
        marginal = np.sum(temp_prob[pic_index, :])
        # normalization with marginal constant
        temp_prob[pic_index, :] = temp_prob[pic_index, :] / marginal
        # notice: neg/neg -> pos  (argmax -> argmin)

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
    print(f"error rate:{error / test_amount}")

    return temp_prob


# [[[]*class_num]*test_amount]
def get_posterior_continuous(test_x, test_y, prior, prob_matrix):
    # p(c=0|測圖1) = [p(測圖1|c=0)*p(c=0)] / p(測圖1)
    # p(測圖1|c=0) = p(測圖1之pixel1|c=0) * p(測圖1之pixel2|c=0)* .... * p(測圖1之pixel784|c=0)
    # p(c=0): prior
    # p(測圖1): p(測圖1|c=0)*p(c=0) = p(測圖1, c=0), sum up w.r.t. c

    test_amount = test_x.shape[0]
    class_num = prior.shape[0]
    pic_size = test_x.shape[1]

    temp_prob = np.zeros((test_amount, class_num))

    epoch = trange(0, test_amount, dynamic_ncols=True)
    for pic_index in epoch:
        for now_focus_class in range(class_num):
            for now_focus_pixel in range(pic_size):
                pixel_value = int(test_x[pic_index, now_focus_pixel])  # 0-255
                prob_value = prob_matrix[now_focus_class, now_focus_pixel, pixel_value]
                # avoid likelihood too small
                if prob_value < 1e-30:
                    # print("prob_value < 1e-30")
                    prob_value = 1e-30
                # make log transformation (easy to multiply all prob)
                temp_prob[pic_index, now_focus_class] += np.log(prob_value)
                # notice: after log trans, all value become negative

    epoch = trange(0, test_amount, dynamic_ncols=True)
    error = 0
    for pic_index in epoch:
        # prior:[[]*class_number]
        # likelihood * prior
        temp_prob[pic_index, :] = temp_prob[pic_index, :] + np.log(prior)

        # marginal constant: sum up w.r.t. c
        marginal = np.sum(temp_prob[pic_index, :])
        # normalization with marginal constant
        temp_prob[pic_index, :] = temp_prob[pic_index, :] / marginal
        # notice: neg/neg -> pos  (argmax -> argmin)

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
    print(f"error rate:{error / test_amount}")

    return temp_prob
