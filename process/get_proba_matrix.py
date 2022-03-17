import numpy as np
from tqdm import trange

from distribution.gaussian import get_mean, get_variance, get_gaussian_likelihood


def get_prob_matrix_discrete(train_x, train_y, pic_size: (int, int), class_number: int, bins_number: int):
    train_amount = train_x.shape[0]
    all_pix_num = pic_size[0] * pic_size[1]
    prob_matrix = np.zeros((class_number, all_pix_num, bins_number))
    base_bin = (256 // bins_number)
    epoch = trange(0, train_amount, dynamic_ncols=True)
    for pic_index in epoch:
        epoch.set_description(f"getting prob matrix discrete {pic_index} (count bins)")
        now_pic_label = train_y[pic_index]
        for pixel_index in range(all_pix_num):
            pixel_value = train_x[pic_index, pixel_index]
            to_bin = int(pixel_value) // base_bin
            prob_matrix[now_pic_label, pixel_index, to_bin] += 1  # 在所屬的bin加一計數

    epoch = trange(0, class_number, dynamic_ncols=True)
    for now_focus_class in epoch:
        epoch.set_description(f"getting prob matrix discrete {pic_index} (aggregation)")
        for pixel_index in range(all_pix_num):
            count = 0
            for now_focus_bin in range(bins_number):
                count += prob_matrix[now_focus_class, pixel_index, now_focus_bin]
            if count == 0:
                print(f"have 0: [{now_focus_class},{pixel_index},{now_focus_bin}]")
            prob_matrix[now_focus_class, pixel_index, :] = prob_matrix[now_focus_class, pixel_index, :] / count

    return prob_matrix


def get_prob_matrix_continuous(train_x, train_y, pic_size: (int, int), class_num: int, pix_level=256):
    train_amount = train_x.shape[0]
    all_pix_num = pic_size[0] * pic_size[1]
    prob_matrix = np.zeros((class_num, all_pix_num, pix_level))
    epoch = trange(0, class_num, dynamic_ncols=True)
    for now_focus_class in epoch:
        epoch.set_description(f"getting prob matrix conti {now_focus_class} class")
        all_data_on_class = train_x[train_y == now_focus_class]   # (#data on focus class, 28*28)
        for pix_index in range(all_pix_num):
            all_value_on_focus_pixel = all_data_on_class[:, pix_index]  # (#data on focus class)
            mean_on_focus_pixel = get_mean(all_value_on_focus_pixel)
            var_on_focus_pixel = get_variance(all_value_on_focus_pixel)
            for pix_level_index in range(pix_level):
                likelihood = get_gaussian_likelihood(pix_level_index, mean_on_focus_pixel, var_on_focus_pixel)
                prob_matrix[now_focus_class, pix_index, pix_level_index] = likelihood

    return prob_matrix
