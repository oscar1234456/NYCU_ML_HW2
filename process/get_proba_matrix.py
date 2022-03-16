import numpy as np
from tqdm import trange


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
            to_bin = int(pixel_value)//base_bin
            prob_matrix[now_pic_label, pixel_index, to_bin] += 1  # 在所屬的bin加一計數

    epoch = trange(0, class_number, dynamic_ncols=True)
    for now_focus_class in epoch:
        epoch.set_description(f"getting prob matrix discrete {pic_index} (aggregation)")
        for pixel_index in range(all_pix_num):
            count = 0
            for now_focus_bin in range(bins_number):
                count += prob_matrix[now_focus_class, pixel_index, now_focus_bin]
            if count==0:
                print(f"have 0: [{now_focus_class},{pixel_index},{now_focus_bin}]")
            prob_matrix[now_focus_class, pixel_index, :] = prob_matrix[now_focus_class, pixel_index, :] / count

    return prob_matrix




def get_prob_matrix_continuous(train_x, train_y):
    pass
