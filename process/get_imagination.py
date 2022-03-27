import numpy as np


def get_imagination_discrete(prob_matrix, class_num=10, pic_size=(28,28)):
    # according to likelihood
    for now_focus_class in range(class_num):
        print(f"{now_focus_class}:")
        for row_index in range(pic_size[0]):
            for col_index in range(pic_size[1]):
                highest_bin = np.argmax(prob_matrix[now_focus_class, row_index*28+col_index])
                # assumption: all bins->32, over half value(16) is to be 1
                if highest_bin >= 16:
                    print(1, end=" ")
                else:
                    print(0, end=" ")
            print()
        print()



def get_imagination_continuous(prob_matrix, class_num=10, pic_size=(28,28)):
    # according to likelihood
    for now_focus_class in range(class_num):
        print(f"{now_focus_class}:")
        for row_index in range(pic_size[0]):
            for col_index in range(pic_size[1]):
                highest_value = np.argmax(prob_matrix[now_focus_class, row_index * 28 + col_index])
                # assumption: over half value(128) is to be 1
                if highest_value >= 128:
                    print(1, end=" ")
                else:
                    print(0, end=" ")
            print()
        print()