import codecs
import numpy as np
from tqdm import tqdm, trange
from config.constant import TRAIN_X_FILE_NAME, TRAIN_Y_FILE_NAME, TEST_X_FILE_NAME, TEST_Y_FILE_NAME


class DataSet:
    def __init__(self, file_path):
        with open(file_path+TRAIN_X_FILE_NAME, "rb") as f:
            print("==Training Pic Data Loading==")
            self.train_pic_data = f.read()
            self.train_pic_amount = self._hex_to_int(self.train_pic_data[4:8])
            self.train_rows_length = self._hex_to_int(self.train_pic_data[8:12])
            self.train_cols_length = self._hex_to_int(self.train_pic_data[12:16])
            print(f"->pic amount: {self.train_pic_amount}")
            print(f"->pic rows_length:{self.train_rows_length}")
            print(f"->pic cols_length:{self.train_cols_length}")
            print("==Training Pic Data Loading complete==")

        with open(file_path+TRAIN_Y_FILE_NAME, "rb") as f:
            print("==Training Label Data Loading==")
            self.train_label_data = f.read()
            print("==Training Label Data Loading complete==")

        with open(file_path+TEST_X_FILE_NAME, "rb") as f:
            print("==Testing Pic Data Loading==")
            self.test_pic_data = f.read()
            self.test_pic_amount = self._hex_to_int(self.test_pic_data[4:8])
            self.test_rows_length = self._hex_to_int(self.test_pic_data[8:12])
            self.test_cols_length = self._hex_to_int(self.test_pic_data[12:16])
            print(f"->pic amount: {self.test_pic_amount}")
            print(f"->pic rows_length:{self.test_rows_length}")
            print(f"->pic cols_length:{self.test_cols_length}")
            print("==Testing Pic Data Loading complete==")

        with open(file_path+TEST_Y_FILE_NAME, "rb") as f:
            print("==Testing Label Data Loading==")
            self.test_label_data = f.read()
            print("==Testing Label Data Loading complete==")


    def get_training_data(self):
        train_x = np.zeros((self.train_pic_amount, self.train_rows_length * self.train_cols_length), dtype="uint8")
        load_pointer = 16
        epochs = trange(0, self.train_pic_amount, dynamic_ncols=True)

        train_pic_data = self.train_pic_data

        for pic_index in epochs:
            epochs.set_description(f"now process {pic_index} Pic (Training)")
            for pixel_index in range(self.train_rows_length * self.train_cols_length):
                train_x[pic_index, pixel_index] = int.from_bytes(train_pic_data[load_pointer:load_pointer + 1], byteorder="big")
                load_pointer += 1

        train_y = np.zeros(self.train_pic_amount, dtype="uint8")
        load_pointer = 8
        epochs = trange(0, self.train_pic_amount, dynamic_ncols=True)

        train_label_data = self.train_label_data

        for pic_index in epochs:
            epochs.set_description(f"now process {pic_index} Label (Training)")
            train_y[pic_index] = int.from_bytes(train_label_data[load_pointer:load_pointer + 1], byteorder="big")
            load_pointer += 1
        return train_x, train_y


    def get_testing_data(self):
        test_x = np.zeros((self.test_pic_amount, self.test_rows_length * self.test_cols_length), dtype="uint8")
        load_pointer = 16
        epochs = trange(0, self.test_pic_amount, dynamic_ncols=True)

        test_pic_data = self.test_pic_data

        for pic_index in epochs:
            epochs.set_description(f"now process {pic_index} Pic (Testing)")
            for pixel_index in range(self.train_rows_length * self.train_cols_length):
                test_x[pic_index, pixel_index] = int.from_bytes(test_pic_data[load_pointer:load_pointer + 1], byteorder="big")
                load_pointer += 1

        test_y = np.zeros(self.test_pic_amount, dtype="uint8")
        load_pointer = 8
        epochs = trange(0, self.test_pic_amount, dynamic_ncols=True)

        test_label_data = self.test_label_data

        for pic_index in epochs:
            epochs.set_description(f"now process {pic_index} Label")
            test_y[pic_index] = int.from_bytes(test_label_data[load_pointer:load_pointer + 1], byteorder="big")
            load_pointer += 1
        return test_x, test_y

    def _hex_to_int(self,data):
        return int(codecs.encode(data, "hex"),16)

if __name__ == "__main__":
    a = DataSet("../data/")
    train_x, train_y = a.get_training_data()
    test_x, test_y = a.get_testing_data()
    print()
