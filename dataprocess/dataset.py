import codecs


class DataSet:
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = f.read()
            print (int(codecs.encode(self.data[4:8], 'hex'), 16))
    def


if __name__ == "__main__":
    a = DataSet("../train-images.idx3-ubyte")
    print()