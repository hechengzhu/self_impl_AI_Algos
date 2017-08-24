#!/usr/bin/env python3

import random


class FileOps(object):
    def __init__(self):
        self.__dict = {"Iris-setosa":0, "Iris-versicolor":1}

    def __read_file_into_list(self, filename):
        file_contents = []
        with open(filename, "r+") as f:
            file_contents = f.readlines()

        return file_contents

    def __random_shuffle(self, X, y):
        X_len = len(X)

        # Repeat for array_length times
        for i in range(X_len):
            index1 = (int)(random.random() * X_len)
            index2 = (int)(random.random() * X_len)
            X[index1], X[index2] = X[index2], X[index1]
            y[index1], y[index2] = y[index2], y[index1]

        return X, y


    def normalize_data(self, filename):
        file_contents = self.__read_file_into_list(filename)

        X = []
        y = []
        for i in range(len(file_contents)):
            data_line = file_contents[i].split(",")
            temp_data = data_line[:-1]
            X.append([float(temp_data[i]) for i in range(len(temp_data))])
            y.append(self.__dict[data_line[-1].strip("\n")])

        return self.__random_shuffle(X, y)



if __name__ == '__main__':
    file_ops = FileOps()

    print(file_ops.normalize_data("./training_datasets.txt"))

