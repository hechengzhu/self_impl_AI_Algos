#!/usr/bin/env python3

import math
import random


class DataGenerator(object):
    def __init__(self):
        pass


    def generate_training_datasets(self):
        # Just hard code here

        results = []
        for i in range(50000):
            results.append([(i+1)/10000.0, ((i+1) * 0.900099)/10000])

        for i in range(50000):
            results.append([(i+1)/10000.0, ((i+1) * 1.00001)/10000])

        return results

    def generate_test_datasets(self):
        data = self.generate_training_datasets()
        test_datasets = []
        for i in range(1000):
            test_datasets.append(data[(int)(random.random() * 1000)])

        return test_datasets



class LinearRegression(object):
    def __init__(self):
        pass

    def train_model(self, training_datasets, learning_rate, iteration):
        pass

        # 1. Random initialize w0 and w1
        original_w0 = w0 = random.random()
        original_w1 = w1 = random.random()

        for i in range(iteration):
            #2. Forward to calculate the J(Wi)
            square_error = 0
            for j in range(len(training_datasets)):
                X, y = training_datasets[j]
                square_error += 0.5 * ((w0 * 1 + w1 * X) - y) * ((w0 * 1 + w1 * X) - y)

            #print("===>Iteration " + repr(i+1) + ": the squared_error is ||" + repr(square_error) + "||")

            # 3. Back propagation to calculate the derivative
            for k in range(len(training_datasets)):
                X, y = training_datasets[k]
                temp0 = learning_rate * ((w0 + w1 * X) - y )
                temp1 = learning_rate * ((w0 + w1 * X) - y) * X

                # 4. Updates the w0 and w1
                w0 = w0 - temp0
                w1 = w1 - temp1

        return w0, w1

    def predict(self, w0, w1, datasets):
        predict_results = []
        for i in range(len(datasets)):
            y = w0 * 1 + w1 * datasets[i]
            predict_results.append(y)

        return predict_results

if __name__ == '__main__':
    data_generator = DataGenerator()
    training_datasets = data_generator.generate_training_datasets()
    test_datasets = data_generator.generate_test_datasets()

    linear_model = LinearRegression()

    # Try different learning retes, will not use momentum, Adam, or RMSProp method .etc
    learning_rates = [0.00001, 0.00002, 0.00005, 0.00007, 0.0001, 0.0002, 0.0005, 0.001, 0.01, 0.02, 0.05]
    for i in range(len(learning_rates)):
        w0, w1 = linear_model.train_model(training_datasets, learning_rates[i], 20)
        print("==" + repr(i+1) + "==>Larning_rate = ||" + repr(learning_rates[i]) + "||, w0 = ||" + repr(w0) + "||, w1 = ||" + repr(w1) + "||")
        print("-"* 50)
        predict_test_datasets_value = linear_model.predict(w0, w1, [test_datasets[k][0] for k in range(len(test_datasets))])
        accuracy_error = 0
        test_datasets_y = [test_datasets[k][1] for k in range(len(test_datasets))]
        for j in range(len(test_datasets_y)):
            if (math.fabs(test_datasets_y[j] - predict_test_datasets_value[j]) >= 0.08):
                accuracy_error += 1
        print("Accuracy = ||" + repr(1 - accuracy_error * 1.0 / len(test_datasets_y)) + "||")

