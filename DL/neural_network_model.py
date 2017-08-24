#!/usr/bin/env python3

import math
import random
from preprocess_data import FileOps


class NeuralNetwork(object):
    def __init__(self, layers_num, outputs_k, learning_rate, epoch):
        self.__layers_num = layers_num
        self.__outputs_k = outputs_k
        self.__learning_rate = learning_rate
        self.__epoch = epoch

    def fit(self, training_X, training_y):
        # fie steps
        # 1. Set up the model architecture
        #    First Layer :  X0 = 1, x1, x2, x3, x4
        #    Second Layer:  A0 = 1, a1, a2, a3, a4, a5
        #    Third Layer :  c
        first_layer_num = 4
        second_layer_num = 5
        third_layer_num = 1

        # 2. Initial the original weights, w21, w32
        w21, w32 = self.__initialize_weights(first_layer_num,
                          second_layer_num, third_layer_num)

        # Loop for epoch times
        for epo in range(self.__epoch):
           cost_value = 0.0
           for i in range(len(training_X)):
               # 3. Forward pass to compute the cost function
               c_hat, loss_value, X_vector, hidden_vector = self.__forward_pass(
                       training_X[i], training_y[i], w21, w32, epo)
               cost_value += loss_value * loss_value

               # 4. Back propagation to compute the partial derivatives
               # 4.1. Compute the output layer partial derivatives
               w32_updated = []
               w32_deriv = []
               for j in range(len(w32)):
                   temp_w32_updated = []
                   for k in range(len(w32[j])):
                       activition_deriv = 0.0
                       if c_hat > 0:
                           activition_deriv = 1
                       c_derivative = loss_value * c_hat * (1 - c_hat) * hidden_vector[k]
                       temp_w32_updated.append(w32[j][k] - self.__learning_rate 
                                * c_derivative)
                   w32_deriv = temp_w32_updated[:]
                   w32_updated.append(temp_w32_updated)

               # 4.2. Compute the hiddle layer partial derivatives
               w21_updated = []
               for j in range(len(w21)):
                   temp_w21_updated = []
                   for k in range(len(w21[j])):
                       activition_deriv = 0
                       if hidden_vector[j] > 0:
                           activition_deriv = 1
                       h_derivative = w32_deriv[j] * hidden_vector[j] * (1 - hidden_vector[j]) * X_vector[k]
                       temp_w21_updated.append(h_derivative)
                   w21_updated.append(temp_w21_updated)

               # 5. Update the weights
               w32 = w32_updated[:]
               w21 = w21_updated[:]
           print("Iteration " + repr(epo + 1) + ": cost value = " + 
                 repr(cost_value))

        return w21, w32

    def predict(self, test_X, w21, w32):
        for i in range(len(test_X)):
            X_vector = test_X[i][:]
            X_vector.insert(0,1)
            X_vector = [float(X_vector[p]) for p in range(len(X_vector))]
            a = []
            for j in range(len(w21)):
                count = 0.0
                for k in range(len(w21[j])):
                    count += self.__activation(X_vector[k] * w21[j][k])
                a.append(count)

            a.insert(0,1)
            out_value = 0.0
            for j in range(len(w32)):
                count = 0.0
                for k in range(len(w32[j])):
                    count += a[k] * w32[j][k]
                out_value = self.__sigmoid_activation(count)
            print(out_value)


    def __forward_pass(self, training_X, training_y, w21, w32, epo):
        # Add bias term x0 = 1 here
        # 1. Compute value for the hidden layer
        X_vector = training_X[:]
        X_vector.insert(0, 1)
        hidden_layer_vector = []
        for j in range(len(w21)):
            temp_a = 0.0
            for k in range(len(w21[j])):
                temp_a += X_vector[k] * w21[j][k]
            # Activation function
            temp_a = self.__sigmoid_activation(temp_a)
            hidden_layer_vector.append(temp_a)

        # 2. Compute value for the output layer
        hidden_layer_vector.insert(0, 1)
        loss_value = 0.0
        output_value = 0.0
        for j in range(len(w32)):
            temp_c = 0.0
            for k in range(len(w32[j])):
                temp_c += hidden_layer_vector[k] * w32[j][k]
            temp_c = self.__sigmoid_activation(temp_c)
            output_value = temp_c
            loss_value = temp_c - training_y

        return output_value, loss_value, X_vector, hidden_layer_vector

    def __sigmoid_activation(self, value):
        return 1.0 / (1 + math.exp(-1 * value))

    def __activation(self, value):
        # Just use ReLu, may try Leaky ReLu later
        if value < 0:
            return 0
        return value

    def __initialize_weights(self, first_layer_num, second_layer_num,
                             third_layer_num):
        w21 = []
        for i in range(second_layer_num):
            temp = []
            for j in range(first_layer_num + 1):
                temp.append(random.random())
            w21.append(temp)

        w32 = []
        for i in range(third_layer_num):
            temp=[]
            for j in range(second_layer_num + 1):
                temp.append(random.random())
            w32.append(temp)

        return w21, w32


if __name__ == "__main__":
    file_ops = FileOps()
    training_X, training_y = file_ops.normalize_data("./training_datasets.txt")
    test_X, test_y = file_ops.normalize_data("./test_datasets.txt")

    neural_network = NeuralNetwork(3, 3, 0.0005, 3500)
    w21, w32 = neural_network.fit(training_X, training_y)

    neural_network.predict(test_X, w21, w32)
