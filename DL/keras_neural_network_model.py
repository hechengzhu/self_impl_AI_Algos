#!/usr/bin/env python3


from keras.models import Sequential
from keras.layers import Dense
import numpy as np


if __name__ == '__main__':
    np.random.seed(7)

    dataset = np.loadtxt('./training_datasets_for_keras.txt', delimiter=',')

    X = dataset[:, 0:5]
    Y = dataset[:,-1]

    model = Sequential()
    model.add(Dense(5, input_dim=4, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    scores = model.evaluate([[5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5]], [0,0,1,1])
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
