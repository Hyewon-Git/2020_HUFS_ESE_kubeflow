from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import matplotlib.pyplot as plt
import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot

class MyFashionMnist(object):
    def train(self):
        print("TensorFlow version: ", tf.__version__)
        
        #hyper
        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--dropout', default=0.2, type=float)
        args = parser.parse_args()
        # fashion_mnist
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=5)
        model.evaluate(test_images, test_labels, verbose=2)
        
if __name__=='__main__':
    local_train = MyFashionMnist()
    local_train.train()
