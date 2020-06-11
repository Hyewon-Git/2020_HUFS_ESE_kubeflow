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
        
        #하이퍼파라미터를 입력받기위한 argparse 라이브러리
        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--dropout', default=0.2, type=float)
        args = parser.parse_args()
        
        # fashion_mnist DATASET 불러오기
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        # Reserve 10,000 samples for validation
        x_val = train_images[-10000:]
        y_val = train_labels[-10000:]
        x_train = train_images[:-10000]
        y_train = train_labels[:-10000]
           
        #모델구조설정
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        print("Training...")
        # model.fit(train_images, train_labels, epochs=5)        
        katib_metric_log_callback = KatibMetricLog()
        training_history = model.fit(x_train, y_train, batch_size=64, epochs=10,
                                     validation_data=(x_val, y_val),
                                     callbacks=[katib_metric_log_callback])
        
        print("\ntraining_history:", training_history.history)
        
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        
        results = model.evaluate(test_images, test_labels, batch_size=128)
        print('test loss, test acc:', results)
        
        
class KatibMetricLog(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # RFC 3339
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\nEpoch {}".format(epoch+1))
        print("{} accuracy={:.4f}".format(local_time, logs['acc']))
        print("{} loss={:.4f}".format(local_time, logs['loss']))
        print("{} Validation-accuracy={:.4f}".format(local_time, logs['val_acc']))
        print("{} Validation-loss={:.4f}".format(local_time, logs['val_loss']))

        
if __name__=='__main__':
    local_train = MyFashionMnist()
    local_train.train()
