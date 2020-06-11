from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime, timezone

from tensorflow import keras


class MyFashionMnist(object):
    def train(self):
        print("TensorFlow version: ", tf.__version__)
        
        #하이퍼파라미터를 입력받기위한 argparse 라이브러리
        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--epochs', default=10, type=int)
        args = parser.parse_args()
        
        # fashion_mnist DATASET 불러오기
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
           
        #모델구조설정
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        print("Training...")
        # model.fit(train_images, train_labels, epochs=5)        
        katib_metric_log_callback = KatibMetricLog()
        training_history = model.fit(train_images,train_labels, batch_size=64,epochs=args.epochs,
                                     validation_split=0.1,
                                     callbacks=[katib_metric_log_callback])
        
        print("\ntraining_history:", training_history.history)
        
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        
        results = model.evaluate(test_images, test_labels,verbose=2)
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
