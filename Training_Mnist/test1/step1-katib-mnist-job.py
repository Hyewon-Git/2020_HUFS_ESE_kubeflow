from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tempfile
import os

import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime, timezone

from tensorflow import keras

class MyMnist(object):
  def train(self):
    print("TensorFlow version: ", tf.__version__)
    
    #하이퍼파라미터를 입력받기위한 argparse 라이브러리
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Reserve 10,000 samples for validation
    x_val = train_images-10000:]
    y_val =train_labels[-10000:]
    train_images = train_images[:-10000]
    train_labels = train_labels[:-10000]

    # Define the model architecture.모델구조설정
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    model.summary()

    # Train the digit classification model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Training...")
    # model.fit(train_images, train_labels, epochs=5)        
    katib_metric_log_callback = KatibMetricLog()
    training_history = model.fit(train_images,train_labels, batch_size=64,epochs=args.epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[katib_metric_log_callback])
    
    print("\\ntraining_history:", training_history.history)
        
    # Evaluate the model on the test data using `evaluate`
    print('\\n# Evaluate on test data')
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
    local_train = MyMnist()
    local_train.train()
