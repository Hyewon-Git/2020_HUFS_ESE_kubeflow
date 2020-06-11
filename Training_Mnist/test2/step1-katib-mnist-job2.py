from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime, timezone

from tensorflow import keras

def train():
    print("TensorFlow version: ", tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    args = parser.parse_args()

    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    model = tf.keras.models.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)

          
#       tf.keras.layers.Flatten(input_shape=(28, 28)),
#       tf.keras.layers.Dense(128, activation='relu'),
#       tf.keras.layers.Dropout(args.dropout),
#       tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    print("Training...")

    katib_metric_log_callback = KatibMetricLog()
    training_history = model.fit(x_train, y_train, batch_size=64, epochs=10,
                                 validation_data=(x_val, y_val),
                                 callbacks=[katib_metric_log_callback])

    print("\\ntraining_history:", training_history.history)

    # Evaluate the model on the test data using `evaluate`
    print('\\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)


class KatibMetricLog(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # RFC 3339
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\\nEpoch {}".format(epoch+1))
        print("{} accuracy={:.4f}".format(local_time, logs['acc']))
        print("{} loss={:.4f}".format(local_time, logs['loss']))
        print("{} Validation-accuracy={:.4f}".format(local_time, logs['val_acc']))
        print("{} Validation-loss={:.4f}".format(local_time, logs['val_loss']))


if __name__ == '__main__':
    train()
