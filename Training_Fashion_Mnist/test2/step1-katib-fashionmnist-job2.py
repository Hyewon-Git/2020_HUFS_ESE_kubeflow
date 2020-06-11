import tensorflow as tf
import os
import argparse
from tensorflow.python.keras.callbacks import Callback

from tensorflow import keras

class MyFashionMnist(object):
  def train(self):
    # fashion_mnist
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 입력 값을 받게 추가합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
    parser.add_argument('--dropout_rate', required=False, type=float, default=0.2)
    # epoch 5 ~ 15
    parser.add_argument('--epoch', required=False, type=int, default=5)   
    args = parser.parse_args()    
        
    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
      keras.layers.Dropust(args.dropout_rate)
    ])
    model.summary()
   
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              verbose=2,
              validation_data=(x_test, y_test),
              epochs=args.epoch,
              callbacks=[KatibMetricLog()])

    model.evaluate(x_test,  y_test, verbose=2)

class KatibMetricLog(Callback):
    def on_batch_end(self, batch, logs={}):
        print("batch=" + str(batch),
              "accuracy=" + str(logs.get('acc')),
              "loss=" + str(logs.get('loss')))
    def on_epoch_begin(self, epoch, logs={}):
        print("epoch " + str(epoch) + ":")
    
    def on_epoch_end(self, epoch, logs={}):
        print("Validation-accuracy=" + str(logs.get('val_acc')),
              "Validation-loss=" + str(logs.get('val_loss')))
        return

if __name__ == '__main__':
  remote_train = MyFashionMnist()
  remote_train.train()
