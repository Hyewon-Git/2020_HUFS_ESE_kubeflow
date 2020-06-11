
#

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import matplotlib.pyplot as plt

import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras

import tensorflow_model_optimization as tfmot

print(tf.__version__) # 버전은 2.0.0이상

#
class MyFashionMnist(object):
    def train(self):
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

#
if __name__=='__main__':
    local_train = MyFashionMnist()
    local_train.train()

#
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils

        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        fairing.config.set_builder(
            'append',
            image_name='fairing-job',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-gpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 2, memory 5GiB
        fairing.config.set_deployer('job',
                                    namespace='dudaji',
                                    pod_spec_mutators=[
                                        k8s_utils.get_resource_mutator(cpu=2,
                                                                       memory=5)]
         
                                   )
        fairing.config.run()
    else:
        remote_train = MyFashionMnist()
        remote_train.train()
