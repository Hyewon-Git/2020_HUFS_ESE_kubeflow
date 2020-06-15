from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime, timezone
import tempfile


print("TensorFlow version: ", tf.__version__)

# tfjob으로 학습하기위해
tf_config = os.environ.get('TF_CONFIG', '{}')
print("TF_CONFIG %s", tf_config)
tf_config_json = json.loads(tf_config)
cluster = tf_config_json.get('cluster')
job_name = tf_config_json.get('task', {}).get('type')
task_index = tf_config_json.get('task', {}).get('index')
print("cluster={} job_name={} task_index={}}", cluster, job_name, task_index)

#경로설정
tb_dir = '/app/data/logs'
model_dir = '/app/data/export'
version = 1
export_dir = os.path.join(model_dir, str(version))

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


# Define the model architecture.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.144),
  tf.keras.layers.Dense(10, activation='softmax')
])


# Train the digit classification model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.193),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
print("Training...")

training_history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                             validation_split=0.1,
                             callbacks=[tf.keras.callbacks.TensorBoard(log_dir=tb_dir)])

print("\\ntraining_history:", training_history.history)
model.summary()



# Evaluate the model on the test data using `evaluate`
print('\\n# Evaluate on test data')
results = model.evaluate(test_images, test_labels, verbose=0)
print('test loss, test acc:', results)

#         모델 저장 코드 지금은 tmp위치에 저장함 -> 모델 저장소에 저장해야함
#     단, 저장할때 모델은 .h5 형태여야함으로 찾아보고 저장소에 저장할 것
_, model_file = tempfile.mkstemp('.h5') 
tf.keras.models.save_model(model, model_file, include_optimizer=False)
print('Saved Mnist model to:', model_file)

#     model_dir = '/data-vol-1'
#     version = 1
#     export_dir = os.path.join(model_dir, str(version))
#     model.save(export_dir)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





import tensorflow_model_optimization as tfmot # 꼭 필요 모듈

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

"""
hyper parameter tuning 예상
    [batch_size, epochs, initial_sparsity,
    final_sparsity, begin_step]
    
"""

batch_size = 128
epochs = 6
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.90,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


logdir = tempfile.mkdtemp()

#fine-tuning
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
 
#모델 학습시킴
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

model_for_pruning.summary()
_, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)
print('Pruned test accuracy:', model_for_pruning_accuracy)


model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

#모델 저장 코드 지금은 tmp위치에 저장함 -> 모델 저장소에 저장해야함
# 단, 저장할때 모델은 .h5 형태여야함으로 찾아보고 저장소에 저장할 것
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)
