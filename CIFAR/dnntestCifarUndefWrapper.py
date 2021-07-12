# Copyright 2020 Max Planck Institute for Software Systems

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from sys import argv
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)

class MyModel:
    def __init__(self):
        self.model = load_model('/data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deepsearch2.h5')
    def predict(self, images):
        return np.array(self.model(images - x_train_mean))

mymodel = MyModel()

if __name__ == '__main__':
    with open('cifar_indices.txt', 'r') as f:
        target_set = [int(i) for i in f.readlines()]

    import os
    import pickle
    attacked_path = '/data/dnntest/zpengac/DeepSearch/CIFAR/DSBatched/2021-07-04 10:39:24.088576'
    
    #x_test = []
    #for i in target_set:
    #    img = pickle.load(open(os.path.join(attacked_path, 'image_{i}.pkl'.format(i=i)), 'rb'))
    #    img = img.reshape(32, 32, 3)
    #    x_test.append(img)
    #x_test = np.stack(x_test)
    
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    mymodel.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #y_test = y_test[target_set]
    print(mymodel.model.evaluate(x_test - x_train_mean, y_test))
