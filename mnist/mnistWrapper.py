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
from PIL import Image
import numpy as np
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def mnist_preprocessing(x):
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(32, 32))
        img = np.asarray(img).astype(np.float32) / 255.0
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x

x_train = mnist_preprocessing(x_train)
x_test = mnist_preprocessing(x_test)

x_train_mean = np.mean(x_train, axis=0)
#x_test -= x_train_mean

#import imp
#MainModel = imp.load_source('MainModel', "/data/dnntest/zpengac/models/resnet20/resnet20.py")
#mymodel = torch.load("/data/dnntest/zpengac/models/resnet20/resnet20.pt")
class MyModel:
    def __init__(self):
        self.model = load_model('/data/dnntest/zpengac/models/lenet/mnist_lenet5_keras_32_py2.h5')
    def predict(self, images):
        return np.array(self.model(images - x_train_mean))

mymodel = MyModel()

#x_test = torch.tensor(x_test, dtype=torch.float)
#x_test = x_test.reshape((10000, 3, 32, 32))
#x_test = cifar_preprocessing(x_test)
#y_test = torch.tensor(y_test)

#predicted = mymodel(x_test)
#predicted = predicted.argmax(dim=1)
#y_test = torch.tensor(y_test)
#acc = torch.sum(predicted == y_test) / 10000
#print(acc)

#mymodel=CompatModel()

if __name__ == '__main__':
    with open('mnist_indices.txt', 'r') as f:
        target_set = [int(i) for i in f.readlines()]
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    mymodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    x_test = x_test[target_set]
    y_test = y_test[target_set]
    print(mymodel.evaluate(x_test, y_test))
