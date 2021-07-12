import pickle
import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from PIL import Image

#resizing = Resizing(32, 32)
#x_test = resizing(x_test).numpy()

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

#x_train = mnist_preprocessing(x_train)
#x_test = mnist_preprocessing(x_test)

#x_train_mean = np.mean(x_train, axis=0)

#x_test = x_test.astype('float32')
#x_test = np.expand_dims(x_test, -1)
#x_test -= x_train_mean

output_path = '/data/dnntest/zpengac/DeepSearch/mnist/DSBatched/2021-07-03 20:22:38.487988'

with open('mnist_indices.txt', 'r') as f:
    target_set = [int(i) for i in f.readlines()]

result_l2 = 0
result_linf = 0
resizing = Resizing(32, 32)
for i in target_set:
    new_img = pickle.load(open(os.path.join(output_path, f'image_{i}.pkl'), 'rb'))
    new_img = new_img * 255
    new_img = new_img.astype('uint8').astype('float32')
    img = x_test[i:i+1].astype('float32')
    img = resizing(img.reshape(1,28,28,1)).numpy()
    if result_l2 == 0:
        img_file = Image.fromarray(new_img.reshape(32,32).astype('uint8'), 'L')
        img_file.save('example3.bmp')
        orig_file = Image.fromarray(img.reshape(32,32).astype('uint8'), 'L')
        orig_file.save('example_orig3.bmp')
        print(np.sqrt(np.sum(np.square(new_img - img))))
        print(np.max(np.abs(new_img - img)))
    l2 = np.sqrt(np.sum(np.square(new_img - img)))
    linf = np.max(np.abs(new_img - img))
    result_l2 += l2
    result_linf += linf

print(result_l2 / len(target_set))
print(result_linf / len(target_set))

