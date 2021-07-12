import pickle
import os
from tensorflow import keras
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)

output_path = '/data/dnntest/zpengac/DeepSearch/CIFAR/DSBatched/3'

with open('cifar_indices.txt', 'r') as f:
    target_set = [int(i) for i in f.readlines()]

result_l2 = 0
result_linf = 0
for i in target_set:
    new_img = pickle.load(open(os.path.join(output_path, f'image_{i}.pkl'), 'rb'))
    new_img = new_img * 255
    new_img = new_img.astype('uint8').astype('float32')
    #for i in range(3):
    #    new_img[:, :, :, i] = new_img[:, :, :, i] * std[i] + mean[i]
    img = x_test[i:i+1].astype('float32')
    if i == 99:
        img_file = Image.fromarray(new_img.reshape(32,32,3).astype('uint8'), 'RGB')
        img_file.save('example3.bmp')
        orig_file = Image.fromarray(img.reshape(32,32,3).astype('uint8'), 'RGB')
        orig_file.save('example_orig3.bmp')
    l2 = np.sqrt(np.sum(np.square(new_img - img)))
    linf = np.max(np.abs(new_img - img))
    result_l2 += l2
    result_linf += linf

print(result_l2 / len(target_set))
print(result_linf / len(target_set))

