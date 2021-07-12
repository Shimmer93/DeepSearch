import pickle
import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

#x_test = np.load('/data/dnntest/zpengac/datasets/imagenet/x_test.npy')
#x_test = x_test.astype('float64')
mean = [103.939, 116.779, 123.68]

output_path = '/data/dnntest/zpengac/DeepSearch/ImageNet/DSBatched/2021-07-03 15:23:04.123722'

with open('imagenet_indices.txt', 'r') as f:
    target_set = [int(i) for i in f.readlines()]

result_l2 = 0
result_linf = 0
for i in range(1000):
    new_img = pickle.load(open(os.path.join(output_path, f'image_{i}.pkl'), 'rb'))
    new_img *= 255.0

    id = target_set[i]
    path="/data/dnntest/zpengac/datasets/imagenet/val/ILSVRC2012_val_000"+str(id+1).zfill(5)+".JPEG"
    img = image.load_img(path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    new_img = new_img.astype('uint8')
    img = img.astype('float32')

    if result_l2 == 0:
        img_file = Image.fromarray(new_img.reshape(256,256,3), 'RGB')
        img_file.save('example3.bmp')
        orig_file = Image.fromarray(img.reshape(256,256,3).astype('uint8'), 'RGB')
        orig_file.save('example_orig3.bmp')
        
    l2 = np.sqrt(np.sum(np.square(new_img - img)))
    linf = np.max(np.abs(new_img - img))
    result_l2 += l2
    result_linf += linf

print(result_l2 / len(target_set))
print(result_linf / len(target_set))

