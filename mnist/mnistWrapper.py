from torchvision import transforms, datasets
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np

import imp
MainModel = imp.load_source('MainModel', "/data/dnntest/zpengac/models/lenet5.py")
my_model = torch.load("/data/dnntest/zpengac/models/lenet5.pt")
my_model = my_model.eval()

data_test = datasets.MNIST(root="/data/dnntest/zpengac/datasets/mnist",
                           transform = transforms.ToTensor(),
                           train = False,
                           download = True)

x_test, y_test = zip(*data_test)
x_test = np.array(torch.stack(list(x_test)))
y_test = np.array(list(y_test))
