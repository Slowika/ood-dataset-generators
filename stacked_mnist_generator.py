from dataclasses import dataclass
import cv2
import os
import numpy as np
import random
import pickle
import argparse

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate Stacked MNIST')
parser.add_argument('--train_size', type=int, default=16000)
parser.add_argument('--test_size', type=int, default=4000)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--digit_size', type=int, default=16)
parser.add_argument('--is-ood', action='store_true', default=False)
parser.add_argument('--nb_digits_train', type=int, default=3)
parser.add_argument('--max_digits', type=int, default=5)
parser.add_argument('--pickle_name', type=str, default='stacked-mnist')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

train_size = args.train_size
test_size = args.test_size
img_size = args.image_size
dig_size = args.digit_size
nb_train_dig = args.nb_digits_train

random.seed(args.seed)

dirs = './stacked_mnist'

try:
  os.makedirs(dirs)
except:
  print('directory {} already exists'.format(dirs))

train_set = torchvision.datasets.MNIST('files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Resize((dig_size, dig_size),
                                                                       interpolation=InterpolationMode.NEAREST)]))
test_set = torchvision.datasets.MNIST('files/', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((dig_size, dig_size),
                                                                      interpolation=InterpolationMode.NEAREST)]))

def draw_on(data, a):
  pos_x = random.randint(0, img_size - dig_size - 1)
  pos_y = random.randint(0, img_size - dig_size - 1)
  for y in range(dig_size):
    for x in range(dig_size):
      data[pos_y + y][pos_x + x] = max(a[y][x], data[pos_y + y][pos_x + x])

SIZE_TRAIN_MNIST = 60000
SIZE_TEST_MNIST = 10000

SIZE_TRAIN = train_size
SIZE_TEST = test_size

#fig = plt.figure()
train_stacked = []
test_stacked = []

for i in range(SIZE_TRAIN):
  if args.is_ood:
    num_times = random.choice([1, nb_train_dig])
  else:
    num_times = random.randint(1, args.max_digits)
  img = np.zeros(shape=(img_size, img_size))
  label = np.zeros(shape=(10))
  for _ in range(num_times):
    idx = random.randint(0, SIZE_TRAIN_MNIST - 1)
    draw_on(img, train_set[idx][0][0])
    label[train_set[idx][1]] = 1
  train_stacked.append((np.array([img, img, img]), num_times, label))

for i in range(SIZE_TEST):
  num_times = 1 + random.choice(range(args.max_digits))
  img = np.zeros(shape=(img_size, img_size))
  label = np.zeros(shape=(10))
  for _ in range(num_times):
    idx = random.randint(0, SIZE_TEST_MNIST - 1)
    draw_on(img, test_set[idx][0][0])
    label[test_set[idx][1]] = 1
  test_stacked.append((np.array([img, img, img]), num_times, label))


filename = os.path.join(dirs, args.pickle_name + '.pickle')
with open(filename, 'wb') as f:
    pickle.dump((train_stacked, test_stacked), f)
print('datasets saved at {}'.format(filename))
