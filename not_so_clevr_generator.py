import cv2
import os
import numpy as np
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='Generate Sort-of-ClEVR (https://arxiv.org/abs/1706.01427)')
parser.add_argument('--train_size', type=int, default=9800)
parser.add_argument('--test_size', type=int, default=2000)
parser.add_argument('--image_size', type=int, default=75)
parser.add_argument('--object_size', type=int, default=5)
parser.add_argument('--nb_questions', type=int, default=10)
parser.add_argument('--nb_heldout_colors', type=int, default=0)
parser.add_argument('--pickle_name', type=str, default='not-so-clevr')

args = parser.parse_args()

train_size = args.train_size
test_size = args.test_size
img_size = args.image_size
size = args.object_size

random.seed(123)

dirs = './notsoclevr'

all_colors = [
    (255,0,0),##r
    (0,255,0),##g
    (0,0,255),##b
    (255,165,0),##o
    (0, 0, 0),##k
    (255,255,0)##y
]

try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_dataset(is_train):
    objects = []
    img = np.ones((img_size,img_size,3)) * 255


    center1 = center_generate(objects)
    objects.append((0, center1, 0))
    center2 = center_generate(objects)
    r = random.random()
        
    if is_train:
        color1 = random.choice(range(6))
        color2 = color1
        if r < 0.5:
            is_same = 1
            if color1 == 0:
                shape1 = shape2 = 0
            else:
                shape1 = random.randint(0, 1)
                shape2 = shape1
        else:
            is_same = 0
            while color1 == color2:
                color2 = random.choice(range(6))
            shape1 = 0 if color1 == 0 else random.randint(0, 1)
            shape2 = 0 if color2 == 0 else random.randint(0, 1)
    else:
        color1 = color2 = 0
        shape1 = shape2 = 1
        if r < 0.5:
            is_same = 1
        else:
            is_same = 0
            while color1 == color2 and shape1 == shape2:
                color2 = random.choice(range(6))
                shape2 = random.randint(0, 1)
    
    if shape1 == 1:
        start = (center1[0]-size, center1[1]-size)
        end = (center1[0]+size, center1[1]+size)
        cv2.rectangle(img, start, end, all_colors[color1], -1)
    else:
        center_ = (center1[0], center1[1])
        cv2.circle(img, center_, size, all_colors[color1], -1)
    if shape2 == 1:
        start = (center2[0]-size, center2[1]-size)
        end = (center2[0]+size, center2[1]+size)
        cv2.rectangle(img, start, end, all_colors[color2], -1)
    else:
        center_ = (center2[0], center2[1])
        cv2.circle(img, center_, size, all_colors[color2], -1)

    question = random.randint(0, 1)
    dataset = ((img, question), is_same ^ question)
    return dataset


print('building test datasets...')
test_datasets = [build_dataset(False) for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset(True) for _ in range(train_size)]
print('saving datasets...')
filename = os.path.join(dirs, args.pickle_name + '.pickle')
with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))
