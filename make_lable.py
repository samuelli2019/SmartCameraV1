#!/usr/bin/env python3

import os
import numpy as np


cat_dict = dict()
dog_dict = dict()
human_dict = dict()
img_name = None
img_tag = .0
file_line = None

with open('./VOC2012/ImageSets/Main/cat_trainval.txt') as f:
    while True:
        file_line = f.readline()
        if len(file_line) < 3:
            break
        img_name = file_line[:file_line.index(' ')]
        img_tag = float(file_line[file_line.rindex(' ')+1:])
        cat_dict[img_name] = img_tag if img_tag == 1.0 else .0

with open('./VOC2012/ImageSets/Main/dog_trainval.txt') as f:
    while True:
        file_line = f.readline()
        if len(file_line) < 3:
            break
        img_name = file_line[:file_line.index(' ')]
        img_tag = float(file_line[file_line.rindex(' ')+1:])
        dog_dict[img_name] = img_tag if img_tag == 1.0 else .0

with open('./VOC2012/ImageSets/Main/person_trainval.txt') as f:
    while True:
        file_line = f.readline()
        if len(file_line) < 3:
            break
        img_name = file_line[:file_line.index(' ')]
        img_tag = float(file_line[file_line.rindex(' ')+1:])
        human_dict[img_name] = img_tag if img_tag == 1.0 else .0


result_dict = dict()
for k, v in cat_dict.items():
    result_dict[k] = (v, dog_dict[k], human_dict[k])

import pickle
with open('tag.pkl', 'wb') as f:
    pickle.dump(result_dict, f)