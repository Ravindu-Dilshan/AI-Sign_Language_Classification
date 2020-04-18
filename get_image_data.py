# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 07:42:27 2020

@author: user
"""
import numpy as np
import os
from PIL import Image

IMG_SAVE_PATH = 'asl_alphabet_train'

CLASS_MAP = {"A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"K": 9,
             "L": 10,"M": 11,"N": 12,"O": 13,"P": 14,"Q": 15,"R": 16,"S": 17,"T": 18,"U": 19,
             "V": 20,"W": 21,"X": 22,"Y": 23,"nothing": 24}
NUM_CLASSES = len(CLASS_MAP)

CLASS_MAP_REV = dict(map(reversed, CLASS_MAP.items()))
def mapper(val):
    return CLASS_MAP[val]

def mapper_rev(val):
    return CLASS_MAP_REV[val]

data = []
labels = []
cur_path = os.getcwd()
for i in range(NUM_CLASSES):
    path = os.path.join(cur_path,IMG_SAVE_PATH,mapper_rev(i))
    images = os.listdir(path)
    print(str(i)+" start")
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((48,48))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
    print(str(i)+" end")    
    
data = np.array(data)
labels = np.array(labels)

from numpy import save
save('data.npy', data)
save('lables.npy', labels)