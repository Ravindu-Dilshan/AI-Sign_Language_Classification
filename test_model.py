# -*- coding: utf-8 -*-
from PIL import Image
import numpy
from keras.models import load_model
model = load_model('hand_sign.h5')

CLASS_MAP = {"A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"K": 9,
             "L": 10,"M": 11,"N": 12,"O": 13,"P": 14,"Q": 15,"R": 16,"S": 17,"T": 18,"U": 19,
             "V": 20,"W": 21,"X": 22,"Y": 23,"nothing": 24}
NUM_CLASSES = len(CLASS_MAP)

CLASS_MAP_REV = dict(map(reversed, CLASS_MAP.items()))
def mapper(val):
    return CLASS_MAP[val]

def mapper_rev(val):
    return CLASS_MAP_REV[val]


file_path = "asl_alphabet_test/A_test.jpg"
image = Image.open(file_path)
image = image.resize((48,48))
image = numpy.expand_dims(image, axis=0)
image = numpy.array(image)
print(image.shape)
pred = model.predict([image])[0]
sign = mapper_rev(numpy.argmax(pred))
acc = str(int(max(pred)*100))+"%"
print("Detcted Letter = "+sign+"\nAccuracy = "+acc)
