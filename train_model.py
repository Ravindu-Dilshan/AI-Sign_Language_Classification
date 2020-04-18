import numpy as np
from keras.optimizers import Adam
from keras_squeezenet import SqueezeNet
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

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

def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(48, 48, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

#load data
from numpy import load
data = load('data.npy')
print(data)
labels = load('lables.npy')
print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
lbl = y_test
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

del data
del labels

# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# save the model for later use
model.save("hand_sign.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('acc_curve.png')
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

from sklearn.metrics import accuracy_score
pred = model.predict_classes(X_test)

#Accuracy with the test data
print(accuracy_score(lbl, pred))

from sklearn.metrics import confusion_matrix

#----------------start confusion matriix----------------------
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=30)
    plt.xlabel('Predicted label',fontsize=30)
    plt.tight_layout()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
class_names = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P',
               'Q','R','S','T','U','V','W','X','Y','NONE']


# Compute confusion matrix
cnf_matrix = confusion_matrix(lbl, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(15,10))
plot_confusion_matrix(cnf_matrix, classes=class_names)
plt.title('Confusion Matrix',fontsize=30)
plt.savefig('confusion_matrix.png')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(lbl, pred))
