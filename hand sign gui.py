import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
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
                 
#initialise GUI
top=tk.Tk()
top.geometry('300x420')
top.title('ASL sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((48,48))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict([image])[0]
    key = mapper_rev(numpy.argmax(pred))
    acc = str(int(max(pred)*100))+"%"
    sign = key + " "+acc
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.3,rely=0.7)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="ASL sign Language",pady=20, font=('arial',15,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
