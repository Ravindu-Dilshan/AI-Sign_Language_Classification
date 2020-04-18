from keras.models import load_model
import cv2
import numpy as np

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

def identifyGesture(image):   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    key = mapper_rev(np.argmax(pred))
    acc = str(int(max(pred)*100))+"%"
    return key,acc

cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = (1000,1000)
FPS = 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT);
cap.set(cv2.CAP_PROP_FPS,FPS);
detect = False
prev_move = None
while True:
    ret, frame = cap.read()
    #if not ret:
        #continue
    cv2.rectangle(frame, (150, 50), (400, 300), (255, 255, 255), 2)
    cv2.rectangle(frame, (550, 50), (800, 300), (255, 255, 255), 2)

    roi = frame[50:300, 150:400]
    if(detect):
        user_move_name, acc = identifyGesture(roi)

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Letter: " + user_move_name,
                (150, 350), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Accuracy: " + acc,
                (550, 350), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place Hand on Left Box",
                (0, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    icon = cv2.imread("letters/{}.png".format(user_move_name))
    icon = cv2.resize(icon, (250, 250))
    frame[50:300, 550:800] = icon

    cv2.imshow("ASL Detection", frame)
    cv2.imshow('Hand', roi)
    
    k = cv2.waitKey(10)
    if k == 27:
        break
    if k == 32:
        if(detect):
            detect = False
        else:
            detect = True

cap.release()
cv2.destroyAllWindows()
