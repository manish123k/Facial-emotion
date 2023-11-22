import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
# import os
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

#layers of cnn

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')


#dictionary

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}

emoji_dist = {0: "emojis/angry.png", 1: "emojis/disgusted.png", 2: "emojis/fearful.png", 3: "emojis/happy.png", 4: "emojis/neutral.png", 5: "emojis/sad.png", 6: "emojis/surpriced.png"}

answer=[0]
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cant open the camera")
    exit(True)

def show_vid():
    flag, frame = cap.read()
    frame = cv2.resize(frame, (600, 500))
    frame = cv2.flip(frame, 1)
    bounding_box = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    myimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("cv2image shape-->", myimage.shape)
    num_faces = bounding_box.detectMultiScale(myimage, scaleFactor=1.3, minNeighbors=4)
    print("numface shape-->", num_faces)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 5)
        roi_gray_frame = myimage[y:y + h, x:x + w]
        print("croping-->", roi_gray_frame.shape)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1),0)
        print("dimension of input image-->",cropped_img.shape)
        prediction = emotion_model.predict(cropped_img)

        minimax = int(np.argmax(prediction))
        answer[0]=minimax
        print("emotion-->",emotion_dict[answer[0]])

    if flag is None:
        print("Major error!")
    else:
        last_frame1 = frame.copy()
        cv2image = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(20, show_vid)


def show_vid2():
    frame2 = cv2.imread(emoji_dist[answer[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[answer[0]], font=('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    lmain2.after(20, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()

    heading = Label(root, text="Welcome in Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black',
                    fg='#F5B041')
    heading.pack()

    heading1 = Label(root, text="Photo ---> Emoji", font=('arial', 20, 'bold'), bg='black', fg='green')
    heading1.pack()
    heading1.place(x=620, y=300)

    lmain = tk.Label(master=root, padx=10, bd=5)
    lmain2 = tk.Label(master=root, bd=5)
    lmain3 = tk.Label(master=root,text="Emotion name", pady=10, bd=10, font=('arial', 25, 'bold'), fg="#76D7C4", bg='black')

    lmain.pack(side=LEFT)
    lmain.place(x=2, y=100)
    lmain3.pack()
    lmain3.place(x=990, y=580)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=880, y=150)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    btn = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold'))
    btn.pack(side=BOTTOM)
    btn.place(x=690, y=550, rely=0.1)
    show_vid()
    show_vid2()
    root.mainloop()
