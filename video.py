import tkinter as tk
from tkinter import *
from tkinter import messagebox
import tkinter.filedialog
import cv2
import time
import numpy as np
from PIL import Image
from yolo import YOLO
import GUI


def realTime():
    test(0)


def video():
    test(GUI.fileLoc())


def test(choice):
    yolo = YOLO()

    # ---------------------------------------------------#
    #   cv2.VideoCapture(0) : Capture Video from Camera
    #   cv2.VideoCapture("videoLocation") : video
    # ---------------------------------------------------#
    # climbOut / removeTube
    capture = cv2.VideoCapture(choice)
    fps = 0.0
    coAlarm, rtAlarm = GUI.onOff()
    GUI.alarmMsg(coAlarm, "climb out")
    GUI.alarmMsg(rtAlarm, "remove tubes")

    while (True):
        t1 = time.time()
        #  Capture frame-by-frame
        ref, frame = capture.read()

        #  BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #  Transform  into image
        frame = Image.fromarray(np.uint8(frame))
        #  Detection
        frame, climbOut = yolo.detect_image(frame, coAlarm, rtAlarm)
        frame = np.array(frame)
        #  RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(frame.shape)

        fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %.2f"%(fps))
        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if climbOut == 1:
            # For testing : (h,w) = (208,1000)
            frame = cv2.putText(frame, "Climb Out!", (208, 1000), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 2,
                                cv2.LINE_AA)

        # For adjusting the window
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.imshow("video", frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break


window = Tk()
window.title("Patient Risk Detection")
window.geometry("500x500")

B1 = tkinter.Button(window, width=25, height=5, text="Capture Video from Camera", fg="blue", command=realTime).place(x=150, y=100)
B2 = tkinter.Button(window, width=25, height=5, text="Loading Video File(s)", fg="green", command=video).place(x=150,
                                                                                                              y=300)

window.mainloop()
