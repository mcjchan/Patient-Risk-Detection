from PIL import Image
from yolo import YOLO
import tkinter as tk
from tkinter import filedialog
import GUI


# ----------------------------------------------------------#
#   Pop-up message window
# ----------------------------------------------------------#
def popmsg(msg):
    NORM_FONT = ("Helvetica", 10)
    popup = tk.Tk()
    popup.wm_title("!")
    label = tk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

yolo = YOLO()
coAlarm, rtAlarm = GUI.onOff()
GUI.alarmMsg(coAlarm, "climb out")
GUI.alarmMsg(rtAlarm, "remove tubes")

while True:
    # img = input('Input image filename:')
    img = GUI.fileLoc()
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image, climbOut = yolo.detect_image(image, coAlarm, rtAlarm)
        # Save image
        # r_image.save("imgName.jpg")
        r_image.show()

        if climbOut == 1:
            popmsg("!Climb Out!")