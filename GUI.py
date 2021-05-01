import tkinter as tk
from tkinter import *
from tkinter import messagebox
import tkinter.filedialog
import winsound

root = Tk()
root.withdraw()


def onOff():
    coAlarm = tk.messagebox.askquestion('Patient Risk Detection - Climb Out', 'Turn on climb out alarming ?')

    rtAlarm = tk.messagebox.askquestion('Patient Risk Detection - Remove Tubes', 'Turn on remove tubes alarming ?')

    return coAlarm, rtAlarm


def alarmMsg(onOff, act):
    if onOff == "yes":
        print("Turn the " + act + " alarm ON")
    else:
        print("Turn the " + act + " alarm OFF")


def fileLoc():
    filename = tkinter.filedialog.askopenfilename(title="Please select a file:")
    return filename
