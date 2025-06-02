from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder



root = tk.Tk()
root.title("Intrustion Detection and Prevention System Using Machine Learning and Deep Learning Techniques")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.configure(background="white")


image = Image.open('graph.png')

image = image.resize((900,500))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=250, y=70) #

root.mainloop()
