import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

def run():
    subprocess.Popen(["python3", "drowsiness_detect.py"])

# Create the main window
window = tk.Tk()
window.title('Driver Drowsiness Detector')
window.geometry('900x700')

# Load the background image
bg_image = Image.open("image.jpg")
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas widget and display the background image
canvas = tk.Canvas(window, width=900, height=700)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title
title_label = ttk.Label(master=canvas, text="ARE YOU DROWSY?", font='Arial 30')
title_label.pack(pady = 50)

# Input field
input_frame = ttk.Frame(master=canvas)
button = ttk.Button(master=input_frame, text='Click to Check', padding = 20 , command=run)
button.pack()
# input_frame.place(relx=0.5, rely=0.5, anchor="center")
input_frame.pack()

# Run the Tkinter event loop
window.mainloop()
