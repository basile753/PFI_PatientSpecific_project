"""
Software that makes an automatic segmentation of the following components of the knee : Bones, cartilages and meniscus.
The model is based on the open source MPUnet algorythm trained on a dataset of children's knee MRI with PFI.
Thus, this model has been validated for this use only.
For a general auto-segmentation use or for training your own models, use the original MPUnet models :
https://github.com/perslev/MultiPlanarUNet
"""

import tkinter as tk
from tkinter import *
from tkinter import font
from tkinter import Canvas
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time

class Fenetre(Tk):
    def __init__(self, title: str, height: int, width: int):
        super().__init__()

        #Screen settings
        self.title(title)
        pos_x = self.winfo_screenwidth() // 2 - width // 2
        pos_y = self.winfo_screenheight() // 2 - height // 2
        self.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

        #Load the images
        self.logo_uot = PhotoImage(file="PFI_AutoSegmentation_Software_DATA/uOttawa_logo.png").subsample(7)
        self.MRI_ex = PhotoImage(file = "PFI_AutoSegmentation_Software_DATA/MRI_ex.png").subsample(7)
        self.seg_ex = PhotoImage(file="PFI_AutoSegmentation_Software_DATA/segmented_ex.png").subsample(6)
        self.sim_ex = PhotoImage(file="PFI_AutoSegmentation_Software_DATA/sim_ex.png").subsample(1)

        #Set the dynamic attributes
        self.path = tk.StringVar()
        self.path.set("NO DATA LOADED")
        self.progress = tk.StringVar()
        self.progress.set("0%")
        self.int_progress = 0

        #Load the front page
        self.p_intro()

    def p_intro(self):
        """
        Load the front page of the software
        """

        #Main title
        self.reset()
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        Label(text="Analysis software for the children's knee with PFI", font=title_font, fg="black").grid(row=0, column=0, columnspan=3)

        #Adding the illustrations
        canvas_logo = Canvas(height=100, width=100)
        canvas_logo.create_image(50, 50, image=self.logo_uot, anchor='center')
        canvas_logo.grid(row=0, column=4)
        canvas_MRI = Canvas(height=200, width=200)
        canvas_MRI.create_image(100, 100, image=self.MRI_ex, anchor='center')
        canvas_MRI.grid(row=1, column=0, rowspan=2)
        canvas_seg = Canvas(height=200, width=200)
        canvas_seg.create_image(100, 100, image=self.seg_ex, anchor='center')
        canvas_seg.grid(row=1, column=1, rowspan=2)
        canvas_sim = Canvas(height=170, width=170)
        canvas_sim.create_image(100, 100, image=self.sim_ex, anchor='center')
        canvas_sim.grid(row=1, column=2, rowspan=2)

        #The buttons
        Button(text="Info", relief="raised", command=self.info, height=2, width=6).grid(row=1, column=4)
        Button(text="Start", relief="raised", command=self.p_loaddata, height=2, width=6).grid(row=2, column=4)

        #Legends
        Label(text="input : MRI", fg="black").grid(row=3,column=0)
        Label(text="Auto-segmentation", fg="black").grid(row=3, column=1)
        Label(text="Simulation to predict PFI risks", fg="black").grid(row=3, column=2)

    def reset(self):
        """
        Refresh the page by destroying every widget
        """
        #Set the calculation flag to True
        self.segm_finish = True

        #Erase every widget
        for widget in self.winfo_children():
            widget.destroy()

        #Recreate the logo
        canvas_logo = Canvas(height=100, width=100)
        canvas_logo.create_image(50, 50, image=self.logo_uot, anchor='center')
        canvas_logo.grid(row=0, column=4, sticky='ne')

    def info(self):
        info = Information()
        info.mainloop()

    def p_loaddata(self):
        """
        Load the data page
        """
        self.reset()
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        Label(text="Load your MRI data", font=title_font, fg="black").grid(row=0, column=0, columnspan=3)
        # Adding the illustrations
        canvas_MRI = Canvas(height=200, width=200)
        canvas_MRI.create_image(100, 100, image=self.MRI_ex, anchor='center')
        canvas_MRI.grid(row=1, column=0, rowspan=2)

        # The buttons
        Button(text="Info", relief="raised", command=self.info, height=2, width=6).grid(row=1, column=4)
        Button(text="Auto_segment", relief="raised", command=self.p_autosegmentation, height=2, width=12).grid(row=2, column=4)
        Button(text="load data", relief="raised", command=self.search, height=2, width=10).grid(row=1, column=1, rowspan=2)
        Button(textvariable=self.path, relief="sunken", height=2, width=40).grid(row=1, column=2, rowspan=2)
        Button(text="Return", relief="raised", command=self.p_intro, height=2, width=6).grid(row=2, column=1, columnspan=2)

    def p_autosegmentation(self):
        """
        Launch the auto-segmentation and shows the -calcul in progress- page
        :return:
        """
        self.reset()
        self.state = 0
        self.segm_finish = False

        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        Label(text="Auto-Segmentation in progress...", font=title_font, fg="black").grid(row=0, column=0, columnspan=3)
        Button(text="Return", relief="raised", command=self.p_loaddata, height=2, width=6).grid(row=2, column=2)
        Button(textvariable=self.progress, relief="sunken", height=2, width=20).grid(row=1, column=3)
        Label(text="Progression : ", width=15).grid(row=1, column=2)
        threading.Thread(target=self.run_segmentation).start()


    def run_segmentation(self):
        while self.segm_finish == False:
            self.show_loading_gif()
            time.sleep(0.01)  # Simulate processing time (replace with your segmentation code)
            self.hide_loading_gif()
            self.int_progress += 1
            self.progress.set(f'{self.int_progress}%') #TO ERASE, this is just to show a beta
            if self.int_progress == 100:
                self.segm_finish = True
                Button(text="Complete!", relief="raised", height=2, width=10).grid(row=1, column=4)
                self.show_loading_gif()

    def show_loading_gif(self):
        # Load the GIF
        self.loading_gif = PhotoImage(file='PFI_AutoSegmentation_Software_DATA/multi_planar_training.gif', format=f'gif -index {self.state}').subsample(2)  # Replace with your GIF path
        self.loading_label = Label(self, image=self.loading_gif)
        self.loading_label.grid(row=1, column=0, columnspan=2, rowspan=2)

    def hide_loading_gif(self):
        self.loading_label.destroy()
        self.state = self.state + 1 % 50

    def search(self):
        file_path = filedialog.askopenfilename(
            title="Select a File",
            filetypes=[("DICOM files", "*.dcm"), ("NIFTI files", "*.nii")],
        )
        self.path.set(file_path)

class Information(Tk):
    def __init__(self):
        super().__init__()
        self.title("Information")
        Label(self, text=("This program aims to do automatic segmentation from an MRI to propose a person-specific\n "
                          "model in order to simulate the risk of PFI for the individual... BLABLABLA\nThe "
                          "machine-learning model used is a replica of the MPUnet model trained on a dataset of 70 "
                          "children's knee MRIs, more info about the MPUnet model here : INSERT LINK\n"
                          "Thus, this programs is validated to be used on youth's MRIs images only, with no results "
                          "guaranteed out of these boudaries.\n\nThe process has 2 steps : \n\t°The segmentation phase, "
                          "you will need to insert the MRI's data as an imput\n\t°The simulation phase, you will choose "
                          "between different simulation scenario in order to see the risk of PFI\n\n This software is "
                          "open source and made by XXX from the uOttawa Health Faculty Research Program"), justify="left").grid()

def main():
    soft = Fenetre("uOttawa Health Faculty", 350, 700)
    soft.mainloop()

if __name__ == '__main__':
    main()
