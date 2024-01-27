import os
import threading
import time
import cv2


import tkinter as tk
from tkinter import simpledialog, Listbox, ttk, PhotoImage
from PIL import Image, ImageTk
from tkinter import *
from natsort import natsorted
import Run


class arayuz_tasarim():
    def __init__(self):
        self.input = "None"
        self.window = tk.Tk()
        self.text_box = tk.Text(self.window, height=1, width=60)
        self.text_box2 = tk.Text(self.window, height=1, width=20)
        self.text = "None"
        self.previous_inputs = []
        self.fps = 20
        self.target_folder = r"C:\Users\PC\PycharmProjects\pythonProject2\frame_folder"
        self.result_folder = r"C:\Users\PC\PycharmProjects\pythonProject2\results"


    def get_input(self):

        self.input = self.text_box.get(1.0, "end-1c")
        self.fps_input = self.text_box2.get(1.0, "end-1c")
        self.text = self.input
        if len(self.fps_input)  > 1:
         self.fps = int(self.fps_input)
        self.previous_inputs.append(self.input)
        self.text_box.delete(1.0, "end")
        Run.extract_frames(self.target_folder, self.input)
        self.frame_list = os.listdir(self.target_folder)
        total_image = len(self.frame_list)
        self.islem_ekranı(total_image,1.47)









    def ana_ekran(self):
        self.clear_window(self.window)
        self.window.config(background="lightblue")
        self.window.geometry("900x900")

        listbox = Listbox(self.window, width=75)

        label = tk.Label(self.window, text="Previous Inputs", font=("Arial", 12))
        remove_folders_label = tk.Button(self.window, text="Remove Folders", font=("Arial", 12), command=lambda: [Run.remove_frame_folder(self.target_folder), Run.remove_frame_folder(self.result_folder)])
        fps_txt = tk.Label(self.window, text="FPS: ", font=("Arial", 15), background="lightblue")
        video_dir = tk.Label(self.window, text="Video direction:", font=("Arial", 15), background="lightblue")
        enter_button = tk.Button(self.window, text="ENTER", command=self.get_input, width=20, height=2, background="gray")
        self.text_box = tk.Text(self.window, height=1, width=60)
        self.text_box2 = tk.Text(self.window, height=1, width=20)


        label.place(x=100, y=175)
        remove_folders_label.place(x=10, y=595)
        listbox.place(x=100, y=200)
        fps_txt.place(x=10, y=495)
        video_dir.place(x=10, y=395)

        enter_button.place(x=650, y=390)

        self.text_box.place(x=150, y=400)
        self.text_box2.place(x=150, y=500)


        if len(self.previous_inputs)   >= 3:
         for index in range(len(self.previous_inputs) - 3, len(self.previous_inputs)):
            listbox.insert(index, self.previous_inputs[index])
        else:
            for index ,input in enumerate(self.previous_inputs):
                listbox.insert(index, input)

        self.window.mainloop()

    def clear_window(self, window):
         for widget in window.winfo_children():
            widget.destroy()

    def back_to_main(self, progress_bar):
        progress_bar.stop()
        self.clear_window(self.window)
        self.ana_ekran()



    def islem_ekranı(self, total_image_count, processing_time_per_img):
        start_time = time.strftime("%H:%M")


        total_sure = round(((total_image_count * processing_time_per_img) / 60), 1)
        self.clear_window(self.window)
        self.window.config(background="black")


        text =  "Video will be read in about  " + str(total_sure) + "  Minutes"

        label1 = tk.Label(self.window, text=text, background="black", width=40, height=5, font=("Arial", 15), fg="blue")
        buton = tk.Button(self.window, text="START PROCESS", width=30, height=10,font=("Arial", 15), command=self.start_process, background="blue")



        label1.place(x=220, y=395)
        buton.place(x=275, y=150)



        #Pencereyi dondurduğu için eklemedim
        '''style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TProgressbar",
                        background="lightblue",
                        troughcolor="lightgray",
                        bordercolor="darkblue",
                        lightcolor="lightblue",
                        darkcolor="darkblue")
        progress_bar = ttk.Progressbar(self.window, length=450, mode='determinate', orient="horizontal")'''
     
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        else:
         if len(os.listdir(self.result_folder)) > 10:
            self.play_video()



    def start_process(self):
        self.window.destroy()
        Run.draw_boxes(self.frame_list)
        self.play_video()






    def return_adress(self):
        return self.text

    def start_video(self):
        self.window = tk.Tk()
        self.window.config(background="lightblue")
        self.window.geometry("900x900")
        buton = tk.Button(self.window, text="START VIDEO",width=30,height=10,font=("Arial", 15),command=self.play_video)
        buton.place(x=275, y=150)

    def play_video(self):

        result_folder = r"C:\Users\PC\PycharmProjects\pythonProject2\results"
        for frames in natsorted(os.listdir(result_folder)):
            frame = cv2.imread(os.path.join(result_folder, frames))
            frame = cv2.resize(frame, (800, 600))

            cv2.imshow("asd", frame)

            cv2.waitKey(self.fps)

        cv2.destroyAllWindows()
        self.window.deiconify()
        self.ana_ekran()





if __name__ == '__main__':
 x = arayuz_tasarim()
 x.ana_ekran()






