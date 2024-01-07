import os
import time

import cv2

from torchvision.io import read_image, read_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import numpy as np
from natsort import natsorted



# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

import cv2

# Giriş video dosyasının yolu
video_path = cv2.VideoCapture(r"C:\Users\PC\Downloads\istockphoto-1072509294-640_adpp_is.mp4")
target_dir = r"C:\Users\PC\PycharmProjects\pythonProject2\frame_folder"
def extract_frames(target_dir, video):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    count = 0
    while True:
        success, image = video.read()
        if not success:
            break
        else:

            direction = os.path.join(target_dir, "frame%d.jpg")
            cv2.imwrite(direction % count, image)  # save frame as JPEG file
            if cv2.waitKey(10) == 27:  # exit if Escape is hit
                break
            count += 1



def remove_frame_folder(target_dir):
    frame_list = os.listdir(target_dir)
    for frame in frame_list:
        os.remove(os.path.join(target_dir, frame))
    os.rmdir(target_dir)


results = []
# Step 3: Apply inference preprocessing transforms
def detection(img):
 batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
 prediction = model(batch)[0]
 labels = [weights.meta["categories"][i] for i in prediction["labels"]]
 box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
 im = to_pil_image(box.detach())


 return im
result_folder = r"C:\Users\PC\PycharmProjects\pythonProject2\results"
os.mkdir(result_folder)
direction2 = os.path.join(result_folder, "frame%d.jpg")

extract_frames(target_dir, video_path)
def detect_one_by_one():
    frame_list = os.listdir(target_dir)
    sorted_frame_list = natsorted(frame_list)
    for frame_num, frame in enumerate(sorted_frame_list):
        baslangic = time.time()
        print(frame)
        frame_dir = os.path.join(target_dir, frame)
        img = read_image(frame_dir)
        result = detection(img)
        print(result)
        cv2.imwrite(direction2 % frame_num, np.array(result))
        bitis = time.time()
        print(bitis - baslangic)
        results.append(result)




detect_one_by_one()





remove_frame_folder(target_dir)





# Giriş video dosyasının yolu



















