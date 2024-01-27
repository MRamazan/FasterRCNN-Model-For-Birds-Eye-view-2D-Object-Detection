import os
import time

import PIL
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, \
FasterRCNN_ResNet50_FPN_Weights
from natsort import natsorted




model_path = r'C:\Users\PC\PycharmProjects\pythonProject2\weights\epoch_1.pkl'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 11)


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])


model = model.to(device="cpu")
model.eval()

def preprocess(image):
    read_img = PIL.Image.open(image).convert("RGB")
    crop = read_img.resize((416, 416))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    process = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    batch = process(crop)

    return batch

def resize(image, h, w):
    resized_img = cv2.resize(image, (int(h), int(w)))
    return resized_img


import cv2

# Giriş video dosyasının yolu
video_path = r"C:\Users\PC\Downloads\istockphoto-1447043419-640_adpp_is.mp4"
target_dir = r"C:\Users\PC\PycharmProjects\pythonProject2\frame_folder"
def extract_frames(target_dir, video):
    captured_video = cv2.VideoCapture(video)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    count = 0
    while True:
        success, image = captured_video.read()
        if not success:
            break
        else:

            direction = os.path.join(target_dir, "frame%d.jpg")
            cv2.imwrite(direction % count, image)
            if cv2.waitKey(10) == 27:  # exit if Escape is hit
                break
            count += 1









def remove_frame_folder(target_dir):
    frame_list = os.listdir(target_dir)
    for frame in frame_list:
        os.remove(os.path.join(target_dir, frame))
    os.rmdir(target_dir)




def resize(image, h, w):
    resized_img = cv2.resize(image, (int(h), int(w)))
    return resized_img



result_folder = r"C:\Users\PC\PycharmProjects\pythonProject2\results"
if not os.path.exists(result_folder):
 os.mkdir(result_folder)



def detect_one_by_one(image_list, loop_count):
    batch = []
    process_start_time = time.time()

    for images in image_list:
        processed_image = preprocess(images)
        batch.append(processed_image)


    with torch.no_grad():
          pred = model(batch)

          image_count = 0
          for count, frame in enumerate(pred):
              image_index = 0
              image = cv2.imread(image_list[count])
              resized_img = resize(image, 416, 416)
              for boxes in frame["boxes"]:
                 score = frame["scores"][image_index]
                 if score > 0.7:
                   top_left = int(boxes[0]), int(boxes[1])
                   bot_right = int(boxes[2]), int(boxes[3])
                   cv2.rectangle(resized_img, top_left, bot_right, color=(0, 0, 255))

                 else:
                  continue
                 image_index += 1
              index = str(loop_count) + str(image_count)
              cv2.imwrite(r"C:\Users\PC\PycharmProjects\pythonProject2\results\frame%a.jpg" % index, resized_img)
              image_count += 1
    process_finish_time = time.time()
    process_finish_time = round(process_finish_time, 3)
    process_start_time = round(process_start_time, 3)
    process_per_image = (process_finish_time - process_start_time) / 10
    print(round(process_per_image, 3))











def draw_boxes(frame_list):
    image_dirs = []
    count = 0
    for  frame in natsorted(frame_list):
        image_dir = os.path.join(target_dir, frame)
        image_dirs.append(image_dir)
        if len(image_dirs) == 10:
            count += 1
            detect_one_by_one(image_dirs, count)
            image_dirs = []
        else:
            continue

if __name__ == '__main__':
 extract_frames(target_dir, video_path)
 draw_boxes(os.listdir(target_dir))





















