import cv2
import numpy as np
import cv2 as cv
import os
from time import time
import torch
from torchvision import transforms
from torch.utils import data
import pandas as pd

class VisdroneDataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.anno_dir = os.path.join(path, "annotations")
        self.image_dir = os.path.join(path, "images")
        self.anno_list = os.listdir(self.anno_dir)
        self.image_list = os.listdir(self.image_dir)
        self.msg = {
            "annotations": []
        }



    def __len__(self):
        return len(self.anno_list)

    def object_infos(self, txt_dir):
            msg = {
            "annotations": []
            }
            with open(txt_dir, "r", encoding="utf8") as f:
                for count,line in enumerate(f.read().split("\n")):
                    line = line.replace(",", " ").split()
                    if len(line) == 8:
                        x1, y1, w, h = [float(k) for k in line[:4]]
                        class_ = line[5]

                        score = int(line[4])
                        if score == 0:
                            continue

                        if int(class_) == 0:
                            # class_id 0 = ignored_regions
                            continue


                        msg["annotations"].append([x1, y1, w, h, class_])
            return msg








    def read_image(self, image_dir):
        image = cv2.imread(image_dir)
        assert image is not None, f"File {image_dir} does not exist or broken!"
        return image

    def __getitem__(self, idx):
        image = self.image_list[idx]
        image_dir = os.path.join(self.image_dir, image)
        read_img = cv2.imread(image_dir)
        dir = self.anno_list[idx]
        txt_dir = os.path.join(self.anno_dir, dir)
        anno = self.object_infos(txt_dir)
        lenght = len(anno["annotations"])
        for item in range(0, lenght):
            class_id = anno["annotations"][item][4]
            x = anno["annotations"][item][0]
            y = anno["annotations"][item][1]
            w = anno["annotations"][item][2]
            h = anno["annotations"][item][3]
            top_left = (int(x), int(y))
            bot_right = (int(x) + int(w), int(y) + int(h))
            obj = DetectedObject(read_img, class_id, (top_left, bot_right))
            image = cv2.rectangle(read_img, (top_left), (bot_right), color=(0, 0, 255))
            if idx % 50 == 0 and idx != 0:
             cv2.imshow("asd", image)
             cv2.waitKey(0)
            else:
                    cv2.destroyAllWindows()

        return obj, class_id

class DetectedObject:
    def __init__(self, img, detection_class, box_2d):

        self.img = self.format_img(img, box_2d)
        self.detection_class = detection_class

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])

        # crop image

        pt1 = box_2d[0]
        pt2 = box_2d[1]

        image = cv2.rectangle(img, (pt1), (pt2), color=(0, 0, 255))
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]


        if crop.shape[0] > 0 and crop.shape[1] > 0 and crop.shape[2] > 0:
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            print("Warning: Crop height is zero. Skipping resize.")



        # recolor, reformat
        batch = process(crop)


        return batch
