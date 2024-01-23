import PIL
import cv2
import os
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
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
        self.obj = 0
        self.class_id = 0
        self.targets = []
        self.count = 0



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


                        msg["annotations"].append([x1, y1, w, h, class_, score])
            return msg








    def read_image(self, image_dir):
        image = cv2.imread(image_dir)
        assert image is not None, f"File {image_dir} does not exist or broken!"
        return image

    def __getitem__(self, idx):
        self.count += 1
        labels = []
        target = {}
        boxes = []
        image = self.image_list[idx]
        image_dir = os.path.join(self.image_dir, image)
        read_img = PIL.Image.open(image_dir).convert("RGB")
        dir = self.anno_list[idx]
        txt_dir = os.path.join(self.anno_dir, dir)
        anno = self.object_infos(txt_dir)
        lenght = len(anno["annotations"])
        for item in range(0, lenght):
            self.class_id = anno["annotations"][item][4]
            x = anno["annotations"][item][0]
            y = anno["annotations"][item][1]
            w = anno["annotations"][item][2]
            h = anno["annotations"][item][3]
            score = anno["annotations"][item][5]
            top_left = int(x), int(y)
            bot_right = (int(x) + int(w), int(y) + int(h))
            cords = [top_left[0],top_left[1], bot_right[0],bot_right[1]]
            boxes.append(torch.tensor(cords))

            labels.append(torch.tensor(int(self.class_id), dtype=torch.int64))
            target["image_id"] = (torch.as_tensor(idx))


            # crop image



        # recolor, reformat
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)
        label = {
            'boxes': boxes,
            "labels" : labels

        }
        self.targets.append(label)
        crop = read_img.resize((416, 416))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        process = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        batch = process(crop)





        if self.count == 1:
            print(self.targets[0])
            return batch, self.targets[0]
        else:
            print(self.targets[self.count - 1])
            return batch, self.targets[self.count - 1]

class DetectedObject:
    def __init__(self, img, detection_class, box_2d):

        self.img = self.normalize_image(img, box_2d)
        self.detection_class = detection_class

    def normalize_image(self, img, box_2d):

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





        crop = cv2.resize(src=img, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)




        # recolor, reformat
        batch = process(crop)


        return batch
