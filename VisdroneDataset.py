import PIL
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


    def __getitem__(self, idx):
        self.count += 1
        labels = []
        target = {}
        boxes = []
        image = self.image_list[idx]
        image_dir = os.path.join(self.image_dir, image)
        read_img = PIL.Image.open(image_dir).convert("RGB")
        crop = read_img.resize((416, 416))
        org_x, org_y = read_img.size
        scale_x = 416 / org_x
        scale_y = 416 / org_y
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
            x = scale_x * x + 1
            y = scale_y * y + 1
            w = scale_x * w + 1
            h = scale_y * h + 1
            top_left = int(x), int(y)
            bot_right = (int(x) + int(w), int(y) + int(h))

            cords = [top_left[0],top_left[1], bot_right[0],bot_right[1]]
            boxes.append(torch.tensor(cords))

            labels.append(torch.tensor(int(self.class_id), dtype=torch.int64))
            target["image_id"] = (torch.as_tensor(idx))




            #IF YOU WANT TO CHECK IF BOUNDING BOX POSITIONS ARE TRUE:
            '''rect = cv2.rectangle(np.array(crop),top_left, bot_right,(0, 0, 255))
            cv2.imshow("window", rect)
            cv2.waitKey(0)'''


            



       
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)
        label = {
            'boxes': boxes,
            "labels" : labels

        }
        self.targets.append(label)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        process = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


        batch = process(crop)


        if self.count == 1:
            return batch, self.targets[0]
        else:

            return batch, self.targets[self.count - 1]


