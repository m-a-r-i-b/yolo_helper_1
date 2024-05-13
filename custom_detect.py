import os
import sys
from pathlib import Path
from turtle import width
import cv2 
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_sync
import numpy as np

from models.common import DetectMultiBackend
weights = "runs/train/exp8/weights/best.pt"
test_img = 'as.jpeg'
data = 'data/custom.yaml'


def convert_mid_to_corner(x,y,w,h):
    x1 = (x-(w/2))
    y1 = (y-(h/2))
    x2 = x1 + w
    y2 = y1 + h
    return [x1,y1,x2,y2]

def convert_to_int(width, height,line_point):
    x1,y1,x2,y2 = line_point
    x1 = int(x1*width)
    x2 = int(x2*width)
    y1 = int(y1*height)
    y2 = int(y2*height)
    return x1, y1, x2, y2

if __name__ == "__main__":
    device = select_device('0')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    img = cv2.imread(test_img)
    img0 = cv2.imread(test_img)
    print(img0.shape)
    height, width, _  = img0.shape
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
    img = img.reshape(1,988,1280,3)
    img = img.transpose((0,3,1,2))
    img = img/255.0
    img = torch.from_numpy(img).to(device).float()
    pred = model(img,augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.9, 0.45, None, True, max_det=1000)
    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 15
    for i, det in enumerate(pred):
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            x,y,w,h = line[1], line[2], line[3], line[4]
            print(x,y,w,h )
            line_point = convert_mid_to_corner(x,y,w,h)
            print(line_point)
            x1,y1,x2,y2 = convert_to_int(width, height,line_point)
            print(x1,y1,x2,y2)
            cv2.rectangle(img0,(x1, y1), (x2, y2),color,thickness)
            cv2.imshow('test',img0)
            cv2.waitKey(0)