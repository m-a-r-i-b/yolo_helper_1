import os
import cv2
import json
import tqdm
coco_formate = {}
coco_formate['images'] = []
coco_formate['annotations'] = []
coco_formate['categories'] =[
      {
        "supercategory": "none",
        "id": 1,
        "name": "Car"
      },
      {
        "supercategory": "none",
        "id": 2,
        "name": "Truck"
      },
      {
        "supercategory": "none",
        "id": 3,
        "name": "StopSign"
      },
      {
        "supercategory": "none",
        "id": 4,
        "name": "traffic_lights"
      }
    ]

def add_annotations(id, img, height, width):
    with open('labels/'+img ,'r') as fp:
        annotations = fp.readlines()
        for anno in annotations:
            cl, x, y, w, h, conf = anno.strip().split(' ')
            w_coco = float(w)*width
            h_coco = float(h)*height
            x_coco = float(x)*width - (w_coco/2)
            y_coco = float(y)*height - (h_coco/2)
            cl = int(cl) + 1 #coco classes start from 1 yolo start from 0 
            coco_formate['annotations'].append({
                    "image_id":id ,
                    "bbox":[int(x_coco), int(y_coco), int(w_coco), int(h_coco)] ,
                    "category_id": cl,
                    "id": len(coco_formate['annotations']),
                    "confidence": round(float(conf),3)
            })



images = os.listdir('test2_images')


for id, image in tqdm.tqdm(enumerate(images)):

    h, w, _ = cv2.imread('test2_images/'+image).shape
    coco_formate['images'].append({
        "file_name": image,
        "id": id,
        "height": h, 
        "width": w
    })
    img = image.split('.')[0] + '.txt'
    try:
        add_annotations(id, img, h, w)
    except:
        pass

with open("submission.json", "w") as outfile:
    json.dump(coco_formate, outfile)
