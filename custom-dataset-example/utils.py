import numpy as np 
import pandas as pd 
import os
from PIL import Image
from detectron2.structures import BoxMode
import cv2

def get_chestxray_dicts(df, class_name, img_dir):
    COCO_detectron2_list = [] # list(dict())
    img_paths = []
    img_ids = df["image_id"].unique().tolist()

    for i, img_id in enumerate(img_ids):
        img_file = img_id + ".jpg"
        img_path = os.path.join(img_dir, img_file)
        img_paths.append(img_paths)
        img = Image.open(img_path)
        width, height = img.size
        id_ = i + 1
        img_classes_name = df[df["image_id"] == img_id]["class_name"].values.tolist()
        img_bboxes = df[df["image_id"] == img_id][["x_min", "y_min", "x_max", "y_max"]].values
        x_min = img_bboxes[:, 0]
        y_min = img_bboxes[:, 1]
        x_max = img_bboxes[:, 2]
        y_max = img_bboxes[:, 3]

        annotaions = [] # list(dict())
        for j, img_class_name in enumerate(img_classes_name):
            annotaions_dct = {"bbox" : [x_min[j], y_min[j], x_max[j], y_max[j]],
                                "bbox_mode" : BoxMode.XYXY_ABS,
                                "category_id" : class_name.index(img_class_name)
                             }
            annotaions.append(annotaions_dct)
        
        COCO_detectron2_dct = {"image_id" : id_,
                                "file_name" : img_path,
                                "height" : height,
                                "width" : width,
                                "annotations" : annotaions
                              }

        COCO_detectron2_list.append(COCO_detectron2_dct)
    
    return COCO_detectron2_list

def draw_bbox(img, classes_name, classes_id, bboxes, color):
    '''
    Args:
        img: (ndarray) img array (H, W, C)
        classes_name: (list) contains classes name of dataset.
        classes_id: (ndarray) contains all classes id of img with correspoding to index of class name in classes_name.
        color: (list) contains color of classes with correspoding to index of class name in classes_name.
    
    Return:
        final_img: (ndarrau) img array with bbox.
    '''

    final_img = img.copy()
    for i, class_id in enumerate(classes_id):
        box = bboxes[i]
        
        cv2.putText(final_img,
                    text = classes_name[class_id].upper(),
                    org = (int(box[0]), int(box[1] - 5)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = color[class_id],
                    thickness = 1, lineType = cv2.LINE_AA)
        cv2.rectangle(final_img, pt1 = (int(box[0]), int(box[1])), pt2 = (int(box[2]), int(box[3])), 
                      color = color[class_id], thickness = 3)
    
    return final_img