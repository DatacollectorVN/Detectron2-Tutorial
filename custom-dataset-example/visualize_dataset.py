import numpy as np 
import pandas as pd 
import os
import random
import cv2
import sys
from detectron2.data import MetadataCatalog, DatasetCatalog # read in document to know how it work
# https://detectron2.readthedocs.io/en/latest/modules/utils.html?highlight=Visualizer#detectron2.utils.visualizer.Visualizer
from detectron2.utils.visualizer import Visualizer
from utils import get_chestxray_dicts

ANNOTATIONS_CSV = "annotation_small.csv"
DETECTRON2 = "detectron2-tutorial-example"
IMG_DIR = os.path.join(DETECTRON2, "imgs")

def main():
    df = pd.read_csv(os.path.join(DETECTRON2, ANNOTATIONS_CSV))
    class_name  = df.class_name.unique().tolist()
    DatasetCatalog.register("my_dataset", lambda : get_chestxray_dicts(df, class_name, IMG_DIR))
    MetadataCatalog.get("my_dataset").set(thing_classes = class_name)
    chestxray_metatdata = MetadataCatalog.get("my_dataset")
    chestxray_metatdata.thing_colors = [(255, 0, 0), (0, 0, 255)]
    dataset = get_chestxray_dicts(df, class_name, IMG_DIR)
    
    # get random sample
    index = random.randint(0, 99)
    sample = dataset[index]
    img = cv2.imread(sample["file_name"])
    visualizer = Visualizer(img_rgb = img[:, :, ::-1], # img is loaded by opencv (BGR) --> ::-1 inverse channel tp (RGB)
                            metadata = chestxray_metatdata, 
                            scale = 1)
    
    # https://detectron2.readthedocs.io/en/latest/modules/utils.html?highlight=Visualizer#detectron2.utils.visualizer.Visualizer.draw_dataset_dict
    out = visualizer.draw_dataset_dict(dic = sample) # return VisImage object
    
    # out.get_image()[:, :, ::-1] --> ndarray
    cv2.imshow("window", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
