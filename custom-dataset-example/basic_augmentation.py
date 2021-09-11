import pandas as pd 
import os
import cv2
import sys
from detectron2.data import MetadataCatalog, DatasetCatalog # read in document to know how it work
# https://detectron2.readthedocs.io/en/latest/modules/utils.html?highlight=Visualizer#detectron2.utils.visualizer.Visualizer
from detectron2.utils.visualizer import Visualizer
from utils import get_chestxray_dicts, draw_bbox
import detectron2.data.transforms as T
from detectron2.config import get_cfg
# use default mapper 
from detectron2.data import DatasetMapper, build_detection_train_loader

ANNOTATIONS_CSV = "annotation_small.csv"
DETECTRON2 = "detectron2-tutorial-example"
IMG_DIR = os.path.join(DETECTRON2, "imgs")
COLOR = [[255, 0, 0], [0, 0, 255]]

def main():
    df = pd.read_csv(os.path.join(DETECTRON2, ANNOTATIONS_CSV))
    classes_name = df.class_name.unique().tolist()
    DatasetCatalog.register("my_dataset", lambda : get_chestxray_dicts(df, classes_name, IMG_DIR))
    MetadataCatalog.get("my_dataset").set(thing_classes = classes_name)
    
    # get configuration
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("my_dataset")

    #basic_augmentation.ipynb - see how to get dataset inside dataloader
    dataloader = build_detection_train_loader(cfg, mapper = DatasetMapper(cfg, is_train = True, 
                                                                          augmentations = [T.RandomRotation(angle = 10), 
                                                                                           T.RandomFlip(prob = 1, horizontal = True, vertical = False), 
                                                                                           T.Resize((600, 600))]),
                                              total_batch_size = 5)
    samples = iter(dataloader).__next__()
    sample = samples[0]
    img = samples[0]['image']

    # permute tenosor with shape (H, W, C)
    img = img.permute(1, 2, 0)
    
    img_file = sample["file_name"]
    bboxes = sample["instances"].get("gt_boxes").tensor
    classes_id = sample["instances"].get("gt_classes")

    # convert to ndarray
    img, bboxes, classes_id = img.detach().cpu().numpy(), bboxes.detach().cpu().numpy(), classes_id.detach().cpu().numpy()    

    img_draw = draw_bbox(img, classes_name, classes_id, bboxes, COLOR)
    cv2.imshow("window", img_draw)
    cv2.waitKey()

if __name__ == "__main__":
    main()
