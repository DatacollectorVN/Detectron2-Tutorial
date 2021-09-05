# Detectron2-Tutorial
1. Create virtual environment
```bash
conda create -n detectron2 python=3.7
conda activate detectron2
```

2. Clone this repository

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Setup for detectron2
`bash python -m pip install -e detectron2`

5. Inference demo with Pre-trained models
`bash cd detectron2/demo/`

InstanceSegmentation - MASK-RCNN 
```bash 
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../../imgs-test/BlackPink.jpg --output ../../imgs-test/predicted_BalckPink.jpg --opt MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl 

```
Keypoints - RCNN-ResNet50-FPN
```bash
python demo.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --input ../../imgs-test/Jisso.jpg --output ../../imgs-test/predicted_Jisso.jpg --opt MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl
```