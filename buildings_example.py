
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime

def get_building_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        for annotation in annos:

            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            # print(annos['region_attributes'])
            region_attributes = annotation["region_attributes"]["class"]

            if all(k in region_attributes for k in ("building", "window")):
                category_id = 2
            elif "building" in region_attributes:
                category_id = 1
            elif "window" in region_attributes:
                category_id = 0

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "buildings_" + d,
            lambda d=d: get_building_dicts("./via-2.0.8/buildings/" + d),
        )
        # MetadataCatalog.get("buildings_" + d).set(thing_classes=["buildings"])
    building_metadata = MetadataCatalog.get("buildings_train")

    dataset_dicts = get_building_dicts("./via-2.0.8/buildings/train")

    for i, d in enumerate(random.sample(dataset_dicts, 5)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        plt.imshow(vis.get_image()[:, :, ::-1])

        plt.savefig(f"./inputs/input_{i}.jpg")

    cfg = get_cfg()
    cfg.merge_from_file(
        "./detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.DATASETS.TRAIN = ("buildings_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)


    start = datetime.now()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.50  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("buildings_val",)
    predictor = DefaultPredictor(cfg)

    print(datetime.now() - start)

    dataset_dicts = get_building_dicts("./via-2.0.8/buildings/val")
    for i, dataset in enumerate(dataset_dicts):
 
        im = cv2.imread(dataset["file_name"])
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=building_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        #print(dir(dataset))
        #if i==5:
        #    print(outputs)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.savefig(f"./outputs/output_{i}.jpg")
