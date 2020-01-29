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
import pickle
from pathlib import Path
from tqdm import tqdm


def get_building_dicts(img_dir):
    """This function loads the JSON file created with the annotator and converts it to 
    the detectron2 metadata specifications.
    """
    # load the JSON file
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            # reformat the polygon information to fit the specifications
            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region_attributes = annotation["region_attributes"]["class"]

            # specify the category_id to match with the class.

            if "building" in region_attributes:
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
    # on Windows, PyTorch code has to be run within if __name__ == "__main__":
    # the data has to be registered within detectron2, once for the train and once for
    # the val data
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
        # the folder inputs has to be created first
        plt.savefig(f"./inputs/input_{i}.jpg")

    cfg = get_cfg()
    # you can choose alternative models as backbone here
    cfg.merge_from_file(
        # "./detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        "./detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )

    cfg.DATASETS.TRAIN = ("buildings_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    # if you changed the model above, you need to adapt the following line as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR, 0.00025 seems a good start
    cfg.SOLVER.MAX_ITER = (
        150000  # 300 iterations is a good start, for better accuracy increase this value
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512  # (default: 512), select smaller if faster training is needed
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # for the two classes window and building

    start = datetime.now()
    # for inferencing, the following 4 lines of code should be commented out
    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #trainer = DefaultTrainer(cfg)
    #trainer.resume_or_load(resume=False)
    #trainer.train()

    # load the trained weights from the output folder
    # cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.70  # set the testing threshold for this model
    )

    # load the validation data
    cfg.DATASETS.TEST = ("buildings_val",)
    # create a predictor
    predictor = DefaultPredictor(cfg)

    print("Time needed for training:", datetime.now() - start)
    start = datetime.now()

    validation_folder = Path("./via-2.0.8/buildings/val")

    for i, file in enumerate(validation_folder.glob("*.jpg")):
        file = str(file)
        file_name = file.split("\\")[-1]
        im = cv2.imread(file)

        outputs = predictor(im)
        output_with_filename = {}
        output_with_filename["file_name"] = file_name
        output_with_filename["file_location"] = file
        output_with_filename["prediction"] = outputs
        with open(f"./data/predictions/predictions_{i}.pkl", "wb") as f:
            pickle.dump(output_with_filename, f)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=building_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.savefig(f"./outputs/{file_name}.jpg")
    print("Time needed for inferencing:", datetime.now() - start)

