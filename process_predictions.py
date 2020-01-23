import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

font_id = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 15)
font_result = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 40)
text = "50%"
text_color = (255, 255, 255, 128)
background_bbox_window = (0, 247, 255, 50)
background_bbox_building = (255, 167, 14, 50)
background_text = (0, 0, 0, 150)
background_mask_window = (0, 247, 255, 100)
background_mask_building = (255, 167, 14, 100)
device = "cpu"

def draw_bounding_box(img, bounding_box, text, category, id, draw_box=False):
    x = bounding_box[0]
    y = bounding_box[3]

    draw = ImageDraw.Draw(img, "RGBA")
    if draw_box:
        if category == 0:
            draw.rectangle(bounding_box, fill=background_bbox_window, outline=(0, 0, 0))
        elif category == 1:
            draw.rectangle(bounding_box, fill=background_bbox_building, outline=(0, 0, 0))
    w, h = font_id.getsize(id)

    draw.rectangle((x, y, x + w, y - h), fill=background_text)
    draw.text((x , y - h), id, fill=text_color, font=font_id)
    if category == 1:
        w, h = font_result.getsize(text)
        draw.rectangle((x, y, x + w, y + h), fill=background_text)
        draw.text((x, y), text, fill=text_color, font=font_result)


def draw_mask(img, mask, category):

    img = img.convert("RGBA")

    mask_RGBA = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    if category == 0:
        mask_RGBA[mask] = background_mask_window
    elif category == 1:
        mask_RGBA[mask] = background_mask_building
    mask_RGBA[~mask] = [0, 0, 0, 0]
    rgb_image = Image.fromarray(mask_RGBA).convert("RGBA")

    img = Image.alpha_composite(img, rgb_image)

    return img

def calculate_window_perc(masks, categories, maybe_filename):
    # for each building loop for each window
    # check if and how many Trues of the window lie within the building
    # count the data in a list
    # add up all the Trues
    # count the pixels for each building mask as well
    # buidling Trues/building_mask -> % of window to fassade
    pass
    return window_perc

def create_csv(data):
    """for each building and window one row with
    add window id and size as well to csv file
    save result + building id + image name + mask size in csv file"""
    pass

def process_data(file_path, plot_data=True):
    # print(file_path)
    with open(file_path, "rb") as f:

        prediction = pickle.load(f)
        # print(dir(prediction["prediction"]["instances"]))
        # print(prediction["prediction"]["instances"].get_fields()["pred_masks"])
        # print(prediction["prediction"]["instances"].get_fields()["pred_boxes"])
        # print(dir(prediction["prediction"]["instances"].get_fields()["pred_boxes"]))
        # print(prediction["prediction"]["instances"].get_fields()["pred_classes"])
        # print(prediction["prediction"]["instances"].image_size)
        # print(prediction["prediction"]["instances"].get_fields()["pred_boxes"].tensor.to(device).numpy())
        boxes = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_boxes"]
            .tensor.to(device)
            .numpy()
        )

        img = Image.open(prediction["file_location"])
        categories = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_classes"]
            .to(device)
            .numpy()
        )
        masks = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_masks"]
            .to(device)
            .numpy()
        )

        dataset = []
        counter_window = 0
        counter_building = 0
        for i, box in enumerate(boxes):

            data = {}
            data["file_name"] = prediction["file_name"]
            data["file_location"] = prediction["file_location"]
            if categories[i] == 0:
                data["id"] = f"w_{counter_window}"
                counter_window = counter_window + 1
            elif categories[i] == 1:
                data["id"] = f"b_{counter_building}"
                counter_building = counter_building + 1

            data["bounding_box"] = box
            data["category"] = categories[i]
            data["mask"] = masks[i]
            dataset.append(data)

        for i, data in enumerate(dataset):
            calculate_window_perc()

        if plot_data:
            for i, data in enumerate(dataset):
                draw_bounding_box(
                    img,
                    data["bounding_box"],
                    text,
                    data["category"],
                    data["id"],
                    draw_box=True,
                )
            for i, data in enumerate(dataset):
                img = draw_mask(img, data["mask"], data["category"])


            img.save(f"./data/prediction_results/{data['file_name']}_prediction.png", quality=95)




def apply_mp_progress(func, n_processes, **kwargs):
    p = mp.Pool(n_processes)
    # print(kwargs)
    res_list = []
    with tqdm(total=len(kwargs["file_path"])) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, kwargs["file_path"], kwargs["plot_data"]))):
            pbar.update()
            res_list.append(res)
        pbar.close()
    p.close()
    p.join()
    return res_list

if __name__ == "__main__":

    prediction_folder = Path("./data/predictions/")
    prediction_list = []
    start = datetime.now()
    for i, file in enumerate(prediction_folder.glob("*.pkl")):
        
        file = str(file)
        prediction_list.append(file)
        # print(file)
        #file_name = file.split("\\")[-1]
        #print(file_name)
        #process_data(file, plot_data=True)

    apply_mp_progress(process_data,  mp.cpu_count(), **{"file_path": prediction_list, "plot_data": True})
    # process_data(file, plot_data=True)
    print(datetime.now()- start)




