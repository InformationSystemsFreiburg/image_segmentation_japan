import cv2
from pathlib import Path

validation_folder = Path("./via-2.0.8/buildings/val")


for i, file in enumerate(validation_folder.glob('*.jpg')):
    file = str(file)   
    file_name = file.split("\\")[-1]
    print(file_name)
    im = cv2.imread(file)
