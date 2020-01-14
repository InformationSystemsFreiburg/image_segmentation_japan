# Image Segmentation with detectron2

## Installation (Windows)

- install [Python 3](https://www.python.org/ftp/python/3.8.1/python-3.8.1.exe) and add it to PATH
- install [git](https://git-scm.com/download/win) and add it to PATH
- clone this repository with
`git clone https://github.com/InformationSystemsFreiburg/imgage_segmentation_tokyo`
- install [Visual Studio Code Community Edition](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16) to get the GCC compiler

```bash
# check in terminal if gcc is installed, if not, try restarting the computer
gcc --version
# move to the folder of the image_segmentation-tokyo repo
cd .\imgage_segmentation_tokyo\
# create a new Python virtualenvironment
pip install virtualenv
virtualenv detectron2-env
.\detectron2-env\Scripts\activate
# install the following Python packages
pip install numpy pandas tqdm matplotlib seaborn psutil cython opencv-python
pip install "git+https://github.com/MarkusRosen/fvcore"
pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/MarkusRosen/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

Pytorch libraries have to be changed on Windows to work with `detectron2`

Location of first file:
`.\detectron2-env\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h`

Search for

```C
static constexpr size_t DEPTH_LIMIT = 128;
```

change to →

```C
static const size_t DEPTH_LIMIT = 128;
```

Location of second file:
`.\detectron2-env\Lib\site-packages\torch\include\pybind11\cast.h`

Search for

```C
explicit operator type&() { return *(this->value); }
```

change to →

```C
explicit operator type&() { return *((type*)this->value); }
```

```bash
# clone this repository
git clone https://github.com/MarkusRosen/detectron2
# move into the detectron2 folder
cd detectron2
# build the package (this will take a few minutes)
python setup.py build develop
# restart terminal/editor
# check if installation was successful with the following code:
.\detectron2-env\Scripts\activate
python
>>> from detectron2.utils.logger import setup_logger
>>> setup_logger()
<Logger detectron2 (DEBUG)>
>>> exit()
```

To check if everything is working with the models, download this image, save it as `input.jpg` and run the following code:
`http://images.cocodataset.org/val2017/000000439715.jpg`

```python
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import matplotlib.pyplot as plt

im = cv2.imread("./input.jpg")
plt.imshow(im)
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    "./detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(v.get_image()[:, :, ::-1])
# save image
plt.savefig("output.jpg")
```

The `output.jpg` should look like this:
![output](output.jpg "Output")

## Data Preparation

- Download [VGG Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.8.zip) and open the HTML file in the browser:
- Short introductions ([1](http://www.robots.ox.ac.uk/~vgg/software/via/docs/add_images_to_project.html), [2](http://www.robots.ox.ac.uk/~vgg/software/via/docs/drawing_regions.html)) on how to use the tool:

Exporting Annotations:

- Click Annotations → Export annotations in top menubar (COCO format)

Saving and Loading a Project:

- Click Project → Save in top menubar
- Click Project → Load in top menubar and select the json file containing VIA project.

describe the code for creating a dataset
