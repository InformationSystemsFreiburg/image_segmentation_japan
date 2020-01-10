# Image Segmentation

## Installation (Windows)

```bash
# install Python 3 and add it to PATH
https://www.python.org/ftp/python/3.8.1/python-3.8.1.exe
# install git and add it to PATH
https://git-scm.com/download/win
# install Visual Studio Code Community Edition to get the GCC compiler
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16
# check in terminal if gcc is installed, if not, try restarting the computer
gcc --version
# create a new Python virtualenvironment
pip install virtualenv
virtualenv detectron2-env
.\detectron2-env\Scripts\activate
# install the following Python packages
pip install numpy pandas tqdm matplotlib seaborn psutil cython opencv-python
pip install git+https://github.com/facebookresearch/fvcore
pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

Pytorch libraries have to be changed on Windows to work with `detectron2`
Location of first file:
`.\detectron2-env\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h`
Search for

```C
static constexpr size_t DEPTH_LIMIT = 128;
```

change to ->

```C
static const size_t DEPTH_LIMIT = 128;
```

Location of second file:
`.\detectron2-env\Lib\site-packages\torch\include\pybind11\cast.h`
Search for

```C
explicit operator type&() { return *(this->value); }
```

change to ->

```C
explicit operator type&() { return *((type*)this->value); }
```

```bash
# clone this repository
git clone https://github.com/conansherry/detectron2
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