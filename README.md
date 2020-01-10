# Image Segmentation

## Installation (Windows)

Install git

pip install numpy pandas tqdm matplotlib seaborn psutil cython opencv-python
pip install git+https://github.com/facebookresearch/fvcore
pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

install https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16
check if gcc is installed
gcc --version

file1: 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
    static constexpr size_t DEPTH_LIMIT = 128;
      change to -->
    static const size_t DEPTH_LIMIT = 128;
file2: 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
    explicit operator type&() { return *(this->value); }
      change to -->
    explicit operator type&() { return *((type*)this->value); }


conda activate {your env}

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

git clone https://github.com/conansherry/detectron2

cd detectron2

python setup.py build develop

restart terminal/editor

