# High-Performance Integrated hydrodynamic Modelling System ***hybrid***

The file "CHANS-HiPIMS_server.py" is part of the HiPIMS-FLAMEGPU Coupled Model Framework developed by Haoyang Qin based on HiPIMS.

## About
HiPIMS names for High-Performance Integrated hydrodynamic Modelling System. It uses state-of-art numerical schemes (Godunov-type finite volume) to solve the 2D shallow water equations for flood simulations. To support high-resolution flood simulations, HiPIMS is implemented on multiple GPUs (Graphics Processing Unit) using CUDA/C++ languages to achieve high-performance computing. To find out how to use the model, please see the wiki.

### Contributing
HiPIMS is maintained by the Hydro-Environmental Modelling Labarotory [URL](https://www.hemlab.org/), a research hub for technological innovation and interdisciplinary collaboration. We warmly welcome the hydro-environmental modelling community to contribute to the project, believing that this project will benefit the whole community.

 [Qiuhua Liang](https://www.lboro.ac.uk/departments/abce/staff/qiuhua-liang/), ([Q.Liang@lboro.ac.uk](mailto:Q.Liang@lboro.ac.uk)) is the Head of HEMLab.

#### Authors
Jiaheng Zhao, HemLab ([j.zhao@lboro.ac.uk ](mailto:j.zhao@lboro.ac.uk))

Xue Tong, Loughborough University ([x.tong2@lboro.ac.uk](mailto:x.tong2@lboro.ac.uk))  
  
#### Co-authors
Kaicui Chen, Loughborough University ([K.Chen@lboro.ac.uk](mailto:K.Chen@lboro.ac.uk)) 

Xiaoli Su, Loughborough University ([X.Su@lboro.ac.uk](mailto:X.Su@lboro.ac.uk)) 

Jinghua Jiang, Loughborough University ([J.Jiang3@lboro.ac.uk](mailto:J.Jiang3@lboro.ac.uk)) 

Haoyang Qin, Loughborough University ([H.Qin@lboro.ac.uk](mailto:H.Qin@lboro.ac.uk)) 

## Requirements
HiPMS deployment requires the following software to be installed on the build machine.
- Anaconda Python environment management tool
- The CUDA 10.0 or 11.3 Toolkit (based on the HPC computational ability)
- Python 3.7 (CUDA==10.0) or 3.8 (CUDA==11.3)
- pip
- GCC and G++ ( >= 7 if CUDA is 10.0, >= 10 if CUDA is 11.3)

## Environment configuration
_a. Package manager_
Anaconda will be chosen as the package manager in HiPIMS 3.0.
- Install Anaconda if needed:
```
cd /tmp
curl https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -o anaconda3.sh
bash anaconda3.sh
```

_b. check the CUDA version of your environment_
```
nvcc -V
```

_c. Create and activate an Anaconda virtual environment with Python_

***if CUDA==10.0***
```
conda create --name torch_hipims python=3.7
conda activate torch_hipims
```
***if CUDA==11.3***
```
conda create --name torch_hipims python=3.8 
conda activate torch_hipims
```

_d. Install pytorch, torchvision and the CUDA toolkit_

***if CUDA==10.0***
```
conda install -c conda-forge -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
```
***if CUDA==11.3***
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

_e. Install the python packages used by HiPIMS_
```
conda install numpy matplotlib pyqt seaborn rasterio geopandas
pip install hipims_io
```

## Compilation and Installation
The main HiPIMS library consists of `.cpp` and `.cu` code that must be compiled. `setuptools` is used to drive the compilation and installation of the library. The code and installation script is in the `cuda` subdirectory. The HiPIMS library is created as follows. 
```
cd $HiPIMS_ROOT/cuda/
python setup.py install
```
