# HiPIMS-FLAMEGPU Coupled Human And Natural Systems (CHANS) modelling framework

The HiPIMS-FLAMEGPU Coupled Model Framework integrates the high-performance hydrodynamic modelling capabilities of HiPIMS with the flexible agent-based modelling environment of FLAMEGPU. This combined framework allows for the simulation of complex interactions between human activities and flooding processes. CHANS-HiPIMS and CHANS-FLAMEGPU are two modules of the framework and their interactions are based on mprpc library.

## Features

- High-resolution hydrodynamic simulations with HiPIMS.
- High-performance agent-based modeling with FLAMEGPU.
- Seamless integration between hydrodynamic and agent-based components.
- Scalable performance on multi-core CPUs and GPUs.
- User-friendly input/output formats.

### Prerequisites

CHANS-HiPIMS deployment requires the following software to be installed on the build machine.
- Anaconda Python environment management tool
- The CUDA 10.0 or 11.3 Toolkit (based on the HPC computational ability)
- Python 3.7 (CUDA==10.0) or 3.8 (CUDA==11.3)
- pip
- GCC and G++ ( >= 7 if CUDA is 10.0, >= 10 if CUDA is 11.3)

Dependencies for CHANS-FLAMEGPU:
- CUDA 8.0 or later
- Compute Capability 2.0 or greater GPU (CUDA 8)
    - Compute Capability 3.0 or greater GPU (CUDA 9)
- Windows
    - Microsoft Visual Studio 2015 or later
    - *Visualisation*:
        - `freeglut` and `glew` are included with FLAME GPU.
    - *Optional*: make
- Linux
    - `make`
    - `g++` (which supports the cuda version used)
    - `xsltproc`
    - *Visualistion*:
        - `GL` (deb: `libgl1-mesa-dev`, yum: `mesa-libGL-devel`)
        - `GLU` (deb: `libglu1-mesa-dev`, yum: `mesa-libGLU-devel`)
        - `GLEW` (deb: `libglew-dev`, yum: `glew-devel`)
        - `GLUT` (deb: `freeglut3-dev`, yum: `freeglut-devel`)
    - *Optional*: `xmllint`

#### Environment configuration for CHANS-HiPIMS
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

**Compilation and Installation**
The main HiPIMS library consists of `.cpp` and `.cu` code that must be compiled. `setuptools` is used to drive the compilation and installation of the library. The code and installation script is in the `cuda` subdirectory. The HiPIMS library is created as follows. 
```
cd $hipims/cuda/
python setup.py install
```

#### Building the CHANS-FLAMEGPU

FLAME GPU can be built for Windows and Linux. MacOS *should* work, but is unsupported.

**Windows using Visual Studio**

Visual Studio 2015 solutions are provided for the example FLAME GPU projects.
*Release* and *Debug* build configurations are provided, for both *console* mode and (optionally) *visualisation* mode.
Binary files are places in `bin/x64/<OPT>_<MODE>` where `<OPT>` is `Release` or `Debug` and `<MODE>` is `Console` or `Visualisation`.

***`make` for Linux and Windows***

`make` can be used to build FLAME GPU simulations under linux and windows (via a windows implementation of `make`).

Makefile is provided `CHANS-FLAMEGPU/CHANS_ABM/Makefile`), and for batch building (`CHANS_ABM/Makefile`).

To build a console project in release mode:

```
cd CHANS-FLAMEGPU/CHANS_ABM/
make console
```

##### Usage

***Runing CHANS-HiPIMS on Linux***
1. Navigate to the `CHANS-FLAMEGPU/CHANS_ABM/` directory
2. Run `python CHANS-HiPIMS_server.py`


CHANS-FLAMEGPU module can be executed as either a console application or as an interactive visualisation.

***Running CHANS-FLAMEGPU on Windows***

Assuming the `CHANS_ABM` project has been compiled for visualisation, there are several options for running the example.

1. Run the included batch script in `bin/x64/`: `CHANS_ABM_visualisation.bat`
2. Run the executable directly with an initial states file
    1. Navigate to the `CHANS-FLAMEGPU/CHANS_ABM/` directory in a command prompt
    2. Run `..\..\bin\x64\Release_Visualisation\CHANS_ABM.exe iterations\map.xml`

***Running CHANS-FLAMEGPU on Linux***

Assuming the `CHANS_ABM` project has been compiled for visualisation, there are several options for running the example.

1. Run the included bash script in `bin/linux-x64/`: `CHANS_ABM_visualisation.sh `
2. Run the executable directly with an initial states file
    1. Navigate to the `CHANS-HiPIMS-FLAMEGPU/CHANS-HiPIMS/` directory
    2. Run `../../bin/linux-x64/Release_Visualisation/CHANS_ABM iterations/map.xml`


###### Contribution

**HiPIMS for hydrodynamic modelling**
HiPIMS is maintained by the Hydro-Environmental Modelling Labarotory [URL](https://www.hemlab.org/), a research hub for technological innovation and interdisciplinary collaboration. We warmly welcome the hydro-environmental modelling community to contribute to the project, believing that this project will benefit the whole community.

 [Qiuhua Liang](https://www.lboro.ac.uk/departments/abce/staff/qiuhua-liang/), ([Q.Liang@lboro.ac.uk](mailto:Q.Liang@lboro.ac.uk)) is the Head of HEMLab.

***Authors of HiPIMS***
Jiaheng Zhao ([j.zhao@lboro.ac.uk ](mailto:j.zhao@lboro.ac.uk)), Xue Tong ([x.tong2@lboro.ac.uk](mailto:x.tong2@lboro.ac.uk))

***Co-authors of HiPIMS***
Haoyang Qin ([H.Qin@lboro.ac.uk](mailto:H.Qin@lboro.ac.uk)), Jinghua Jiang ([J.Jiang3@lboro.ac.uk](mailto:J.Jiang3@lboro.ac.uk)), Xiaoli Su ([X.Su@lboro.ac.uk](mailto:X.Su@lboro.ac.uk)), Kaicui Chen ([K.Chen@lboro.ac.uk](mailto:K.Chen@lboro.ac.uk)).
 

**FLAMEGPU for agent-based modelling**
FLAME GPU is developed as an open-source project by the [Visual Computing research group](https://www.sheffield.ac.uk/dcs/research/groups/visual-computing/home) in the [Department of Computer Science](https://www.sheffield.ac.uk/dcs/) at the [University of Sheffield](https://www.sheffield.ac.uk/).
The primary author is [Dr Paul Richmond](http://paulrichmond.shef.ac.uk/).

**CHANS modelling framework**
This HiPIMS-FLAMEGPU Coupled Human And Natural Systems (CHANS) modelling framework is developed by Haoyang Qin ([H.Qin@lboro.ac.uk](mailto:H.Qin@lboro.ac.uk)). For any questions or feedback, please open an issue on GitHub or contact Haoyang.
