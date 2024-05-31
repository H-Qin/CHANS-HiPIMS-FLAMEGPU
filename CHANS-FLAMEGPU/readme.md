# FLAME GPU

CHANS-FLAMEGPU is part of the HiPIMS-FLAMEGPU Coupled Model Framework developed by Haoyang Qin based on FLAMEGPU (Version: `1.5.0`).


FLAME GPU (Flexible Large-scale Agent Modelling Environment for Graphics Processing Units) is a high performance Graphics Processing Unit (GPU) extension to the FLAME framework.

It provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code.
This includes a number of key ABM building blocks such as multiple agent types, agent communication and birth and death allocation.
The advantages of our contribution are three fold.

1. Agent Based (AB) modellers are able to focus on specifying agent behaviour and run simulations without explicit understanding of CUDA programming or GPU optimisation strategies.
2. Simulation performance is significantly increased in comparison with desktop CPU alternatives. This allows simulation of far larger model sizes with high performance at a fraction of the cost of grid based alternatives.
3. Massive agent populations can be visualised in real time as agent data is already located on the GPU hardware.

## Documentation

The FLAME GPU documentation and user guide can be found at [http://docs.flamegpu.com](http://docs.flamegpu.com), with source hosted on GitHub at [FLAMEGPU/docs](https://github.com/FLAMEGPU/docs).


## Building FLAME GPU

FLAME GPU can be built for Windows and Linux. MacOS *should* work, but is unsupported.

### Dependencies
+ CUDA 8.0 or later
+ Compute Capability 2.0 or greater GPU (CUDA 8)
    + Compute Capability 3.0 or greater GPU (CUDA 9)
+ Windows
    + Microsoft Visual Studio 2015 or later
    + *Visualisation*:
        + `freeglut` and `glew` are included with FLAME GPU.
    + *Optional*: make
+ Linux
    + `make`
    + `g++` (which supports the cuda version used)
    + `xsltproc`
    + *Visualistion*:
        + `GL` (deb: `libgl1-mesa-dev`, yum: `mesa-libGL-devel`)
        + `GLU` (deb: `libglu1-mesa-dev`, yum: `mesa-libGLU-devel`)
        + `GLEW` (deb: `libglew-dev`, yum: `glew-devel`)
        + `GLUT` (deb: `freeglut3-dev`, yum: `freeglut-devel`)
    + *Optional*: `xmllint`


### Windows using Visual Studio

Visual Studio 2015 solutions are provided for the example FLAME GPU projects.
*Release* and *Debug* build configurations are provided, for both *console* mode and (optionally) *visualisation* mode.
Binary files are places in `bin/x64/<OPT>_<MODE>` where `<OPT>` is `Release` or `Debug` and `<MODE>` is `Console` or `Visualisation`.


### `make` for Linux and Windows

`make` can be used to build FLAME GPU simulations under linux and windows (via a windows implementation of `make`).

Makefile is provided `CHANS-FLAMEGPU/CHANS_ABM/Makefile`), and for batch building (`CHANS_ABM/Makefile`).

To build a console project in release mode:

```
cd CHANS-FLAMEGPU/CHANS_ABM/
make console
```


Binary files are places in `bin/linux-x64/<OPT>_<MODE>` where `<OPT>` is `Release` or `Debug` and `<MODE>` is `Console` or `Visualisation`.

For more information on building FLAME GPU via make, run `make help` in an example directory.

### Note on Linux Dependencies

If you are using linux on a managed system (i.e you do not have root access to install packages) you can provide shared object files (`.so`) for the missing dependencies.

I.e. `libglew` and `libglut`.

Download the required shared object files specific to your system configuration, and place in the `lib` directory. This will be linked at compile time and the dynamic linker will check this directory at runtime.

Alternatively, to package FLAME GPU executables with a different file structure, the `.so` files can be placed adjacent to the executable file. 

## Usage

FLAME GPU can be executed as either a console application or as an interactive visualisation.
Please see the [documentation](http://docs.flamegpu.com) for further details.


# Console mode
usage: executable [-h] [--help] input_path num_iterations [cuda_device_id] [XML_output_override]


For further details, see the [documentation](http://docs.flamegpu.com) or see `executable --help`.


### Running a Simulation on Windows

Assuming the `CHANS_ABM` project has been compiled for visualisation, there are several options for running the example.

1. Run the included batch script in `bin/x64/`: `CHANS_ABM_visualisation.bat`
2. Run the executable directly with an initial states file
    1. Navigate to the `CHANS-FLAMEGPU/CHANS_ABM/` directory in a command prompt
    2. Run `..\..\bin\x64\Release_Visualisation\CHANS_ABM.exe iterations\map.xml`

### Running a Simulation on Linux

Assuming the `CHANS_ABM` project has been compiled for visualisation, there are several options for running the example.

1. Run the included bash script in `bin/linux-x64/`: `CHANS_ABM_visualisation.sh `
2. Run the executable directly with an initial states file
    1. Navigate to the `CHANS-FLAMEGPU/CHANS_ABM/` directory
    2. Run `../../bin/linux-x64/Release_Visualisation/CHANS_ABM iterations/map.xml`


## Authors

FLAME GPU is developed as an open-source project by the [Visual Computing research group](https://www.sheffield.ac.uk/dcs/research/groups/visual-computing/home) in the [Department of Computer Science](https://www.sheffield.ac.uk/dcs/) at the [University of Sheffield](https://www.sheffield.ac.uk/).
The primary author is [Dr Paul Richmond](http://paulrichmond.shef.ac.uk/).


## Copyright and Software Licence

FLAME GPU is copyright the University of Sheffield 2009 - 2018. Version 1.5.X is released under the MIT open source [licence](LICENSE). Previous versions were released under a University of Sheffield End User licence agreement.