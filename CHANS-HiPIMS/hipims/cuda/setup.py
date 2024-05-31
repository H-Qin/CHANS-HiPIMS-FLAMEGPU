from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('euler_update', [
        'euler_update.cpp',
        'euler_update_Kernel.cu',
    ]),    
    CUDAExtension('fluxCalculation_jh_modified_surface', [
        'fluxCal_jh_modified_surface.cpp',
        'fluxCal_jh_modified_surface_Kernel.cu',
    ]),    
    CUDAExtension('fluxCal_2ndOrder_jh_improved', [
        'fluxCal_2ndOrder_jh_improved.cpp',
        'flucCal_2ndOrder_jh_improved_Kernel.cu',
    ]),    
    CUDAExtension('frictionCalculation', [
        'friction_interface.cpp',
        'frictionCUDA_Kernel.cu',
    ]),
    CUDAExtension('sedi_c_euler_update', [
        'sedi_c_euler_update.cpp',
        'sedi_c_euler_update_kernel.cu',
    ]),
    CUDAExtension('sedi_mass_momentum_update', [
        'sedi_mass_momentum_update.cpp',
        'sedi_mass_momentum_kernel.cu',
    ]),
    CUDAExtension('fluxCalculation_convectionTranport', [
        'fluxCal_convectionTransport_interface.cpp',
        'fluxCal_convectionTransport_kernel.cu',
    ]),
    CUDAExtension('frictionCalculation_implicit', [
        'friction_implicit_interface.cpp',
        'friction_implicit_Kernel.cu',
    ]),
    CUDAExtension('friction_implicit_andUpdate_jh', [
        'friction_implicit_andUpdate_jh_interface.cpp',
        'friction_implicit_andUpdate_jh_Kernel.cu',
    ]),
    CUDAExtension('infiltrationCalculation', [
        'infiltration_interface.cpp',
        'infiltrationCUDA_Kernel.cu',
    ]),
    CUDAExtension('station_PrecipitationCalculation', [
        'stationPrecipitation_interface.cpp',
        'stationPrecipitation_Kernel.cu',
    ]),
    CUDAExtension('timeControl', [
        'timeControl.cpp',
        'timeControl_Kernel.cu',
    ]),
    CUDAExtension('fluxMask', [
        'fluxMaskGenerator.cpp',
        'fluxMaskGenerator_Kernel.cu',
    ]),

    CUDAExtension('randomTest', [
        'random_test_interface.cpp',
        'random_test_kernel.cu',
    ]),
]

# INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "HiPIMS"',
    author='Jiaheng Zhao',
    author_email='j.zhao@lboro.ac.uk',
    license='in CopyRight: in-house code',
    version='1.1.0',
    name='hipims',
    extra_compile_args={
        'cxx': ['-std=c++11', '-O2', '-Wall'],
        'nvcc': [
            '-std=c++11', '--expt-extended-lambda', '--use_fast_math',
            '-Xcompiler', '-Wall', '-gencode=arch=compute_60,code=sm_60',
            '-gencode=arch=compute_61,code=sm_61',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_72,code=sm_72',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_75,code=compute_75'
        ],
    },
    # packages=['hipims'],
    #     install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
