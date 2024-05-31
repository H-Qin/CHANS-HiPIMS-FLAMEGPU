# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import os
__version__ = '1.0.0'

print("         Welcome to the HiPIMS ", __version__)

dir_path = os.path.dirname(os.path.realpath(__file__))

f = open(os.path.join(dir_path, 'banner.txt'), 'r')
file_contents = f.read()
print(file_contents)
f.close()