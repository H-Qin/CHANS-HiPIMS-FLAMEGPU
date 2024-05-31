################################################################################
# FLAME GPU Makefile 
# Copyright 2017 University of Sheffield.  All rights reserved.
# Authors : Dr Mozhgan Kabiri Chimeh, Peter Heywood, Dr Paul Richmond
# Contact : {m.kabiri-chimeh,p.heywood,p.richmond}@sheffield.ac.uk

# Modifications and enhancements made by Haoyang Qin, 2024.
# Loughborough University
# Contact: h.qin@lboro.ac.uk

# This file is part of the HiPIMS-FLAMEGPU Coupled Model Framework.
# All modifications and additions are the property of Haoyang Qin.
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# Haoyang Qin is strictly prohibited.
#
################################################################################
# USAGE : make help
################################################################################

# The name of the example (i.e. the name of the parent directory.)
EXAMPLE := CHANS_ABM

# Flag indicating if a visualistion target should exist
HAS_VISUALISATION := 1

# Flag indicating if a custom visulisation is used. 0 = no, 1 = yes.
CUSTOM_VISUALISATION := 0

# Relative path to the root FLAME GPU directory. Must end with a trainling slash.
FLAMEGPU_ROOT := ../../

# Parent directory for generated executable files.
EXAMPLE_BIN_DIR := $(FLAMEGPU_ROOT)bin

# Directory for temprorary files
EXAMPLE_BUILD_DIR := ./build

# Custom gencode arguments for this model (if any), otherwise the defaults for FLAME GPU are used from common.mk
# SMS ?= 50 60

# Optionally Disable XSLT generation of certain files. 
# TRANSFORM_HEADER_XSLT_DISABLED := 1
# TRANSFORM_FLAMEGPU_KERNALS_XSLT_DISABLED := 1
# TRANSFORM_IO_XSLT_DISABLED := 1
# TRANSFORM_SIMULATION_XSLT_DISABLED := 1
# TRANSFORM_MAIN_XSLT_DISABLED := 1
# TRANSFORM_VISUALISTION_XSLT_DISABLED := 1

# Include the comman makefile from the tools directory.
# If you wish to make a standalone project, you can simply replace this line with the contents of the common.mk file


LOCAL_SRC_FILES := ./lib/librpc.so
LOCAL_EXPORT_C_INCLUDES := ./include/rpc
include $(PREBUILT_SHARED_LIBRARY)


include $(FLAMEGPU_ROOT)tools/common.mk
