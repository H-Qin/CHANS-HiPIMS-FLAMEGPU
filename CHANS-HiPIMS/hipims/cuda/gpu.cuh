// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
#include <algorithm>

constexpr int CUDA_NUM_THREADS = 1024;

constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GET_BLOCKS(const int N) {
  // return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
  //  MAXIMUM_NUM_BLOCKS),1);
  // Use at least 1 block, since CUDA does not allow empty block

  return ((N - 1) / CUDA_NUM_THREADS + 1);
}
