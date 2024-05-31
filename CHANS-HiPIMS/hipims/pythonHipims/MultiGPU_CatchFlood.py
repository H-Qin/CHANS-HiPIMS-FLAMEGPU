# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import torch
import torch.distributed as dist
import math
import sys
import os
import socket
import argparse
import numpy as np
from argparse import ArgumentParser, REMAINDER
import time

try:
    import Parallel_SWE_CUDA
    from Parallel_SWE_CUDA import Parallel_SWE
    from preProcessing import *
    from postProcessing import *
except ImportError:
    from . import Parallel_SWE_CUDA
    from .Parallel_SWE_CUDA import Parallel_SWE
    from .preProcessing import *
    from .postProcessing import *

# import Parallel_SWE_CUDA
# from Parallel_SWE_CUDA import Parallel_SWE
# import preProcessing as pre
# import postProcessing as post


# How to run
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 MultiGPU_CatchFlood.py


def tensorDistribute(t, deviceID):
    # deviceID = ['cuda:0', 'cuda:1']
    n_device = len(deviceID)
    rows = t.size()[0]

    step = math.ceil(rows / n_device)

    for i in range(n_device):
        if i == 0:
            t[0:step + 2, :].to(deviceID[i])
        elif i == n_device - 1:
            t[step * i - 2:, :].to(deviceID[i])
        else:
            t[step * i - 2:step * (i + 1) + 2, :].to(deviceID[i])
    del t


def ave_loadStep(z, size):
    step = np.zeros(size - 1, dtype=np.int32)
    z_cpu = z.cpu().numpy()
    count = np.count_nonzero(z_cpu)

    internal_index = np.argwhere(z_cpu > 0)

    for i in range(size - 1):
        step[i] = internal_index[
            int(float(i + 1) / float(size) * float(count)), 0] - 1
    return step


def ave_index_ToDevice(CELL_MASK, gauge_index_1D, rank, size):
    step = ave_loadStep(CELL_MASK, size)
    if rank == 0:
        number_begin = 0
        number_end = (CELL_MASK[:step[0] + 2, :] > 0).nonzero().size()[0]
        gauges_local = gauge_index_1D[(gauge_index_1D >= number_begin) +
                                      gauge_index_1D < number_end]
        gauges_local -= number_begin
        return gauges_local.to('cuda')
    elif rank == size - 1:
        number_begin = (CELL_MASK[:step[-1] - 2, :] > 0).nonzero().size()[0]
        number_end = (CELL_MASK > 0).nonzero().size()[0]
        gauges_local = gauge_index_1D[(gauge_index_1D >= number_begin) +
                                      gauge_index_1D < number_end]
        gauges_local -= number_begin
        return gauges_local.to('cuda')
    else:
        number_begin = (CELL_MASK[:step[rank - 1] - 2, :] >
                        0).nonzero().size()[0]
        number_end = (CELL_MASK[:step[rank] + 2, :] > 0).nonzero().size()[0]
        gauges_local = gauge_index_1D[(gauge_index_1D >= number_begin) +
                                      gauge_index_1D < number_end]
        gauges_local -= number_begin
        return gauges_local.to('cuda')


def ave_overlapMask(CELL_MASK, rank, size):
    step = ave_loadStep(CELL_MASK, size)
    overlap_mask = []
    for i in range(size):
        if i == 0:
            count = (CELL_MASK[step[0]:step[0] + 2, :] > 0).nonzero().size()[0]
            index = 0
            overlap_mask += count * [index]
        elif i == size - 1:
            count = (CELL_MASK[step[-1] - 2:step[-1], :] >
                     0).nonzero().size()[0]
            index = 2 * i - 1
            overlap_mask += count * [index]
        else:
            count_0 = (CELL_MASK[step[i - 1] - 2:step[i - 1], :] >
                       0).nonzero().size()[0]
            count_1 = (CELL_MASK[step[i]:step[i] + 2, :] >
                       0).nonzero().size()[0]
            index_0 = 2 * i - 1
            index_1 = 2 * i
            overlap_mask += count_0 * [index_0]
            overlap_mask += count_1 * [index_1]
    overlap_mask = torch.tensor(overlap_mask, dtype=torch.int32)
    return overlap_mask


def ave_tensorToDevice(step, t, rank, size):
    # step = ave_loadStep(CELL_MASK, size)
    if rank == 0:
        return t[0:step[0] + 2, :].to('cuda')
    elif rank == size - 1:
        return t[step[-1] - 2:, :].to('cuda')
    else:
        return t[step[rank - 1] - 2:step[rank] + 2, :].to('cuda')


def overlapMask(z, rank, size):
    step = math.ceil(z.size()[0] / size)
    mask = []
    for i in range(size):
        if i == 0:
            count = (z[step:step + 2, :] > 0).nonzero().size()[0]
            index = 0
            mask += count * [index]
        elif i == size - 1:
            count = (z[step * i - 2:step * i, :] > 0).nonzero().size()[0]
            index = 2 * i - 1
            mask += count * [index]
        else:
            count_0 = (z[step * i - 2:step * i, :] > 0).nonzero().size()[0]
            count_1 = (z[step * i:step * i + 2, :] > 0).nonzero().size()[0]
            index_0 = 2 * i - 1
            index_1 = 2 * i
            mask += count_0 * [index_0]
            mask += count_1 * [index_1]
    mask = torch.tensor(mask, dtype=torch.int32)
    return mask


def tensorToDevice(t, rank, size):
    step = math.ceil(t.size()[0] / size)
    if rank == 0:
        return t[0:step + 2, :].to('cuda')
    elif rank == size - 1:
        return t[step * rank - 2:, :].to('cuda')
    else:
        return t[step * rank - 2:step * (rank + 1) + 2, :].to('cuda')


def syncFieldTensor(t_overlap):
    dist.all_reduce(t_overlap,
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)


def syncTimeStepTensor(dt):
    dist.all_reduce(dt, op=torch.distributed.ReduceOp.MIN, async_op=False)


def syncFieldTensor(rank, size):
    pass


def run(rank, size, paraDict):
    print('Hello from rank', rank, 'of world size', size, 'on host',
          socket.gethostname())

    gauge_dataStoreList = []

    if rank == 0:
        simulation_start = time.time()
        print('torch version:', torch.__version__)
        print('torch.cuda.is_available():', torch.cuda.is_available())
        print('torch.cuda.device_count():', torch.cuda.device_count())
        print('torch.distributed.is_nccl_available():',
              dist.is_nccl_available())
        dt_list = []
        t_list = []
    dem_main, cell_type_mask_main, demMeta, gauge_index_1D = importDEMData_And_BC(
        paraDict['rasterPath']['DEM_path'], paraDict['device'])
    if paraDict['Degree']:
        paraDict['dx'] = degreeToMeter(demMeta['transform'][0])
    else:
        paraDict['dx'] = demMeta['transform'][0]
    step = ave_loadStep(cell_type_mask_main, size)
    z = ave_tensorToDevice(step, dem_main, rank, size)
    overlap_mask = ave_overlapMask(cell_type_mask_main, rank, size)
    cell_type_mask = ave_tensorToDevice(step, cell_type_mask_main, rank, size)

    # ========================================================
    # observe gauge setting
    # ========================================================
    observe_gauge_index = torch.tensor([]).to('cuda')
    if gauge_index_1D.size()[0] > 0:
        observe_gauge_index = ave_index_ToDevice(cell_type_mask_main,
                                                 gauge_index_1D, rank, size)
    # ========================================================
    del dem_main, cell_type_mask_main
    torch.cuda.empty_cache()

    landuse_main, landuse_index = importLanduseData(
        paraDict['rasterPath']['Landuse_path'], paraDict['device'],
        paraDict['landLevel'])
    landuse = ave_tensorToDevice(step, landuse_main, rank, size)
    del landuse_main
    torch.cuda.empty_cache()

    rainfall_station_Mask_main = importRainStationMask(
        paraDict['rasterPath']['Rainfall_path'], paraDict['device'])
    rainfall_station_Mask = ave_tensorToDevice(step,
                                               rainfall_station_Mask_main,
                                               rank, size)
    del rainfall_station_Mask_main
    torch.cuda.empty_cache()

    numerical = Parallel_SWE(paraDict['device'],
                             paraDict['dx'],
                             paraDict['CFL'],
                             paraDict['Export_timeStep'],
                             rank,
                             size,
                             t=paraDict['t'],
                             export_n=paraDict['export_n'],
                             secondOrder=paraDict['secondOrder'],
                             firstTimeStep=paraDict['firstTimeStep'],
                             tensorType=paraDict['tensorType'])

    # send the tensor to devices
    # z = ave_tensorToDevice(cell_type_mask_main, dem_main, rank, size)
    # landuse = ave_tensorToDevice(cell_type_mask_main, landuse_main, rank, size)
    # rainfall_station_Mask = ave_tensorToDevice(cell_type_mask_main,
    #                                            rainfall_station_Mask_main,
    #                                            rank, size)
    # cell_type_mask = ave_tensorToDevice(cell_type_mask_main, cell_type_mask_main, rank, size)

    # get the rainmatrix
    rainfallMatrix = voronoiDiagramGauge_rainfall_source(
        paraDict['Rainfall_data_Path'])

    h = torch.zeros_like(z, device=paraDict['device'])
    qx = torch.zeros_like(z, device=paraDict['device'])
    qy = torch.zeros_like(z, device=paraDict['device'])
    wl = h + z

    numerical.init__fluidField_tensor(cell_type_mask, overlap_mask, h, qx, qy,
                                      wl, z, paraDict['device'])
    numerical.setOutPutPath(paraDict['OUTPUT_PATH'])
    numerical.set__frictionField_tensor(paraDict['Manning'],
                                        paraDict['device'])

    # numerical.set_landuse(cell_type_mask, landuse, paraDict['device'])
    # numerical.set_distributed_rainfall_station_Mask(cell_type_mask,
    #                                                 rainfall_station_Mask,
    #                                                 paraDict['device'])
    # ======================================================================
    # use cpu to store the cell_type_mask, landuse and rainfall_station_Mask
    # save the gpu memory
    # ======================================================================
    cell_type_mask_cpu = cell_type_mask.cpu()
    landuse_cpu = landuse.cpu()
    rainfall_station_Mask_cpu = rainfall_station_Mask.cpu()
    torch.cuda.empty_cache()
    numerical.set_landuse_cpu_to_gpu(cell_type_mask_cpu, landuse_cpu,
                                     paraDict['device'])
    numerical.set_distributed_rainfall_station_Mask_cpu_to_gpu(
        cell_type_mask_cpu, rainfall_station_Mask_cpu, paraDict['device'])

    # initial the overlap values
    numerical.update_overlap_send()

    dist.all_reduce(numerical.get_h_overlap(),
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)
    dist.all_reduce(numerical.get_wl_overlap(),
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)
    dist.all_reduce(numerical.get_z_overlap(),
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)
    dist.all_reduce(numerical.get_qx_overlap(),
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)
    dist.all_reduce(numerical.get_qy_overlap(),
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=False)
    if observe_gauge_index.size()[0] > 0:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.update_overlap_receive()
            numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList)
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix,
                                                     paraDict['device'])
            # numerical.addFriction()
            # numerical.time_update_cuda(paraDict['device'])
            numerical.time_friction_euler_update_cuda(paraDict['device'])
            dist.all_reduce(numerical.dt,
                            op=torch.distributed.ReduceOp.MIN,
                            async_op=False)

            numerical.update_overlap_send()
            dist.all_reduce(numerical.get_h_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_wl_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_z_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_qx_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_qy_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            numerical.timeAddStep()
            if rank == 0:
                dt_list.append(numerical.dt.item())
                t_list.append(numerical.t.item())
                print(numerical.t.item())
            numerical.exportField()
    else:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.update_overlap_receive()
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix,
                                                     paraDict['device'])
            # numerical.addFriction()
            # numerical.time_update_cuda(paraDict['device'])
            numerical.time_friction_euler_update_cuda(paraDict['device'])
            dist.all_reduce(numerical.dt,
                            op=torch.distributed.ReduceOp.MIN,
                            async_op=False)

            numerical.update_overlap_send()
            dist.all_reduce(numerical.get_h_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_wl_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_z_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_qx_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            dist.all_reduce(numerical.get_qy_overlap(),
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False)
            numerical.timeAddStep()
            if rank == 0:
                dt_list.append(numerical.dt.item())
                t_list.append(numerical.t.item())
                print(numerical.t.item())
            numerical.exportField()
    gauge_dataStoreList = np.array(gauge_dataStoreList)
    np.savetxt(OUTPUT_PATH + '/' + str(rank) + '_gauges.txt',
               gauge_dataStoreList)
    if rank == 0:
        simulation_end = time.time()
        dt_list.append(simulation_end - simulation_start)
        t_list.append(simulation_end - simulation_start)
        dt_array = np.array(dt_list)
        t_array = np.array(t_list)

        T = np.column_stack((t_array, dt_array))
        np.savetxt(paraDict['OUTPUT_PATH'] + '/t.txt', T)
    # how can we ge the timestep


def init_processes(fn, paraDict, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend)

    fn(dist.get_rank(), dist.get_world_size(), paraDict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)

    CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'rainSource.txt')
    Manning = np.array([0.035, 0.1, 0.035, 0.04, 0.15, 0.03])
    Degree = True

    # CASE_PATH = os.path.join(os.environ['HOME'], 'Eden')
    # RASTER_PATH = os.path.join(CASE_PATH, 'Tiff_Data', 'Tiff')
    # OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    # Rainfall_data_Path = os.path.join(CASE_PATH, 'Tiff_Data',
    #                                   'rainRAD_2015120300_0800.txt')
    # Manning = np.array([0.055, 0.075])
    # Degree = False

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    else:
        pass

    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
    }

    # 1 耕地；0.035
    # 2 林地；0.1
    # 3 草地；0.035
    # 4 水域； 0.04
    # 5 建设用地；0.15
    # 6 未利用地 0.03

    paraDict = {
        'device': torch.device('cuda'),
        'dx': 90.,
        'CFL': 0.5,
        'Export_timeStep': 12. * 3600.,
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 60.0,
        'tensorType': torch.float64,
        # 'EndTime': 384. * 3600.,
        'EndTime': 48. * 3600.,
        'Degree': Degree,
        'landLevel': 1,
        'rasterPath': rasterPath,
        'Rainfall_data_Path': Rainfall_data_Path,
        'OUTPUT_PATH': OUTPUT_PATH,
        'Manning': Manning
    }

    # print(over_mask.size())
    init_processes(run, paraDict, backend='nccl')
    multi_exportRaster_tiff(rasterPath['DEM_path'], OUTPUT_PATH, 2)
