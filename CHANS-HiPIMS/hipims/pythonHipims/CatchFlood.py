# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import torch
import math
import sys
import os
import numpy as np
import time

try:
    from preProcessing import *
    from postProcessing import *
    from SWE_CUDA import Godunov
except ImportError:
    from .preProcessing import *
    from .postProcessing import *
    from .SWE_CUDA import Godunov

# from preProcessing import *
# from postProcessing import *
# from SWE_CUDA import Godunov

CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')
Rainfall_data_Path = os.path.join(CASE_PATH, 'rainSource.txt')
Manning = np.array([0.035, 0.1, 0.035, 0.04, 0.15, 0.03])
Degree = True

# CASE_PATH = os.path.join(os.environ['HOME'], 'Eden')
# RASTER_PATH = os.path.join(CASE_PATH, 'Tiff_Data', 'Tiff')
# OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')
# Rainfall_data_Path = os.path.join(CASE_PATH, 'Tiff_Data',
#                                       'rainRAD_2015120300_0800.txt')
# Manning = np.array([0.055, 0.075])
# Degree = False

rasterPath = {
    'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
    'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
    'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
}

paraDict = {
    'deviceID': 0,
    'dx': 90.,
    'CFL': 0.5,
    # 'Export_timeStep': 24. * 3600.,
    'Export_timeStep': 6. * 3600.,
    't': 0.0,
    'export_n': 0,
    'secondOrder': False,
    'firstTimeStep': 1.0,
    'tensorType': torch.float64,
    # 'EndTime': 384. * 3600.,
    'EndTime': 12. * 3600.,
    'Degree': Degree
}

# 1 耕地；0.035
# 2 林地；0.1
# 3 草地；0.035
# 4 水域； 0.04
# 5 建设用地；0.15
# 6 未利用地 0.03
landLevel = 1

dt_list = []
t_list = []


def run():

    # ===============================================
    # Make output folder
    # ===============================================
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # ===============================================
    # set the device
    # ===============================================
    torch.cuda.set_device(paraDict['deviceID'])
    device = torch.device("cuda", paraDict['deviceID'])

    # ===============================================
    # set the tensors
    # ===============================================
    setTheMainDevice(device)
    dem, mask, demMeta, gauge_index_1D = importDEMData_And_BC(
        rasterPath['DEM_path'], device)
    if paraDict['Degree']:
        paraDict['dx'] = degreeToMeter(demMeta['transform'][0])
    else:
        paraDict['dx'] = demMeta['transform'][0]

    landuse, landuse_index = importLanduseData(rasterPath['Landuse_path'],
                                               device, landLevel)
    rainfall_station_Mask = importRainStationMask(rasterPath['Rainfall_path'],
                                                  device)
    # gauge_index_1D = torch.tensor([100000], dtype=torch.int64)
    gauge_index_1D.to(device)
    z = dem
    h = torch.zeros_like(z, device=device)
    qx = torch.zeros_like(z, device=device)
    qy = torch.zeros_like(z, device=device)
    wl = h + z

    # ===============================================
    # rainfall data
    # ===============================================
    rainfallMatrix = voronoiDiagramGauge_rainfall_source(Rainfall_data_Path)

    # ===============================================
    # set field data
    # ===============================================
    numerical = Godunov(device,
                        paraDict['dx'],
                        paraDict['CFL'],
                        paraDict['Export_timeStep'],
                        t=paraDict['t'],
                        export_n=paraDict['export_n'],
                        firstTimeStep=paraDict['firstTimeStep'],
                        secondOrder=paraDict['secondOrder'],
                        tensorType=paraDict['tensorType'])
    numerical.setOutPutPath(OUTPUT_PATH)
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(Manning, device)
    numerical.set_landuse(mask, landuse, device)
    # ======================================================================
    numerical.set_distributed_rainfall_station_Mask(mask,
                                                    rainfall_station_Mask,
                                                    device)
    # ======================================================================
    # uniform rainfall test
    # ======================================================================
    # rainfallMatrix = np.array([[0.0, 0.0], [3600.0, 0.2 / 3600.0],
    #                            [3610.0, 0.0], [7200.0, 0.0]])
    # numerical.set_uniform_rainfall_time_index()
    # ======================================================================

    del mask, landuse, h, qx, qy, wl, z, rainfall_station_Mask
    torch.cuda.empty_cache()
    numerical.exportField()
    simulation_start = time.time()

    gauge_dataStoreList = []

    if gauge_index_1D.size()[0] > 0:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList)
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)            
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())
    else:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)            
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())
    simulation_end = time.time()
    dt_list.append(simulation_end - simulation_start)
    t_list.append(simulation_end - simulation_start)
    dt_array = np.array(dt_list)
    t_array = np.array(t_list)
    gauge_dataStoreList = np.array(gauge_dataStoreList)

    T = np.column_stack((t_array, dt_array))
    np.savetxt(OUTPUT_PATH + '/t.txt', T)
    np.savetxt(OUTPUT_PATH + '/gauges.txt', gauge_dataStoreList)
    exportRaster_tiff(rasterPath['DEM_path'], OUTPUT_PATH)


if __name__ == "__main__":
    run()