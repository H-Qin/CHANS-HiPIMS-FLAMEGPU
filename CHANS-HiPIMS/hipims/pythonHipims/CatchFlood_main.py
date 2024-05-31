# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
from re import S
from tokenize import Double
import torch
import math
import sys
import os
import numpy as np
import rasterio as rio
import time

try:
    import postProcessing as post
    import preProcessing as pre
    from SWE_CUDA import Godunov
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    from .SWE_CUDA import Godunov


def run(paraDict):

    # ===============================================
    # Make output folder
    # ===============================================
    dt_list = []
    t_list = []
    if not os.path.isdir(paraDict['OUTPUT_PATH']):
        os.mkdir(paraDict['OUTPUT_PATH'])

    # ===============================================
    # set the device
    # ===============================================
    torch.cuda.set_device(paraDict['deviceID'])
    device = torch.device("cuda", paraDict['deviceID'])

    # ===============================================
    # set the tensors
    # ===============================================
    pre.setTheMainDevice(device)
    dem, mask, demMeta, gauge_index_1D = pre.importDEMData_And_BC(
        paraDict['rasterPath']['DEM_path'],
        device,
        gauges_position=paraDict['gauges_position'],
        boundBox=paraDict['boundBox'],
        bc_type=paraDict['bc_type'])
    if paraDict['Degree']:
        paraDict['dx'] = pre.degreeToMeter(demMeta['transform'][0])
    else:
        paraDict['dx'] = demMeta['transform'][0]

    if "Landuse_path" in paraDict['rasterPath']:
        landuse, landuse_index = pre.importLanduseData(
            paraDict['rasterPath']['Landuse_path'], device, paraDict['landLevel'])
    else:
        landuse = torch.zeros_like(dem, device=device)
        landuse = landuse.to(torch.int)
    rainfall_station_Mask = pre.importRainStationMask(
        paraDict['rasterPath']['Rainfall_path'], device)
    # gauge_index_1D = torch.tensor([100000], dtype=torch.int64)
    gauge_index_1D = gauge_index_1D.to(device)
    z = dem

    # with rio.open(paraDict['rasterPath']['DEM_path']) as src:
    #         ABM1_DEM_Masked = src.read(1, masked=True)
    # ABM1_DEM = np.ma.filled(ABM1_DEM_Masked, fill_value=-9999.)
    # ABM1_DEM = torch.from_numpy(ABM1_DEM).to(device=device)

    # with rio.open(paraDict['ABM1_Path']) as src:
    #         ABM1Masked = src.read(1, masked=True)
    # ABM1 = np.ma.filled(ABM1Masked, fill_value=-9999.)
    # ABM1 = torch.from_numpy(ABM1).to(device=device)

    # idx_ABM1_in_whole = torch.where(ABM1 != -9999.)
    # ABM1_DEM[idx_ABM1_in_whole] = 9999. # find index of ABM cells and stamp them as 9999
    # ABM1_stamped_in_dem = ABM1_DEM[ABM1_DEM != -9999.]
    # idx_target_ABM1 = torch.where(ABM1_stamped_in_dem>9000)

    # # start from 172800s:
    # with rio.open(paraDict['h_172800_Path']) as src:
    #         hMasked = src.read(1, masked=True)
    # h = np.ma.filled(hMasked, fill_value=-9999.)
    # h = torch.from_numpy(h).to(device=device, dtype=torch.float)

    # with rio.open(paraDict['qx_172800_Path']) as src:
    #         qxMasked = src.read(1, masked=True)
    # qx = np.ma.filled(qxMasked, fill_value=-9999.)
    # qx = torch.from_numpy(qx).to(device=device, dtype=torch.float)

    # with rio.open(paraDict['qy_172800_Path']) as src:
    #         qyMasked = src.read(1, masked=True)
    # qy = np.ma.filled(qyMasked, fill_value=-9999.)
    # qy = torch.from_numpy(qy).to(device=device, dtype=torch.float)

    h = torch.zeros_like(z, device=device)
    qx = torch.zeros_like(z, device=device)
    qy = torch.zeros_like(z, device=device)

    wl = h + z

    # ===============================================
    # rainfall data
    # ===============================================
    rainfallMatrix = pre.voronoiDiagramGauge_rainfall_source(
        paraDict['Rainfall_data_Path'])

    if "climateEffect" in paraDict:
        rainfallMatrix[:, 1:] *= paraDict['climateEffect']

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
    numerical.setOutPutPath(paraDict['OUTPUT_PATH'])
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(paraDict['Manning'], device)
    # numerical.set__infiltrationField_tensor(paraDict['hydraulic_conductivity'],
    #                                         paraDict['capillary_head'],
    #                                         paraDict['water_content_diff'],
    #                                         paraDict['deviceID'])
    numerical.set_landuse(mask, landuse, device)
    # ======================================================================
    numerical.set_distributed_rainfall_station_Mask(mask,
                                                    rainfall_station_Mask,
                                                    device)
    if 'boundList' in paraDict:
        numerical.set_boundary_tensor(paraDict['boundList'],device)
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
        n = 0
        while numerical.t.item() < paraDict['EndTime']:
            numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList, n)
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)
            numerical.addInfiltrationSource()
            # numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            # numerical.addFriction()
            # numerical.time_update_cuda(device)
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print("{:.3f}".format(numerical.t.item()))
            n+=1
            
    else:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)
            # numerical.addInfiltrationSource()
            # numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            # numerical.addFriction()
            # numerical.time_update_cuda(device)
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print("{:.3f}".format(numerical.t.item()))
            
            ## test getting h and getting partial data
            # results = numerical.get_h()
            # results_ABM1 = results[idx_target_ABM1]

    simulation_end = time.time()
    dt_list.append(simulation_end - simulation_start)
    t_list.append(simulation_end - simulation_start)
    dt_array = np.array(dt_list)
    t_array = np.array(t_list)
    gauge_dataStoreList = np.array(gauge_dataStoreList)

    T = np.column_stack((t_array, dt_array))
    np.savetxt(paraDict['OUTPUT_PATH'] + '/t.txt', T)
    np.savetxt(paraDict['OUTPUT_PATH'] + '/gauges.txt', gauge_dataStoreList)
    post.exportRaster_tiff(paraDict['rasterPath']['DEM_path'],
                           paraDict['OUTPUT_PATH'])


if __name__ == "__main__":
    CASE_PATH = os.path.join(os.environ['HOME'], 'Carlisle_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Carlisle_data')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'data_rainfall.txt')
    h_172800_Path = os.path.join(CASE_PATH, 'h_172800.tif')
    qx_172800_Path = os.path.join(CASE_PATH, 'qx_172800.tif')
    qy_172800_Path = os.path.join(CASE_PATH, 'qy_172800.tif')
    ABM1_Path = os.path.join(CASE_PATH, 'DEM_ABM1_Whole.tif')

    Manning = np.array([0.055])

    Degree = False
    gauges_position = np.array([])
    boundBox = np.array([])
    bc_type = np.array([])
    
    boundList = {
        # 'Q_GIVEN': given_discharge
    }
    
    default_BC = 50

    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM_0.25_resample.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask_0.25_resample.tif')
    }
    landLevel = 0

    paraDict = {
        'deviceID': 3,
        'dx': 20.,
        'CFL': 0.5,
        'Manning': Manning,
        'Export_timeStep': 1 * 3600.,        
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 1.0,
        'tensorType': torch.float64,
        'EndTime': 10 * 3600.,
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'rasterPath': rasterPath,
        'gauges_position': gauges_position,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'landLevel': landLevel,
        'Rainfall_data_Path': Rainfall_data_Path,
        'boundList':boundList,
        'h_172800_Path': h_172800_Path,
        'qx_172800_Path': qx_172800_Path,
        'qy_172800_Path': qy_172800_Path,
        'ABM1_Path': ABM1_Path,
        'default_BC': default_BC
    }

    run(paraDict)