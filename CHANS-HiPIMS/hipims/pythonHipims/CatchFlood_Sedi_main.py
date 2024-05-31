# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 14.02.2022
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.

# I add sediment transport here
import torch
import math
import sys
import os
import numpy as np
import time

try:
    import postProcessing as post
    import preProcessing as pre
    from SWE_CUDA import Godunov
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    from .SWE_CUDA import Godunov


def run(paraDict, sediPara):

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
        boundBox=paraDict['boundBox'],default_BC=paraDict['default_BC'],
        bc_type=paraDict['bc_type'])
    if paraDict['Degree']:
        paraDict['dx'] = pre.degreeToMeter(demMeta['transform'][0])
    else:
        paraDict['dx'] = demMeta['transform'][0]

    landuse, landuse_index = pre.importLanduseData(
        paraDict['rasterPath']['Landuse_path'], device, paraDict['landLevel'])
    rainfall_station_Mask = pre.importRainStationMask(
        paraDict['rasterPath']['Rainfall_path'], device)
    # gauge_index_1D = torch.tensor([100000], dtype=torch.int64)
    gauge_index_1D = gauge_index_1D.to(device)
    z = dem
    h = torch.zeros_like(z, device=device)
    qx = torch.zeros_like(z, device=device)
    qy = torch.zeros_like(z, device=device)
    wl = h + z

    z = z_nonMovable.clone()

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
    numerical.set__infiltrationField_tensor(paraDict['hydraulic_conductivity'],
                                            paraDict['capillary_head'],
                                            paraDict['water_content_diff'],
                                            paraDict['deviceID'])
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

    # enable sediment transport
    numerical.enableSediModule(sedi_para, mask, z_nonMovable, device)

    del mask, landuse, h, qx, qy, wl, z, rainfall_station_Mask
    torch.cuda.empty_cache()
    numerical.exportField()
    simulation_start = time.time()

    gauge_dataStoreList = []

    if gauge_index_1D.size()[0] > 0:
        n = 0
        while numerical.t.item() < paraDict['EndTime']:
            numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList, n)

            numerical.add_sedi_source()
            numerical.addFlux()
            numerical.sedi_euler_update()

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
            numerical.add_sedi_source()
            numerical.addFlux()
            numerical.sedi_euler_update()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)
            numerical.addInfiltrationSource()
            # numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            # numerical.addFriction()
            # numerical.time_update_cuda(device)
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print("{:.3f}".format(numerical.t.item()))
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

    # 1 耕地；0.035
    # 2 林地；0.1
    # 3 草地；0.035
    # 4 水域； 0.04
    # 5 建设用地；0.15
    # 6 未利用地 0.03

    # CASE_PATH = os.path.join(os.environ['HOME'], 'Eden')
    # RASTER_PATH = os.path.join(CASE_PATH, 'Tiff_Data', 'Tiff')
    # OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')
    # Rainfall_data_Path = os.path.join(CASE_PATH, 'Tiff_Data',
    #                                       'rainRAD_2015120300_0800.txt')
    # Manning = np.array([0.055, 0.075])
    # Degree = False

    CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'rainSource.txt')
    Manning = np.array([0.035, 0.1, 0.035, 0.04, 0.15, 0.03])
    Degree = True

    gauges_position = np.array([])
    boundBox = np.array([])
    bc_type = np.array([])

    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
    }

    landLevel = 1

    paraDict = {
        'deviceID': 1,
        'dx': 90.,
        'CFL': 0.5,
        'Manning': Manning,
        # 'Export_timeStep': 24. * 3600.,
        'Export_timeStep': 6. * 3600.,
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 1.0,
        'tensorType': torch.float64,
        # 'EndTime': 384. * 3600.,
        'EndTime': 12. * 3600.,
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'rasterPath': rasterPath,
        'gauges_position': gauges_position,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'landLevel': landLevel,
        'Rainfall_data_Path': Rainfall_data_Path
    }

    sediPara = {
        "landUseTypeNumber": 1, # the number of total land use type
        "rho_w": np.array([1.0]), #density of water
        "rho_s": np.array([2.615]), # density of sediment
        "sedi_diameter": np.array([2.5e-4]), # diameter of sediment
        "MPM_epsilon_calibration": np.array([4.8]), # calibration of mpm eq.
        "sedi_porosity": np.array([0.36]), # sediment porosity
        "repose_angle": np.array([math.pi * wet / 180.]), # sediment repose angle
        "ThetaC_calibration": np.array([1.0]), # calibration para for thetaC
    }

    run(paraDict, sediPara)