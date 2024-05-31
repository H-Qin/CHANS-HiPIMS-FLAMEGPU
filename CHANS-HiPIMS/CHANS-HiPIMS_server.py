import torch
import math
import sys
import os
import numpy as np
import rasterio as rio

from hipims.pythonHipims.SWE_CUDA import Godunov
import hipims.pythonHipims.preProcessing as pre
import hipims.pythonHipims.postProcessing as post
from hipims.pythonHipims.SWE_CUDA import Godunov
from gevent.server import StreamServer
from mprpc import RPCServer
from threading import Thread

from osgeo import gdal

# ===============================================
# HiPIMS input path & files
# ===============================================

CASE_PATH = os.path.join(os.environ['HOME'], 'Eden_case')
RASTER_PATH = os.path.join(CASE_PATH, 'Eden_data')
OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
Rainfall_data_Path = os.path.join(CASE_PATH, 'data_rainfall_2015120421_0700.txt')
h_162000_Path = os.path.join(CASE_PATH, 'h_162000.tif')
qx_162000_Path = os.path.join(CASE_PATH, 'qx_162000.tif')
qy_162000_Path = os.path.join(CASE_PATH, 'qy_162000.tif')
ABM1_Path = os.path.join(CASE_PATH, 'DEM_ABM1_Whole.tif')
# ABM2_Path = os.path.join(CASE_PATH, 'DEM_ABM2_Whole.tif')
# ABM3_Path = os.path.join(CASE_PATH, 'DEM_ABM3_Whole.tif')

Manning = np.array([0.055, 0.055])

Degree = False
gauges_position = np.array([])
boundBox = np.array([])
bc_type = np.array([])
boundList = {}

rasterPath = {
    'DEM_path': os.path.join(RASTER_PATH, 'DEM_0.25_resample.tif'),
    'Landuse_path': os.path.join(RASTER_PATH, 'Landuse_0.25.tif'),
    'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask_0.25_resample.tif')
}
landLevel = 0

default_BC = 50

paraDict = {
    'deviceID': 5,
    'dx': 20.,
    'CFL': 0.5,
    'Manning': Manning,
    'Export_timeStep': 55 * 3600.,        
    't': 0.0,
    'export_n': 0,
    'secondOrder': False,
    'firstTimeStep': 1.0,
    'tensorType': torch.float64,
    'EndTime': 55 * 3600., 
    'Degree': Degree,
    'OUTPUT_PATH': OUTPUT_PATH,
    'rasterPath': rasterPath,
    'gauges_position': gauges_position,
    'boundBox': boundBox,
    'bc_type': bc_type,
    'landLevel': landLevel,
    'Rainfall_data_Path': Rainfall_data_Path,
    'boundList':boundList,
    'h_162000_Path': h_162000_Path,
    'qx_162000_Path': qx_162000_Path,
    'qy_162000_Path': qy_162000_Path,
    'ABM1_Path': ABM1_Path,
    # 'ABM2_Path': ABM2_Path,
    # 'ABM3_Path': ABM3_Path,
    'default_BC': default_BC
    }

# ===============================================
# HiPIMS
# ===============================================

class Simulation:
    results = None
    currentTime = None
    idx_target_ABM1 = None

    def __init__(self):
        self.initial_steps()

    def initial_steps(self):

        # ===============================================
        # Make output folder
        # ===============================================
        self.dt_list = []
        self.t_list = []
        if not os.path.isdir(paraDict['OUTPUT_PATH']):
            os.mkdir(paraDict['OUTPUT_PATH'])

        # ===============================================
        # set the device
        # ===============================================
        torch.cuda.set_device(paraDict['deviceID'])
        self.device = torch.device("cuda", paraDict['deviceID'])

        # ===============================================
        # set the tensors
        # ===============================================

        pre.setTheMainDevice(self.device)
        dem, mask, demMeta, gauge_index_1D = pre.importDEMData_And_BC(
            paraDict['rasterPath']['DEM_path'],
            self.device,
            gauges_position=paraDict['gauges_position'],
            boundBox=paraDict['boundBox'],default_BC=paraDict['default_BC'],
            bc_type=paraDict['bc_type'])
        if paraDict['Degree']:
            paraDict['dx'] = pre.degreeToMeter(demMeta['transform'][0])
        else:
            paraDict['dx'] = demMeta['transform'][0]
        if "Landuse_path" in paraDict['rasterPath']:
            landuse, landuse_index = pre.importLanduseData(
                paraDict['rasterPath']['Landuse_path'], self.device, paraDict['landLevel'])
        else:
            landuse = torch.zeros_like(dem, device=self.device)
            landuse = landuse.to(torch.int)
        rainfall_station_Mask = pre.importRainStationMask(
            paraDict['rasterPath']['Rainfall_path'], self.device)
        # gauge_index_1D = torch.tensor([100000], dtype=torch.int64)
        gauge_index_1D = gauge_index_1D.to(self.device)
        z = dem

        
        # ===============================================
        # Read ABMs with masks
        # ===============================================

        with rio.open(paraDict['rasterPath']['DEM_path']) as src:
                ABM1_DEM_Masked = src.read(1, masked=True)
        ABM1_DEM = np.ma.filled(ABM1_DEM_Masked, fill_value=-9999.)
        ABM1_DEM = torch.from_numpy(ABM1_DEM).to(device=self.device)

        with rio.open(paraDict['ABM1_Path']) as src:
                ABM1Masked = src.read(1, masked=True)
        ABM1 = np.ma.filled(ABM1Masked, fill_value=-9999.)
        ABM1 = torch.from_numpy(ABM1).to(device=self.device)

        idx_ABM1_in_whole = torch.where(ABM1 != -9999.)
        ABM1_DEM[idx_ABM1_in_whole] = 9999. # find index of ABM cells and stamp them as 9999
        ABM1_stamped_in_dem = ABM1_DEM[ABM1_DEM != -9999.]
        self.idx_target_ABM1 = torch.where(ABM1_stamped_in_dem > 9000)

        # with rio.open(paraDict['rasterPath']['DEM_path']) as src:
        #         ABM2_DEM_Masked = src.read(1, masked=True)
        # ABM2_DEM = np.ma.filled(ABM2_DEM_Masked, fill_value=-9999.)
        # ABM2_DEM = torch.from_numpy(ABM2_DEM).to(device=self.device)

        # with rio.open(paraDict['ABM2_Path']) as src:
        #         ABM2Masked = src.read(1, masked=True)
        # ABM2 = np.ma.filled(ABM2Masked, fill_value=-9999.)
        # ABM2 = torch.from_numpy(ABM2).to(device=self.device)

        # idx_ABM2_in_whole = torch.where(ABM2 != -9999.)
        # ABM2_DEM[idx_ABM2_in_whole] = 9999. # find index of ABM cells and stamp them as 9999
        # ABM2_stamped_in_dem = ABM2_DEM[ABM2_DEM != -9999.]
        # self.idx_target_ABM2 = torch.where(ABM2_stamped_in_dem > 9000)

        # with rio.open(paraDict['rasterPath']['DEM_path']) as src:
        #     ABM3_DEM_Masked = src.read(1, masked=True)
        # ABM3_DEM = np.ma.filled(ABM3_DEM_Masked, fill_value=-9999.)
        # ABM3_DEM = torch.from_numpy(ABM3_DEM).to(device=self.device)

        # with rio.open(paraDict['ABM3_Path']) as src:
        #         ABM3Masked = src.read(1, masked=True)
        # ABM3 = np.ma.filled(ABM3Masked, fill_value=-9999.)
        # ABM3 = torch.from_numpy(ABM3).to(device=self.device)

        # idx_ABM3_in_whole = torch.where(ABM3 != -9999.)
        # ABM3_DEM[idx_ABM3_in_whole] = 9999. # find index of ABM cells and stamp them as 9999
        # ABM3_stamped_in_dem = ABM3_DEM[ABM3_DEM != -9999.]
        # self.idx_target_ABM3 = torch.where(ABM3_stamped_in_dem > 9000)

        #######  read raster file from 162000 sec  #######
        with rio.open(paraDict['h_162000_Path']) as src:
                hMasked = src.read(1, masked=True)
        h = np.ma.filled(hMasked, fill_value=-9999.)
        h = torch.from_numpy(h).to(device=self.device, dtype=torch.float)

        with rio.open(paraDict['qx_162000_Path']) as src:
                qxMasked = src.read(1, masked=True)
        qx = np.ma.filled(qxMasked, fill_value=-9999.)
        qx = torch.from_numpy(qx).to(device=self.device, dtype=torch.float)

        with rio.open(paraDict['qy_162000_Path']) as src:
                qyMasked = src.read(1, masked=True)
        qy = np.ma.filled(qyMasked, fill_value=-9999.)
        qy = torch.from_numpy(qy).to(device=self.device, dtype=torch.float)

        wl = h + z

        # ===============================================
        # rainfall data
        # ===============================================

        self.rainfallMatrix = pre.voronoiDiagramGauge_rainfall_source(
            paraDict['Rainfall_data_Path'])

        if "climateEffect" in paraDict:
            self.rainfallMatrix[:, 1:] *= paraDict['climateEffect']

        # ===============================================
        # set field data
        # ===============================================
        self.numerical = Godunov(self.device,
                            paraDict['dx'],
                            paraDict['CFL'],
                            paraDict['Export_timeStep'],
                            t=paraDict['t'],
                            export_n=paraDict['export_n'],
                            firstTimeStep=paraDict['firstTimeStep'],
                            secondOrder=paraDict['secondOrder'],
                            tensorType=paraDict['tensorType'])
        self.numerical.setOutPutPath(paraDict['OUTPUT_PATH'])
        self.numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, self.device)
        self.numerical.set__frictionField_tensor(paraDict['Manning'], self.device)
        # self.numerical.set__infiltrationField_tensor(paraDict['hydraulic_conductivity'],
        #                                         paraDict['capillary_head'],
        #                                         paraDict['water_content_diff'],
        #                                         paraDict['deviceID'])
        self.numerical.set_landuse(mask, landuse, self.device)
        # ======================================================================
        self.numerical.set_distributed_rainfall_station_Mask(mask,
                                                        rainfall_station_Mask,
                                                        self.device)
        if 'boundList' in paraDict:
            self.numerical.set_boundary_tensor(paraDict['boundList'],self.device)

        del mask, landuse, h, qx, qy, wl, z, rainfall_station_Mask
        torch.cuda.empty_cache()
        self.numerical.exportField()

        self.gauge_dataStoreList = []

    def simulate_one_time_period(self):

        self.numerical.addFlux()
        self.numerical.addStation_PrecipitationSource(self.rainfallMatrix, self.device)
        self.numerical.time_friction_euler_update_cuda(self.device)
        self.dt_list.append(self.numerical.dt.item())
        self.t_list.append(self.numerical.t.item())
        # print("{:.3f}".format(self.numerical.t.item()))
        
        whole_catchment_result = self.numerical.get_h()
        self.results = whole_catchment_result[self.idx_target_ABM1]

    def get_latest_data(self):
        # print('get latest data: ', self.results.cpu())
        return self.results.cpu()

    def reset_simulation(self,):
        print('reseting simulate env')

# ===============================================
# Server
# ===============================================

class Server(RPCServer):
    target_time = 0
    current_time = 0
    simulation = Simulation()
    running_simulation = False

    def lock_when_simulate(self):
        while self.running_simulation:
            1

    def run_simulate_loop(self):
        while self.current_time < self.target_time:
            self.running_simulation = True
            self.simulation.simulate_one_time_period()
            self.current_time = self.simulation.numerical.t.item()
        self.current_time = int(self.current_time)
        self.running_simulation = False

    def get_latest_data(self):
        self.lock_when_simulate()

        data = self.simulation.get_latest_data()
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        print(type(data), data.shape)
        return data

    def sync_to_target_time(self,):
        self.lock_when_simulate()
        self.running_simulation = True
        print('running in a new thread')
        running_thread = Thread(target=self.run_simulate_loop)
        running_thread.start()

    def set_target(self, target_time, sync=True):
        target_time = int(target_time)
        dis = target_time - self.current_time
        print('set target time', target_time, 'current_time ', self.current_time)
        if target_time < self.current_time:
            raise Exception("Wrong Time")
        self.target_time = target_time
        if sync:
            self.sync_to_target_time()

    def request_data(self, time):
        time = int(time)

        # self.lock_when_simulate()
        if time > self.current_time:
            self.set_target(time)
        self.lock_when_simulate()
        ret = self.get_latest_data()
        print("ret (numpy array with full data): \n", ret)

        ret = ret.tolist()
        return ret

    def reset_simulation(self):
        print('reset...reset')
        self.lock_when_simulate()
        self.target_time = 0
        self.current_time = 0
        self.simulation.reset_simulation()

    def echo(self, x):
        return x



if __name__ == "__main__":
    server = StreamServer(('127.0.0.1', 6060), Server())
    print('Start server')
    server.serve_forever()