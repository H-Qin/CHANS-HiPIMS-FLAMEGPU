# encoding: utf-8
# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
"""
@author: Jiaheng Zhao
@license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
@contact: j.zhao@lboro.ac.uk
@software: hipims_torch
@file: swe.py
@time: 08.01.2020
@desc:
"""
import sys
import os
from numpy.lib.scimath import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
import math
import frictionCalculation
import timeControl
import euler_update
import infiltrationCalculation
import station_PrecipitationCalculation
import frictionCalculation_implicit
import friction_implicit_andUpdate_jh
import fluxMask
import fluxCalculation_jh_modified_surface
import fluxCal_2ndOrder_jh_improved
import sedi_mass_momentum_update
import sedi_c_euler_update
import fluxCalculation_convectionTranport


# print(torch.__version__)
"""
The direction of matrix
                            [1,1] |1-1|+1 = 1
                             |
                             |
|-1-1|+0 = 2 [-1, 0] <------ c ------->[1, 0] |1-1|+0 = 0
                             |
                             |
                            [-1,1] |-1-1|+1 = 3

index of direction = |dimension[0]-1|+dimension[1]

Returns:
    [type] -- [description]
"""


class Godunov:
    def __init__(self,
                 device,
                 dx,
                 CFL,
                 Export_timeStep,
                 t=0.0,
                 export_n=0,
                 secondOrder=False,
                 firstTimeStep=1.0e-4,
                 tensorType=torch.float64):
        super().__init__()
        self._tensorType = tensorType
        self._secondOrder = secondOrder
        self.dx = torch.tensor([dx], dtype=self._tensorType, device=device)
        self.cfl = torch.tensor([CFL], dtype=self._tensorType, device=device)

        self._manning = torch.tensor([0.0],
                                     dtype=self._tensorType,
                                     device=device)

        self.export_timeStep = Export_timeStep
        self.dt = torch.tensor([firstTimeStep],
                               dtype=self._tensorType,
                               device=device)
        self.t = torch.tensor([t], dtype=self._tensorType, device=device)
        self._export_n = export_n
        
        self._consider_sedimentMovement = False

        self._maxTimeStep = torch.tensor([60.],
                                         dtype=self._tensorType,
                                         device=device)
        # self._maxTimeStep = torch.tensor([CFL * dx / (sqrt(dx / 100.))],
        #                                  dtype=self._tensorType,
        #                                  device=device)
        self._outpath = "outpath"
        # self.init_fluidField_tensor(mask, h, qx, qy, wl, z, device)

        # self.__init__boundary_tensor()
        self._given_depth = torch.tensor([[0.0, 0.0]],
                                         dtype=self._tensorType,
                                         device=device)
        self._given_wl = torch.tensor([[0.0, 0.0]],
                                         dtype=self._tensorType,
                                         device=device)
        self._given_discharge = torch.tensor([[0.0, 0.0, 0.0]],
                                             dtype=self._tensorType,
                                             device=device)
        self.totalRain = torch.tensor([0.0],
                                      dtype=self._tensorType,
                                      device=device)

    def init__fluidField_tensor(self, mask, h, qx, qy, wl, z, device):

        # self.mask = mask

        # overlap = (mask < 0) & (mask > -9000)

        self._landuseMask = torch.zeros_like(h[mask > 0],
                                             dtype=torch.uint8,
                                             device=device)

        self._h_internal = torch.as_tensor(h[mask > 0].type(self._tensorType),
                                           device=device)
        self._qx_internal = torch.as_tensor(qx[mask > 0].type(
            self._tensorType),
                                            device=device)
        self._qy_internal = torch.as_tensor(qy[mask > 0].type(
            self._tensorType),
                                            device=device)
        self._wl_internal = torch.as_tensor(wl[mask > 0].type(
            self._tensorType),
                                            device=device)
        self._z_internal = torch.as_tensor(z[mask > 0].type(self._tensorType),
                                           device=device)
        self._h_max = torch.as_tensor(h[mask > 0].type(self._tensorType),
                                      device=device)

        self._h_update = torch.zeros_like(self._h_internal,
                                          dtype=self._tensorType,
                                          device=device)
        self._qx_update = torch.zeros_like(self._qx_internal,
                                           dtype=self._tensorType,
                                           device=device)
        self._qy_update = torch.zeros_like(self._qy_internal,
                                           dtype=self._tensorType,
                                           device=device)
        self._z_update = torch.zeros_like(self._z_internal,
                                          dtype=self._tensorType,
                                          device=device)
        self._wetMask = torch.flatten(
            (self._h_internal > 1.0e-6).nonzero()).type(torch.int32)

        del h, qx, qy, wl, z
        torch.cuda.empty_cache()
        # self._updatedMask = torch.flatten(
        #     (self._h_update.abs() > 1.0e-8).nonzero()).type(torch.int32)

        # if you want to change the block size 1024 to other, make sure you also change the cuda code
        # self.__accelerator = torch.zeros(int(
        #     (self.__h_internal.size(0) + 1024 - 1) / 1024),
        #                                  dtype=self.__tensorType,
        #                                  device=device)
        # self.__accelerator = torch.zeros_like(self.__h_internal,
        #                                       dtype=self.__tensorType,
        #                                       device=device)

        # =======================================================================
        # self.__index store the neighbor indexes and the self.__index[0] store the internal cell type or/and index
        # =======================================================================
        index_mask = torch.zeros_like(mask, dtype=torch.int32,
                                      device=device) - 1
        # now index are all -1
        index_mask[mask > 0] = torch.tensor(
            [i for i in range((mask[mask > 0]).size()[0])],
            dtype=torch.int32,
            device=device,
        )
        # index_mask[overlap] = mask[overlap]

        oppo_direction = torch.tensor([[-1, 1], [1, 0], [1, 1], [-1, 0]],
                                      device=device)
        self._normal = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=self._tensorType,
            device=device)

        # if self._secondOrder:
        #     self._index = torch.zeros(size=(9, self._h_internal.shape[0]),
        #                               dtype=torch.int32,
        #                               device=device)
        #     self._index[0] = mask[mask > 0]
        #     for i in range(4):
        #         self._index[i + 1] = (index_mask.roll(
        #             oppo_direction[i][0].item(),
        #             oppo_direction[i][1].item()))[mask > 0]

        #         self._index[i + 5] = (index_mask.roll(
        #             oppo_direction[i][0].item() * 2,
        #             oppo_direction[i][1].item()))[mask > 0]
        # else:
        #     self._index = torch.zeros(size=(5, self._h_internal.shape[0]),
        #                               dtype=torch.int32,
        #                               device=device)
        #     self._index[0] = mask[mask > 0]
        #     for i in range(4):
        #         self._index[i + 1] = (index_mask.roll(
        #             oppo_direction[i][0].item(),
        #             oppo_direction[i][1].item()))[mask > 0]
        self._index = torch.zeros(size=(5, self._h_internal.shape[0]),
                                  dtype=torch.int32,
                                  device=device)
        self._index[0] = mask[mask > 0]
        for i in range(4):
            self._index[i + 1] = (index_mask.roll(
                oppo_direction[i][0].item(),
                oppo_direction[i][1].item()))[mask > 0]

        print(self._index.size())
        self._index = torch.flatten(self._index)

        # print(self.__index.size())

        del index_mask, oppo_direction, mask
        torch.cuda.empty_cache()

    def add_concentrationField(self, mask, field, device):
        self._c_internal = torch.as_tensor(field[mask > 0].type(
            self._tensorType),
                                           device=device)
        del mask, field
        torch.cuda.empty_cache()

    def add_cumulativeWaterDepth_Field(self, device):
        self._cumulativeWaterDepth = torch.zeros_like(self._h_internal,
                                                      device=device)

    def set_boundary_tensor(self, boundList, device):
        if "H_GIVEN" in boundList:
            if type(boundList['H_GIVEN']) == str:
                given_depth = np.loadtxt(boundList['H_GIVEN'])
                self._given_depth = torch.tensor(given_depth,
                                                 dtype=self._tensorType,
                                                device=device)
            else:
                self._given_depth = torch.tensor(boundList['H_GIVEN'],
                                                dtype=self._tensorType,
                                                device=device)
        if "WL_GIVEN" in boundList:
            if type(boundList['WL_GIVEN']) == str:
                given_wl = np.loadtxt(boundList['WL_GIVEN'])
                self._given_wl = torch.tensor(given_wl,
                                                 dtype=self._tensorType,
                                                device=device)
            else:
                self._given_wl = torch.tensor(boundList['WL_GIVEN'],
                                                dtype=self._tensorType,
                                                device=device)
        if "Q_GIVEN" in boundList:
            if type(boundList['Q_GIVEN']) == str:
                given_discharge = np.loadtxt(boundList['Q_GIVEN'])
                self._given_discharge = torch.tensor(given_discharge,
                                                 dtype=self._tensorType,
                                                device=device)
            else:
                self._given_discharge = torch.tensor(boundList['Q_GIVEN'],
                                                dtype=self._tensorType,
                                                device=device)
            
        del boundList
        torch.cuda.empty_cache()

    def set_landuse(self, mask, landuseMask, device):
        self._landuseMask = torch.as_tensor(landuseMask[mask > 0],
                                            dtype=torch.uint8,
                                            device=device)
        del mask, landuseMask
        torch.cuda.empty_cache()

    def set_landuse_cpu_to_gpu(self, mask, landuseMask, device):
        self._landuseMask = torch.as_tensor(landuseMask[mask > 0],
                                            dtype=torch.uint8,
                                            device=device)
        # self._landuseMask = self._landuseMask.to(device=device)
        # del mask, landuseMask
        # torch.cuda.empty_cache()

    def observeGauges_write(self, observe_index, dataStoreList, n):
        # if n == 0:
        #     print(self._z_internal[observe_index])

        if n % 10 == 0:
            templist = []
            templist.append(self.t.item())
            templist += list(self._h_internal[observe_index].cpu().numpy())
            # templist += list(self._qx_internal[observe_index].cpu().numpy())
            # templist += list(self._qy_internal[observe_index].cpu().numpy())

            dataStoreList.append(templist)

    def set__frictionField_tensor(self, manning, device):
        if torch.is_tensor(manning):
            self._manning = manning.type(self._tensorType)
        else:
            self._manning = torch.tensor([manning],
                                         dtype=self._tensorType,
                                         device=device)

    def set__infiltrationField_tensor(self, hydraulic_conductivity,
                                      capillary_head, water_content_diff,
                                      device):
        self.add_cumulativeWaterDepth_Field(device)
        if torch.is_tensor(hydraulic_conductivity):
            self._hydraulic_conductivity = hydraulic_conductivity.type(
                self._tensorType)
        else:
            self._hydraulic_conductivity = torch.tensor(
                [hydraulic_conductivity],
                dtype=self._tensorType,
                device=device)
        # ========================================================
        if torch.is_tensor(capillary_head):
            self._capillary_head = capillary_head.type(self._tensorType)
        else:
            self._capillary_head = torch.tensor([capillary_head],
                                                dtype=self._tensorType,
                                                device=device)
        # ========================================================
        if torch.is_tensor(water_content_diff):
            self._water_content_diff = water_content_diff.type(
                self._tensorType)
        else:
            self._water_content_diff = torch.tensor([water_content_diff],
                                                    dtype=self._tensorType,
                                                    device=device)

    def addFlux(self):
        # pass
        if self._consider_sedimentMovement:
            self._wetMask = torch.zeros_like(self._h_internal,
                                             dtype=torch.bool)
            fluxMask.update(self._wetMask, self._h_internal, self._index,
                            self.t)
            self._wetMask = torch.flatten(self._wetMask.nonzero().type(
                torch.int32))
            fluxCalculation_convectionTranport.addFlux(
                self._wetMask,
                self._h_update,
                self._hc_update,
                self._qx_update,
                self._qy_update,
                self._h_internal,
                self._wl_internal,
                self._z_internal,
                self._c_internal,
                self._qx_internal,
                self._qy_internal,
                self._index,
                self._normal,
                self._given_depth,
                self._given_discharge,
                self.dx,
                self.t,
                self.dt,
            )
            torch.cuda.empty_cache()
        else:
            if self._secondOrder:                
                self._wetMask = torch.zeros_like(self._h_internal,
                                                dtype=torch.bool)
                fluxMask.update(self._wetMask, self._h_internal, self._index,
                                self.t)
                self._wetMask = torch.flatten(self._wetMask.nonzero().type(
                    torch.int32))
                fluxCal_2ndOrder_jh_improved.addFlux(
                    self._wetMask,
                    self._h_update,
                    self._qx_update,
                    self._qy_update,
                    self._h_internal,
                    self._wl_internal,
                    self._z_internal,
                    self._qx_internal,
                    self._qy_internal,
                    self._index,
                    self._normal,
                    self._given_depth,
                    self._given_wl,
                    self._given_discharge,
                    self.dx,
                    self.t,
                    self.dt,
                )
                torch.cuda.empty_cache()
            else:                
                self._wetMask = torch.zeros_like(self._h_internal,
                                                dtype=torch.bool)
                fluxMask.update(self._wetMask, self._h_internal, self._index,
                                self.t)
                self._wetMask = torch.flatten(self._wetMask.nonzero().type(
                    torch.int32))
                fluxCalculation_jh_modified_surface.addFlux(                
                    self._wetMask,
                    self._h_update,
                    self._qx_update,
                    self._qy_update,
                    self._h_internal,
                    self._wl_internal,
                    self._z_internal,
                    self._qx_internal,
                    self._qy_internal,
                    self._index,
                    self._normal,
                    self._given_depth,
                    self._given_wl,
                    self._given_discharge,
                    self.dx,
                    self.t,
                    self.dt,
                )
                print(self.t)
        torch.cuda.empty_cache()

    def addFriction(self):
        frictionCalculation.addFriction(
            self._wetMask,
            self._qx_update,
            self._qy_update,
            self._landuseMask,
            self._h_internal,
            self._qx_internal,
            self._qy_internal,
            self._manning,
            self.dt,
        )

    def addInfiltrationSource(self):
        infiltrationCalculation.addinfiltration(
            self._wetMask, self._h_update, self._landuseMask, self._h_internal,
            self._hydraulic_conductivity, self._capillary_head,
            self._water_content_diff, self._cumulativeWaterDepth, self.dt)

    # ====================================================
    # the station rainfall funcs
    # ====================================================
    def set_distributed_rainfall_station_Mask(self, mask,
                                              rainfall_station_Mask, device):
        self._rainfall_station_Mask = torch.as_tensor(
            rainfall_station_Mask[mask > 0], dtype=torch.int16, device=device)
        self._rainfall_station_time_index = 0
        # countList = []
        # for i in range(10):
        #     countList.append((self._rainfall_station_Mask == i).nonzero().size()[0])
        # print(countList)
        # print("DEBUG")
        del mask, rainfall_station_Mask
        torch.cuda.empty_cache()

    def set_distributed_rainfall_station_Mask_cpu_to_gpu(
            self, mask, rainfall_station_Mask, device):
        self._rainfall_station_time_index = 0
        self._rainfall_station_Mask = torch.as_tensor(
            rainfall_station_Mask[mask > 0], dtype=torch.int16, device=device)

        # self._rainfall_station_Mask = self._rainfall_station_Mask.to(
        #     device=device)
        # del mask, rainfall_station_Mask
        # torch.cuda.empty_cache()

    def __voronoiDiagramGauge_rainfall(self, rainfall_ndarray_data, device):
        if self.t.item() < rainfall_ndarray_data[-1, 0]:
            if self.t.item() < rainfall_ndarray_data[
                    self._rainfall_station_time_index + 1, 0]:
                per = (
                    self.t.item() -
                    rainfall_ndarray_data[self._rainfall_station_time_index, 0]
                ) / (rainfall_ndarray_data[self._rainfall_station_time_index +
                                           1, 0] -
                     rainfall_ndarray_data[self._rainfall_station_time_index,
                                           0])
                self._rainStationData = torch.from_numpy(
                    rainfall_ndarray_data[self._rainfall_station_time_index,
                                          1:] + per *
                    (rainfall_ndarray_data[self._rainfall_station_time_index +
                                           1, 1:] -
                     rainfall_ndarray_data[self._rainfall_station_time_index,
                                           1:])).to(device=device)
            else:
                self._rainfall_station_time_index += 1
                self.__voronoiDiagramGauge_rainfall(rainfall_ndarray_data,
                                                    device)
        else:
            self._rainStationData -= self._rainStationData

    def addStation_PrecipitationSource(self, rainfall_ndarray_data, device):
        self.__voronoiDiagramGauge_rainfall(rainfall_ndarray_data, device)
        # self.__h_update[:] += (self.__rainStationData[self.__rainfall_station_Mask]) * self.dt
        station_PrecipitationCalculation.addStation_Precipitation(
            self._h_update, self._rainfall_station_Mask, self._rainStationData,
            self.dt)
        # rainmaskcount = torch.tensor([
        #     1838959, 650223, 1161191, 2498, 1420051, 443506, 1333566, 48667,
        #     4311, 181100
        # ],
        #                              dtype=torch.float64,
        #                              device=device)
        # self.totalRain += torch.sum(
        #     self._rainStationData * rainmaskcount) * self.dt
        # totalWater = torch.sum(self._h_internal)
        # print('Rain: ', self.totalRain.item(), '\twater: ', totalWater.item())

    # ====================================================
    # the uniform rainfall funcs
    # ====================================================
    def set_uniform_rainfall_time_index(self):
        self._rainfall_station_time_index = 0

    def __update_rainfall_data(self, rainfall_ndarray_data, device):
        # if self._rainfall_station_time_index < rainfall_ndarray_data.shape[0]:
        if self.t.item() < rainfall_ndarray_data[
                self._rainfall_station_time_index + 1, 0]:
            per = (
                self.t.item() -
                rainfall_ndarray_data[self._rainfall_station_time_index, 0]
            ) / (rainfall_ndarray_data[self._rainfall_station_time_index + 1,
                                       0] -
                 rainfall_ndarray_data[self._rainfall_station_time_index, 0])
            self._rainStationData = torch.tensor([
                rainfall_ndarray_data[self._rainfall_station_time_index, 1] +
                per *
                (rainfall_ndarray_data[self._rainfall_station_time_index + 1,
                                       1] -
                 rainfall_ndarray_data[self._rainfall_station_time_index, 1])
            ],
                                                 dtype=self._tensorType,
                                                 device=device)
        else:
            if self.t.item() >= rainfall_ndarray_data[-1, 0]:
                self._rainStationData -= self._rainStationData
            else:
                self._rainfall_station_time_index += 1
                self.__update_rainfall_data(rainfall_ndarray_data, device)

    def add_uniform_PrecipitationSource(self, rainfall_ndarray_data, device):
        self.__update_rainfall_data(rainfall_ndarray_data, device)

        self._h_update[:] += self._rainStationData * self.dt
        # self._h_internal[:] += self._rainStationData * self.dt

    # ====================================================
    # sediment funcs
    # ====================================================

    def addRadarRainfall(self, rainfall_raster):
        self._h_update[:] += rainfall_raster * self.dt

    def get_h(self):
        return self._h_internal

    def get_qx(self):
        return self._qx_internal

    def get_qy(self):
        return self._qy_internal

    # ====================================================
    # sediment funcs
    # ====================================================
    def enableSediModule(self, paraDict, mask, z_nonMovable, device):
        """ 
        The sediment module will be enabled here. With this function, 
        the tensor used for sediment transport will be initialized.
        
        Tensor:
        C: volume concentration
        
        Scalar:
        rho_s: density of sediment
        rho_w: density of water
        d: medium diameter of sediment
        epsilon: calibration parameter, used for controlling the capacity of bedload transport
        p: porosity of bed sediment
        nu: water viscosity, 1.2e-6
        """
        self._c_internal = torch.zeros_like(self._h_internal,
                                            dtype=self._tensorType,
                                            device=device)
        self._hc_update = torch.zeros_like(self._h_internal,
                                           dtype=self._tensorType,
                                           device=device)

        # z_nonMovable is used for storing the initial bed elevation
        # maybe some area the base bottom is rock and there is no erosion or landslide

        self._z_nonMovable = torch.as_tensor(z_nonMovable[mask > 0],
                                             dtype=self._tensorType,
                                             device=device)
        del mask, z_nonMovable
        torch.cuda.empty_cache()

        self._consider_sedimentMovement = True

        # I will begin to implement the structured data for sediment
        sedi_para = np.array([float(paraDict["landUseTypeNumber"])])

        # add water density
        sedi_para = np.concatenate(
            (sedi_para, np.ones(paraDict["landUseTypeNumber"])))
        # add sediment density
        if "rho_w" in paraDict:
            sedi_para = np.concatenate((sedi_para, paraDict["rho_w"]))
        else:
            sedi_para = np.concatenate(
                (sedi_para, 2.65 * np.ones(paraDict["landUseTypeNumber"])))

        if "rho_s" in paraDict:
            sedi_para = np.concatenate((sedi_para, paraDict["rho_s"]))
        else:
            sedi_para = np.concatenate(
                (sedi_para, 2.65 * np.ones(paraDict["landUseTypeNumber"])))
        if "sedi_diameter" in paraDict:
            sedi_para = np.concatenate((sedi_para, paraDict["sedi_diameter"]))
        else:
            print(
                "The sediment diameter is need!!! The number should follow the types of land uses!!"
            )
        if "MPM_epsilon_calibration" in paraDict:
            sedi_para = np.concatenate(
                (sedi_para, paraDict["MPM_epsilon_calibration"]))
        else:
            sedi_para = np.concatenate(
                (sedi_para, np.ones(paraDict["landUseTypeNumber"])))
        if "sedi_porosity" in paraDict:
            sedi_para = np.concatenate((sedi_para, paraDict["sedi_porosity"]))
        else:
            print(
                "The sediment porosity is need!!! The number should follow the types of land uses!!"
            )
        if "repose_angle" in paraDict:
            sedi_para = np.concatenate((sedi_para, paraDict["repose_angle"]))
        else:
            sedi_para = np.concatenate(
                (sedi_para,
                 np.zeros(paraDict["landUseTypeNumber"]) + math.pi / 6.0))
        if "ThetaC_calibration" in paraDict:
            sedi_para = np.concatenate(
                (sedi_para, paraDict["ThetaC_calibration"]))
        else:
            sedi_para = np.concatenate(
                (sedi_para, np.ones(paraDict["landUseTypeNumber"])))
        
        # I will convert the parameters to a tensor
        self._sedi_para = torch.from_numpy(sedi_para).to(device=device)
        
        # print(self._sedi_para)

    def add_sedi_source(self):
        """ 
        This method will add the source due to sediment transport
        
        1: update the fluid depth h, h_update here.....
        2: update the bottom elevation z, z_update here......
        3: update the averaged sediment contration c, c is not needed to update at here, z_update can be used to update c later
        4: update the momentum source due to sediment transport
        """

        self._wetMask = torch.flatten(
            (self._h_internal >= 1.0e-6).nonzero()).type(torch.int32)
        
        sedi_mass_momentum_update.add_source(
            self._wetMask,
            self._index,
            self._h_internal,
            self._c_internal,
            self._qx_internal,
            self._qy_internal,
            self._z_internal,
            self._z_nonMovable,
            self._h_update,
            self._qx_update,
            self._qy_update,
            self._z_update,
            self._landuseMask,
            self._manning,
            self._sedi_para,
            self.dt,
            self.dx,
        )

        torch.cuda.empty_cache()

    def sedi_euler_update(self):
        self._wetMask = torch.flatten(
            ((self._h_update.abs() > 0.0) +
             (self._h_internal > 1.0e-6)).nonzero()).type(torch.int32)
        sedi_c_euler_update.update(
            self._wetMask,
            self._h_internal,
            self._c_internal,
            self._h_update,
            self._hc_update,
            self._z_update,
            self._sedi_para,
            self._landuseMask,
        )
        self._hc_update[:] = 0.0

    # self._h_update > 1.0e-6 used for labeling the dry cell to wet

    def euler_update(self):
        self._wetMask = torch.flatten(
            ((self._h_update.abs() > 0.0) +
             (self._h_internal > 1.0e-6)).nonzero()).type(torch.int32)

        euler_update.update(self._wetMask, self._h_update, self._qx_update,
                            self._qy_update, self._z_update, self._h_internal,
                            self._wl_internal, self._z_internal,
                            self._qx_internal, self._qy_internal)
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.

    def time_update_cuda(self, device):
        
        # limit the time step not bigger than the five times of the older time step
        UPPER = 10.
        time_upper = self.dt * UPPER

        self._wetMask = torch.flatten(
            (self._h_internal > 1.0e-6).nonzero()).type(torch.int32)
        self._accelerator_dt = torch.full(self._wetMask.size(),
                                          self._maxTimeStep.item(),
                                          dtype=self._tensorType,
                                          device=device)
        timeControl.updateTimestep(
            self._wetMask,
            self._accelerator_dt,
            self._h_max,
            self._h_internal,
            self._qx_internal,
            self._qy_internal,
            self.dx,
            self.cfl,
            self.t,
            self.dt,
        )

        if self._accelerator_dt.size(0) != 0:
            self.dt = torch.min(self._accelerator_dt)
        else:
            # do nothing, keep the last time step
            pass
        # self.dt = min(self.dt, self._maxTimeStep)
        self.dt = min(self.dt, time_upper)
        if (self.dt + self.t).item() >= float(self._export_n +
                                              1) * self.export_timeStep:
            self.dt = (self._export_n + 1) * self.export_timeStep - self.t
            self.exportField()
            self._export_n += 1
            print("give a output")
        self.t += self.dt

    def friction_euler_update_cuda(self):
        self._wetMask = torch.flatten(
            ((self._h_update.abs() > 0.0) +
             (self._h_internal >= 0.0)).nonzero()).type(torch.int32)
        friction_implicit_andUpdate_jh.addFriction_eulerUpdate(
            self._wetMask, self._h_update, self._qx_update, self._qy_update,
            self._z_update, self._landuseMask, self._h_internal,
            self._wl_internal, self._qx_internal, self._qy_internal,
            self._z_internal, self._manning, self.dt)
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.

    def time_friction_euler_update_cuda(self, device):
        
        # limit the time step not bigger than the five times of the older time step
        UPPER = 10.
        time_upper = self.dt * UPPER
        
        self._wetMask = torch.flatten(
            ((self._h_update.abs() > 0.0) +
             (self._h_internal >= 0.0)).nonzero()).type(torch.int32)
        friction_implicit_andUpdate_jh.addFriction_eulerUpdate(
            self._wetMask, self._h_update, self._qx_update, self._qy_update,
            self._z_update, self._landuseMask, self._h_internal,
            self._wl_internal, self._qx_internal, self._qy_internal,
            self._z_internal, self._manning, self.dt)
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.

        # self._wetMask = torch.flatten(
        #     (self._h_internal > 1.0e-6).nonzero()).type(torch.int32)
        self._accelerator_dt = torch.full(self._wetMask.size(),
                                          self._maxTimeStep.item(),
                                          dtype=self._tensorType,
                                          device=device)
        timeControl.updateTimestep(
            self._wetMask,
            self._accelerator_dt,
            self._h_max,
            self._h_internal,
            self._qx_internal,
            self._qy_internal,
            self.dx,
            self.cfl,
            self.t,
            self.dt,
        )

        if self._accelerator_dt.size(0) != 0:
            self.dt = torch.min(self._accelerator_dt)
        else:
            # do nothing, keep the last time step
            pass
        # self.dt = min(self.dt, self._maxTimeStep)
        self.dt = min(self.dt, time_upper)
        if (self.dt + self.t).item() >= float(self._export_n +
                                              1) * self.export_timeStep:
            self.dt = (self._export_n + 1) * self.export_timeStep - self.t
            self.exportField()
            self._export_n += 1
            print("give a output")
        self.t += self.dt

    def setOutPutPath(self, outpath):
        self._outpath = outpath

    def exportField(self):
        # we decide to save the data to pt files
        # the .pt file will be processed by numpy later at the postprocessing

        torch.save(
            self._h_internal,
            self._outpath + "/h_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._qx_internal,
            self._outpath + "/qx_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._qy_internal,
            self._outpath + "/qy_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._wl_internal,
            self._outpath + "/wl_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._h_max,
            self._outpath + "/h_max_" + str(self.t + self.dt) + ".pt",
        )
        if self._consider_sedimentMovement:
                torch.save(
                    self._z_internal,
                    self._outpath + "/z_" + str(self.t + self.dt) + ".pt",
                )
                torch.save(
                    self._c_internal,
                    self._outpath + "/c_" + str(self.t + self.dt) + ".pt",
                )

    def rungeKutta_update(self, rainfallMatrix, device):
        # clone all the physical field
        # temp_h = self._h_internal.clone()
        # temp_qx = self._qx_internal.clone()
        # temp_qy = self._qy_internal.clone()
        temp_h = self._h_internal + 0.
        temp_qx = self._qx_internal + 0.
        temp_qy = self._qy_internal + 0.

        self.addFlux()
        self.add_uniform_PrecipitationSource(rainfallMatrix, device)
        self.friction_euler_update_cuda()
        self.addFlux()
        self.add_uniform_PrecipitationSource(rainfallMatrix, device)
        self.friction_euler_update_cuda()

        self._h_internal[:] = (self._h_internal + temp_h) / 2.0
        self._qx_internal[:] = (self._qx_internal + temp_qx) / 2.0
        self._qy_internal[:] = (self._qy_internal + temp_qy) / 2.0
        self._wl_internal[:] = self._h_internal + self._z_internal

        del temp_h, temp_qx, temp_qy
        torch.cuda.empty_cache()

    def rungeKutta_update_nonUniformRain(self, rainfallMatrix, device):
        # clone all the physical field
        # temp_h = self._h_internal.clone()
        # temp_qx = self._qx_internal.clone()
        # temp_qy = self._qy_internal.clone()
        temp_h = self._h_internal + 0.
        temp_qx = self._qx_internal + 0.
        temp_qy = self._qy_internal + 0.

        self.addFlux()
        self.addStation_PrecipitationSource(rainfallMatrix, device)
        self.friction_euler_update_cuda()
        self.addFlux()
        self.addStation_PrecipitationSource(rainfallMatrix, device)
        self.friction_euler_update_cuda()

        self._h_internal[:] = (self._h_internal + temp_h) / 2.0
        self._qx_internal[:] = (self._qx_internal + temp_qx) / 2.0
        self._qy_internal[:] = (self._qy_internal + temp_qy) / 2.0
        self._wl_internal[:] = self._h_internal + self._z_internal

        del temp_h, temp_qx, temp_qy
        torch.cuda.empty_cache()

    def raster_update(self):
        self.h[self.internal] = self._h_internal
        self.wl[self.internal] = self._wl_internal
        self.z[self.internal] = self._z_internal
        self.qx[self.internal] = self._qx_internal
        self.qy[self.internal] = self._qy_internal

    def run_updating(self, device):
        self.addFlux()
        self.addFriction()
        self.time_update_cuda(device)


if __name__ == "__main__":
    # device = torch.device("cuda", 3)

    deviceID = 0

    torch.cuda.set_device(deviceID)

    device = torch.device("cuda", deviceID)

    print(torch.cuda.current_device())

    t = torch.tensor([0.0], device=device)
    n = torch.tensor([0], device=device)
    # mask, h, qx, qy, wl, z, manning, device, gravity, dx, CFL, h_SMALL, Export_timeStep, t, export_n
    tensorsize = (100, 100)

    mask = torch.ones(tensorsize, dtype=torch.int32, device=device)

    mask *= 10

    mask[1, :] = 30
    mask[-2, :] = 30
    mask[:, 1] = 30
    mask[:, -2] = 30
    mask[0, :] = -9999
    mask[-1, :] = -9999
    mask[:, 0] = -9999
    mask[:, -1] = -9999

    # h[:, 50:] = 0.0
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    z = torch.zeros(tensorsize, device=device)

    # z[:, 60:70] = 1.5

    for i in range(5):
        z[:, 60 + i] = i * 0.3
        z[:, 65 + i] = 1.5 - i * 0.3

    # z[:, 20:30] = 0.5
    wl = torch.ones(tensorsize, device=device)
    # wl[:, 50:] = 0.
    wl = torch.where(z > wl, z, wl)
    h = wl - z

    manning = 0.02
    CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output/dambreak')

    numerical = Godunov(device,
                        0.1,
                        0.5,
                        0.05,
                        0.0,
                        0,
                        secondOrder=False,
                        tensorType=torch.float64)
    numerical.setOutPutPath(OUTPUT_PATH)

    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)

    numerical.set__frictionField_tensor(manning, device)
    print("here")
    # n = 0
    numerical.exportField()
    del mask, h, qx, qy, wl, z, manning
    while numerical.t.item() < 5.0:

        # t1 = time.time()
        numerical.addFlux()
        # numerical.addFriction()
        # numerical.time_friction_euler_update_cuda(device)
        numerical.time_update_cuda(device)
        # t2 = time.time()
        # print("one time:", t2 - t1)
        # n += 1
        # print("times:", n)
        # t1 = time.time()
        # numerical.time_update(device)
        # t2 = time.time()
        # print("Timestep time:", t2-t1)

        # t2 = time.time()
        # print("One step time:", t2 - t1)

        print(numerical.dt)
