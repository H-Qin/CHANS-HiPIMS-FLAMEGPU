# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math

try:
    import SWE_CUDA
    from SWE_CUDA import Godunov
except ImportError:
    from . import SWE_CUDA
    from .SWE_CUDA import Godunov


import timeControl
import euler_update
import friction_implicit_andUpdate_jh


class Parallel_SWE(Godunov):
    def __init__(self,
                 device,
                 dx,
                 CFL,
                 Export_timeStep,
                 rank,
                 world_size,
                 t=0.0,
                 export_n=0,
                 secondOrder=False,
                 firstTimeStep=1.0e-4,
                 tensorType=torch.float64):
        Godunov.__init__(self,
                         device,
                         dx,
                         CFL,
                         Export_timeStep,
                         t=t,
                         export_n=export_n,
                         secondOrder=secondOrder,
                         firstTimeStep=firstTimeStep,
                         tensorType=tensorType)
        # self.__export_n = export_n
        self.__rank = rank
        self.__world_size = world_size
        # self.__tensorType = tensorType
        # self.__secondOrder = secondOrder
        self.exportOption = False

    def __indexMaskCal(self, fieldValue, threshold):
        mask = fieldValue > threshold
        mask[:self._Offset[0]] = False
        mask[self._Offset[1]:] = False
        return torch.flatten(mask.nonzero()).type(torch.int32)

    def __indexDoubleMaskCal(self, fieldValue_0, fieldValue_1, threshold_0,
                             threshold_1):
        mask = fieldValue_0 > threshold_0
        mask += (fieldValue_1 > threshold_1)
        mask[:self._Offset[0]] = False
        mask[self._Offset[1]:] = False
        return torch.flatten(mask.nonzero()).type(torch.int32)

    def __update_wetMask(self):
        self._wetMask = self.__indexMaskCal(self._h_internal, 1.0e-6)

    def __update_updateMask(self):
        self._wetMask = self.__indexDoubleMaskCal(self._h_update.abs(),
                                                  self._h_internal, 1.0e-10,
                                                  1.0e-6)

    def init__fluidField_tensor(self, mask, overlap_mask, h, qx, qy, wl, z,
                                device):

        self._overlap_mask = overlap_mask.to(device)
        # set the label for overlap index
        """
        self.__internal_overlap_index = {1: 1}--------key: 0-> the upper or in 1D, the beginning of the tensor; 1-> the last, in 1D, used minus to represent
        self.__external_overlap_index = {1: 0} ----> value 0, means that the corresponding index in overlap arraymask is 0
        the dict here used for representing the index of internal and external overlap index located in the overlap array
        overlap data structure:
        overlap_mask = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,.....]
        h_overlap = [.........] same len
        wl_overlap = [.........] same len
        .
        .
        .
        
        the value can be up
        """
        # internal = mask > 0
        internal = (mask > 0).cpu()
        torch.cuda.empty_cache()
        if self.__rank == 0:
            self._internal_overlap_index = {1: 1}
            self._external_overlap_index = {1: 0}
            # __offset used for storing the overlap values, external, will not used for updating
            self._Offset = {0: 0, 1: -internal[-2:, :].nonzero().size()[0]}
            # __internal_overlay_offset used for updating overlap array, internal, used for updating
            self._internal_overlay_offset = {
                0: 0,
                1: -internal[-4:, :].nonzero().size()[0]
            }
            # internal[-2:, :] = False
        elif self.__rank == self.__world_size - 1:
            self._internal_overlap_index = {0: 2 * self.__rank - 2}
            self._external_overlap_index = {0: 2 * self.__rank - 1}
            # __offset used for storing the overlap values, external, will not used for updating
            self._Offset = {0: internal[:2, :].nonzero().size()[0], 1: -1}
            print(self._Offset)
            # __internal_overlay_offset used for updating overlap array, internal, used for updating
            self._internal_overlay_offset = {
                0: internal[:4, :].nonzero().size()[0],
                1: -1
            }
        else:
            self._internal_overlap_index = {
                0: 2 * self.__rank - 2,
                1: 2 * self.__rank + 1
            }
            self._external_overlap_index = {
                0: 2 * self.__rank - 1,
                1: 2 * self.__rank
            }
            # __offset used for storing the overlap values, external, will not used for updating
            self._Offset = {
                0: internal[:2, :].nonzero().size()[0],
                1: -internal[-2:, :].nonzero().size()[0]
            }
            # __internal_overlay_offset used for updating overlap array, internal, used for updating
            self._internal_overlay_offset = {
                0: internal[:4, :].nonzero().size()[0],
                1: -internal[-4:, :].nonzero().size()[0]
            }

        # self.__internal_overlap_index = {}

        # print("I want the internal index")
        # print(self.__internal_overlap_index)

        # I will create a overlap mask

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
        # [Offset[0]:-Offset[1]]
        self._wetMask = self.__indexMaskCal(self._h_internal, 1.0e-6)
        # self._updatedMask = self.__indexMaskCal(self._h_update.abs(), 1.0e-8)

        del h, qx, qy, wl, z, internal
        torch.cuda.empty_cache()
        self._normal = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=self._tensorType,
            device=device)

        # =======================================================================
        # self.__index store the neighbor indexes and the self.__index[0] store the
        # internal cell type or/and index. Here is the GPU version
        # =======================================================================
        # index_mask = torch.zeros_like(mask, dtype=torch.int32,
        #                               device=device) - 1
        # # now index are all -1
        # index_mask[mask > 0] = torch.tensor(
        #     [i for i in range((mask[mask > 0]).size()[0])],
        #     dtype=torch.int32,
        #     device=device,
        # )

        # oppo_direction = torch.tensor([[-1, 1], [1, 0], [1, 1], [-1, 0]],
        #                               device=device)

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

        # print(self._index.size())
        # self._index = torch.flatten(self._index)
        # =======================================================================
        # self.__index store the neighbor indexes and the self.__index[0] store the
        # internal cell type or/and index. Here is the CPU version
        # =======================================================================

        index_mask = torch.zeros_like(mask, dtype=torch.int32,
                                      device='cpu') - 1
        # now index are all -1
        mask_cpu = mask.cpu()
        torch.cuda.empty_cache()
        index_mask[mask_cpu > 0] = torch.tensor(
            [i for i in range((mask_cpu[mask_cpu > 0]).size()[0])],
            dtype=torch.int32)

        oppo_direction = torch.tensor([[-1, 1], [1, 0], [1, 1], [-1, 0]])

        self._index = torch.zeros(size=(5, self._h_internal.shape[0]),
                                  dtype=torch.int32,
                                  device='cpu')

        for i in range(4):
            self._index[i + 1] = (index_mask.roll(
                oppo_direction[i][0].item(),
                oppo_direction[i][1].item()))[mask_cpu > 0]
        self._index[0] = mask_cpu[mask_cpu > 0]
        print(self._index.size())
        self._index = torch.flatten(self._index)
        self._index = self._index.to(device=device)
        del index_mask
        torch.cuda.empty_cache()

        # ===================================================================

        # begin to create the overlap
        self._h_overlap = torch.zeros_like(self._overlap_mask,
                                           dtype=self._tensorType,
                                           device=device)
        self._wl_overlap = torch.zeros_like(self._overlap_mask,
                                            dtype=self._tensorType,
                                            device=device)
        self._qx_overlap = torch.zeros_like(self._overlap_mask,
                                            dtype=self._tensorType,
                                            device=device)
        self._qy_overlap = torch.zeros_like(self._overlap_mask,
                                            dtype=self._tensorType,
                                            device=device)
        self._z_overlap = torch.zeros_like(self._overlap_mask,
                                           dtype=self._tensorType,
                                           device=device)

        # print(self.__index.size())

        del oppo_direction, mask, overlap_mask
        torch.cuda.empty_cache()

    def update_overlap_receive(self):
        for k, v in self._external_overlap_index.items():
            # v here is used for the index for overlap_mask
            value_mask = self._overlap_mask == v
            if k == 0:
                self._h_internal[:self.
                                 _Offset[k]] = self._h_overlap[value_mask]
                self._wl_internal[:self.
                                  _Offset[k]] = self._wl_overlap[value_mask]
                self._qx_internal[:self.
                                  _Offset[k]] = self._qx_overlap[value_mask]
                self._qy_internal[:self.
                                  _Offset[k]] = self._qy_overlap[value_mask]
                self._z_internal[:self.
                                 _Offset[k]] = self._z_overlap[value_mask]
            elif k == 1:
                self._h_internal[self.
                                 _Offset[k]:] = self._h_overlap[value_mask]
                self._wl_internal[self.
                                  _Offset[k]:] = self._wl_overlap[value_mask]
                self._qx_internal[self.
                                  _Offset[k]:] = self._qx_overlap[value_mask]
                self._qy_internal[self.
                                  _Offset[k]:] = self._qy_overlap[value_mask]
                self._z_internal[self.
                                 _Offset[k]:] = self._z_overlap[value_mask]

    def get_h_overlap(self):
        return self._h_overlap

    def get_wl_overlap(self):
        return self._wl_overlap

    def get_z_overlap(self):
        return self._z_overlap

    def get_qx_overlap(self):
        return self._qx_overlap

    def get_qy_overlap(self):
        return self._qy_overlap

    def update_overlap_send(self):
        self._h_overlap[:] = 0.
        self._wl_overlap[:] = 0.
        self._qx_overlap[:] = 0.
        self._qy_overlap[:] = 0.
        self._z_overlap[:] = 0.

        for k, v in self._internal_overlap_index.items():
            # we need to makesure that all the overlap back to 0.0
            # later the ringallreduce will be adopt to gather the data by adding all them together
            value_mask = self._overlap_mask == v
            # v here is used for the index for overlap_mask
            if k == 0:
                self._h_overlap[value_mask] = self._h_internal[
                    self._Offset[k]:self._internal_overlay_offset[k]]
                self._wl_overlap[value_mask] = self._wl_internal[
                    self._Offset[k]:self._internal_overlay_offset[k]]
                self._qx_overlap[value_mask] = self._qx_internal[
                    self._Offset[k]:self._internal_overlay_offset[k]]
                self._qy_overlap[value_mask] = self._qy_internal[
                    self._Offset[k]:self._internal_overlay_offset[k]]
                self._z_overlap[value_mask] = self._z_internal[
                    self._Offset[k]:self._internal_overlay_offset[k]]
            elif k == 1:
                self._h_overlap[value_mask] = self._h_internal[
                    self._internal_overlay_offset[k]:self._Offset[k]]
                self._wl_overlap[value_mask] = self._wl_internal[
                    self._internal_overlay_offset[k]:self._Offset[k]]
                self._qx_overlap[value_mask] = self._qx_internal[
                    self._internal_overlay_offset[k]:self._Offset[k]]
                self._qy_overlap[value_mask] = self._qy_internal[
                    self._internal_overlay_offset[k]:self._Offset[k]]
                self._z_overlap[value_mask] = self._z_internal[
                    self._internal_overlay_offset[k]:self._Offset[k]]

    # def syncFieldTensor(self):
    #     dist.all_reduce(self.__h_overlap, op=dist.reduce_op.SUM, async_op=True)
    #     dist.all_reduce(self.__wl_overlap,
    #                     op=dist.reduce_op.SUM,
    #                     async_op=True)
    #     dist.all_reduce(self.__qx_overlap,
    #                     op=dist.reduce_op.SUM,
    #                     async_op=True)
    #     dist.all_reduce(self.__qy_overlap,
    #                     op=dist.reduce_op.SUM,
    #                     async_op=True)
    #     dist.all_reduce(self.__z_overlap, op=dist.reduce_op.SUM, async_op=True)

    # def syncTimeStepTensor(self):
    #     dist.all_reduce(self.dt, op=dist.reduce_op.MIN, async_op=True)
    #     self.t += self.dt

    def exportField(self):
        # we decide to save the data to pt files
        # the .pt file will be processed by numpy later at the postprocessing
        if self.t == (self._export_n + 1) * self.export_timeStep:
            torch.save(
                self._h_internal,
                self._outpath + "/h_" + str(self.t) + ".pt",
            )
            torch.save(
                self._qx_internal,
                self._outpath + "/qx_" + str(self.t) + ".pt",
            )
            torch.save(
                self._qy_internal,
                self._outpath + "/qy_" + str(self.t) + ".pt",
            )
            torch.save(
                self._wl_internal,
                self._outpath + "/wl_" + str(self.t) + ".pt",
            )
            torch.save(
                self._h_max,
                self._outpath + "/h_max_" + str(self.t) + ".pt",
            )
            self._export_n += 1

    def time_update_cuda(self, device):
        self.__update_updateMask()
        # print(self.__wetMask.size())
        
        # limit the time step not bigger than the five times of the older time step
        UPPER = 10.
        time_upper = self.dt * UPPER
        
        euler_update.update(self._wetMask, self._h_update, self._qx_update,
                            self._qy_update, self._z_update, self._h_internal,
                            self._wl_internal, self._z_internal,
                            self._qx_internal, self._qy_internal)

        self.__update_wetMask()

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
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.
        if self._accelerator_dt.size(0) != 0:
            self.dt = torch.min(self._accelerator_dt)
        # self.dt = min(self.dt, self._maxTimeStep)
        self.dt = min(self.dt, time_upper)
        if (self.dt + self.t).item() >= float(self._export_n +
                                              1) * self.export_timeStep:
            self.dt = (self._export_n + 1) * self.export_timeStep - self.t

    def time_friction_euler_update_cuda(self, device):
        
        # limit the time step not bigger than the five times of the older time step
        UPPER = 10.
        time_upper = self.dt * UPPER
        
        self.__update_updateMask()
        # print(self.__wetMask.size())
        friction_implicit_andUpdate_jh.addFriction_eulerUpdate(
            self._wetMask, self._h_update, self._qx_update, self._qy_update,
            self._z_update, self._landuseMask, self._h_internal,
            self._wl_internal, self._qx_internal, self._qy_internal,
            self._z_internal, self._manning, self.dt)
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.
        
        self.__update_wetMask()
        
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
        # self.dt = min(self.dt, self._maxTimeStep)
        self.dt = min(self.dt, time_upper)
        if (self.dt + self.t).item() >= float(self._export_n +
                                              1) * self.export_timeStep:
            self.dt = (self._export_n + 1) * self.export_timeStep - self.t

    def timeAddStep(self):
        self.t += self.dt