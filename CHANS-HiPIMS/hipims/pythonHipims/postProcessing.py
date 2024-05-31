# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import torch
import glob
import os
import rasterio as rio
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import colors
import seaborn as sns
import numpy as np
from rasterio.windows import Window
try:
    from preProcessing import *
except ImportError:
    from .preProcessing import *

def exportRaster_tiff(DEM_path, outPutPath):
    from glob import glob
    result_list = glob(outPutPath + '/*.pt')
    result_list_qx = glob(outPutPath + '/qx*.pt')
    result_list_qy = glob(outPutPath + '/qy*.pt')
    result_list_h = glob(outPutPath + '/h_tensor*.pt')

    result_list_h.sort()
    result_list_qx.sort()
    result_list_qy.sort()

    device = torch.device("cuda",
                          int(result_list[0][result_list[0].rfind(':') + 1]))
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    dem = dem.to(torch.float32)
    z = dem.clone()

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()
    print(len(result_list_h))
    # for i in range(len(result_list_h)):
    #     internal_data_h = torch.load(result_list_h[i])
    #     internal_data_qx = torch.load(result_list_qx[i])
    #     internal_data_qy = torch.load(result_list_qy[i])

    #     internal_data_u = torch.where(internal_data_h > 1.0e-6,
    #                                   internal_data_qx / internal_data_h,
    #                                   internal_data_qx - internal_data_qx)
    #     print("max and min U value")
    #     max_value, max_index = torch.max(internal_data_u, 0)
    #     min_value, min_index = torch.min(internal_data_u, 0)
    #     print("Max U: ", max_value.item(), "\th: ",
    #           internal_data_h[max_index].item(),
    #           "\tqx: ", internal_data_qx[max_index].item(), "\tindex: ",
    #           max_index.item())
    #     print("Min U: ", min_value.item(), "\th: ",
    #           internal_data_h[min_index].item(),
    #           "\tqx: ", internal_data_qx[min_index].item(), "\tindex: ",
    #           min_index.item())

    #     dem[~mask] = internal_data_u.to(torch.float32)
    #     # print(dem)
    #     data_cpu = dem.cpu().numpy()
    #     data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
    #     nodatavalue = -9999.
    #     data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
    #     DATA_meta = demMeta.copy()
    #     DATA_meta.update({'nodata': nodatavalue})
    #     DATA_meta.update({'dtype': np.float32})
    #     print("2D index of U")
    #     print(np.argwhere(data_cpu == max_value.item()))
    #     print(np.argwhere(data_cpu == min_value.item()))

    #     topAddress = result_list_h[i][:result_list_h[i].rfind('_') - 1]
    #     timeAddress = result_list_h[i][result_list_h[i].find('[') +
    #                                    1:result_list_h[i].find(']')]
    #     outPutName = topAddress + "u_" + timeAddress + 'tif'
    #     # print(outPutName)
    #     with rio.open(outPutName, 'w', **DATA_meta) as outf:
    #         outf.write(data_cpu, 1)

    #     internal_data_v = torch.where(internal_data_h > 1.0e-8,
    #                                   internal_data_qy / internal_data_h,
    #                                   internal_data_qy - internal_data_qy)
    #     print("max and min V value")
    #     max_value, max_index = torch.max(internal_data_v, 0)
    #     min_value, min_index = torch.min(internal_data_v, 0)
    #     print("Max V: ", max_value.item(), "\th: ",
    #           internal_data_h[max_index].item(),
    #           "\tqy: ", internal_data_qy[max_index].item(), "\tindex: ",
    #           max_index.item())
    #     print("Min V: ", min_value.item(), "\th: ",
    #           internal_data_h[min_index].item(),
    #           "\tqy: ", internal_data_qy[min_index].item(), "\tindex: ",
    #           min_index.item())

    #     # CheckIndex = 3917617
    #     # print(CheckIndex, " value")
    #     # print("h: ", internal_data_h[CheckIndex].item(), "\tqy: ", internal_data_qy[CheckIndex].item(), "\tqx: ", internal_data_qx[CheckIndex].item())
    #     dem[~mask] = internal_data_v.to(torch.float32)
    #     # print(dem)
    #     data_cpu = dem.cpu().numpy()
    #     data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
    #     nodatavalue = -9999.
    #     data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
    #     DATA_meta = demMeta.copy()
    #     DATA_meta.update({'nodata': nodatavalue})
    #     DATA_meta.update({'dtype': np.float32})

    #     print("2D index of U")
    #     print(
    #         demMeta['transform'][2] +
    #         (np.argwhere(data_cpu == max_value.item())[0, 0]) *
    #         demMeta['transform'][0], demMeta['transform'][5] -
    #         (np.argwhere(data_cpu == max_value.item())[0, 1]) *
    #         demMeta['transform'][0])

    #     print(
    #         "z value: ", z[np.argwhere(data_cpu == max_value.item())[0, 0],
    #                        np.argwhere(data_cpu == max_value.item())[0, 1]])
    #     print(
    #         "z value 1: ",
    #         z[np.argwhere(data_cpu == max_value.item())[0, 0] - 1,
    #           np.argwhere(data_cpu == max_value.item())[0, 1]])
    #     print(
    #         "z value 2: ",
    #         z[np.argwhere(data_cpu == max_value.item())[0, 0] + 1,
    #           np.argwhere(data_cpu == max_value.item())[0, 1]])
    #     print(
    #         "z value 3: ",
    #         z[np.argwhere(data_cpu == max_value.item())[0, 0],
    #           np.argwhere(data_cpu == max_value.item())[0, 1] - 1])
    #     print(
    #         "z value 4: ",
    #         z[np.argwhere(data_cpu == max_value.item())[0, 0],
    #           np.argwhere(data_cpu == max_value.item())[0, 1] + 1])

    #     print(
    #         demMeta['transform'][2] +
    #         (np.argwhere(data_cpu == min_value.item())[0, 0]) *
    #         demMeta['transform'][0], demMeta['transform'][5] -
    #         (np.argwhere(data_cpu == min_value.item())[0, 1]) *
    #         demMeta['transform'][0])

    #     print(
    #         "z value: ", z[np.argwhere(data_cpu == min_value.item())[0, 0],
    #                        np.argwhere(data_cpu == min_value.item())[0, 1]])

    #     print(
    #         "z value 1: ",
    #         z[np.argwhere(data_cpu == min_value.item())[0, 0] - 1,
    #           np.argwhere(data_cpu == min_value.item())[0, 1]])
    #     print(
    #         "z value 2: ",
    #         z[np.argwhere(data_cpu == min_value.item())[0, 0] + 1,
    #           np.argwhere(data_cpu == min_value.item())[0, 1]])
    #     print(
    #         "z value 3: ",
    #         z[np.argwhere(data_cpu == min_value.item())[0, 0],
    #           np.argwhere(data_cpu == min_value.item())[0, 1] - 1])
    #     print(
    #         "z value 4: ",
    #         z[np.argwhere(data_cpu == min_value.item())[0, 0],
    #           np.argwhere(data_cpu == min_value.item())[0, 1] + 1])

    #     outPutName = topAddress + "v_" + timeAddress + 'tif'
    #     # print(outPutName)
    #     with rio.open(outPutName, 'w', **DATA_meta) as outf:
    #         outf.write(data_cpu, 1)

    # dem = dem.to(torch.double)
    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[~mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = demMeta.copy()
        DATA_meta.update({'nodata': nodatavalue})
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        # print(outPutName)
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(data_cpu, 1)


def exportTiff(dem, mask, outPutPath):
    from glob import glob
    print(outPutPath + '/*.pt')
    result_list = glob(outPutPath + '/*.pt')
    mask = mask > 0
    mask_cpu = mask.cpu().numpy()
    dem = dem.to(torch.float32)
    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = {}
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        # print(outPutName)
        # with rio.open(outPutName, 'w', **DATA_meta) as outf:
        #     outf.write(data_cpu, 1)
        with rio.open(outPutName,
                      'w',
                      driver='GTiff',
                      width=1620,
                      height=1000,
                      count=1,
                      dtype=np.float32,
                      nodata=nodatavalue) as dst:
            dst.write(data_cpu, indexes=1)


def ave_loadStep(z, size):
    step = np.zeros(size - 1, dtype=np.int32)
    z_cpu = z.cpu().numpy()
    count = np.count_nonzero(z_cpu)
    internal_index = np.argwhere(z_cpu > 0)
    for i in range(size - 1):
        step[i] = internal_index[
            int(float(i + 1) / float(size) * float(count)), 0] - 1
    return step


def multi_exportRaster_tiff(DEM_path, outPutPath, GPU_num):

    from glob import glob
    result_list = glob(outPutPath + '/*cuda:0*.pt')

    # for g_id in range(GPU_num):

    # device = torch.device("cuda",
    #                       int(result_list[0][result_list[0].rfind(':') + 1]))
    device = torch.device("cuda", 0)
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)
    # dem = dem.to(torch.double)

    z = dem.clone()

    step = ave_loadStep(mask, GPU_num)

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()

    # dem = dem.to(torch.double)
    for i in range(len(result_list)):
        for j in range(GPU_num):
            file_path = result_list[i][:result_list[i].rfind(':') + 1] + str(
                j) + result_list[i][result_list[i].rfind(':') + 2:]
            internal_data = torch.load(file_path).to(device)
            if j == 0:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[step[0] + 2:, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[:step[0], :] = temp_dem[:step[0], :]
                del temp_mask, temp_dem
            elif j == GPU_num - 1:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[:step[-1] - 2, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[step[-1]:, :] = temp_dem[step[-1]:, :]
                del temp_mask, temp_dem
            else:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[:step[j - 1] - 2, :] = False
                temp_mask[step[j] + 2:, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[step[j - 1]:step[j], :] = temp_dem[step[j - 1]:step[j], :]
                del temp_mask, temp_dem

        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = demMeta.copy()
        DATA_meta.update({'nodata': nodatavalue})
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        print(outPutName)
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(data_cpu, 1)


def cutSectPlot(DEM_path, outPutPath, secIndex):
    result_list = glob.glob(outPutPath + '/*.pt')
    device = torch.device("cuda", 3)
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)
    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()
    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[~mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        sec_data = data_cpu[secIndex]


case = 2

if __name__ == "__main__":
    if case == 0:
        exportRaster_tiff('/home/lunet/cvhq/Carlisle_case/Carlisle_data/DEM_0.25_resample.tif',
                          '/home/lunet/cvhq/Carlisle_case/output_single')
    elif case == 1:
        from glob import glob
        result_list = glob(
            '/home/cvjz3/steadyFlow/jh_1st/5/output/h_tensor*.pt')
        # result_list += glob('/home/cvjz3/steadyFlow/output' +
        #                         '/wl_tensor*.pt')
        result_list += glob(
            '/home/cvjz3/steadyFlow/jh_1st/5/output/qx_tensor*.pt')
        device = torch.device("cuda", 0)

        x = range(240)
        for i in range(len(result_list)):
            internal_data = torch.load(result_list[i])
            internal_data = internal_data.resize(8, 240)  # print(dem)
            sec_data = internal_data[0]
            sec_data = sec_data.cpu().numpy()
            fig, ax = plt.subplots()
            plt.plot(x, sec_data)
            plt.ylim((0., 0.02))
            plt.savefig(result_list[i][:-2] + 'png')
    else:
        # multi_exportRaster_tiff(
        #     '/home/lunet/cvhq/Carlisle_case/Carlisle_data/DEM_2048.tif',
        #     '/home/lunet/cvhq/Carlisle_case/output_single', 1)
        # multi_exportRaster_tiff(
        #     '/home/cvjz3/Eden/Tiff_Data/Tiff/DEM.tif',
        #     '/home/cvjz3/Eden/output', 2)
        exportRaster_tiff('/home/lunet/cvhq/Carlisle_case/Carlisle_data/DEM_0.25_resample.tif',
                          '/home/lunet/cvhq/Carlisle_case/output_single')
