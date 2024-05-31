# # High-Performance Integrated hydrodynamic Modelling System ***hybrid***
# @author: Jiaheng Zhao (Hemlab)
# @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
# @contact: j.zhao@lboro.ac.uk
# @software: hipims_hybrid
# @time: 07.01.2021
# This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
# Feel free to use and extend if you are a ***member of hemlab***.
import os
import torch
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import colors
import seaborn as sns
import numpy as np
from shapely.geometry import mapping, box
import rasterio as rio
from rasterio.plot import plotting_extent
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio import mask
from rasterio.enums import Resampling
import numpy.ma as ma

# read the tiff and give the field values

device = torch.device('cuda', 1)

# Useful functions


def reverse_coordinates(pol):
    """
    Reverse the coordinates in pol
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    Returns [[y1,x1],[y2,x2],...,[yN,xN]]
    """
    return [list(f[-1::-1]) for f in pol]


def to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [[wind_.row_off, wind_.col_off],
            [wind_.row_off, wind_.col_off + wind_.width],
            [wind_.row_off + wind_.height, wind_.col_off + wind_.width],
            [wind_.row_off + wind_.height, wind_.col_off],
            [wind_.row_off, wind_.col_off]]


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],
            [bbox[0], bbox[3]], [bbox[0], bbox[1]]]


def pol_to_np(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    return np.array([list(l) for l in pol])


def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:, 0]), np.min(arr[:, 1]), np.max(arr[:, 0]),
                       np.max(arr[:, 1]))


def setTheMainDevice(main_device):
    device = main_device


def shpExtractVertices(shp_path):
    """This will read the shp file and return all the rectangular vertices in the file
    
    Arguments:
        shp_path {[str]} -- [the path for the shp file]
    
    Returns:
        [list] -- [will be a 3D array list]
    """
    df = gpd.read_file(shp_path)
    g = [i for i in df.geometry]

    all_coords = []
    for b in g[0].boundary:  # for first feature/row
        coords = np.dstack(b.coords.xy).tolist()
        all_coords.append(*coords)
    return all_coords


def degreeToMeter(degreeUnit):
    meter = degreeUnit * (2. * np.math.pi * 6371004.) / 360.
    return meter


def read_and_resampleRaster(rasterPath, demMeta, device):
    with rio.open(rasterPath) as dataset:
        # resample data to target shape
        upscale_factor = dataset.meta['transform'][0] / demMeta['transform'][0]
        data = dataset.read(1,
                            out_shape=(dataset.count,
                                       int(dataset.height * upscale_factor),
                                       int(dataset.width * upscale_factor)),
                            resampling=Resampling.bilinear)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-2]),
            (dataset.height / data.shape[-1]))

        rows_min, cols_min = rio.transform.rowcol(transform,
                                                  demMeta['transform'][2],
                                                  demMeta['transform'][5])

        resampleData = data[rows_min:rows_min + demMeta['width'],
                            cols_min:cols_min + demMeta['height']]
        resampleData = torch.from_numpy(resampleData).to(device=device)

        return resampleData


# ===========================================================================
# GPU version, will cause problems when big raster
# ===========================================================================
def importDEMData(DEM_path, device):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    mask = torch.from_numpy(mask).to(device=device)

    # dem_min = torch.zeros_like(dem)
    # dem_min += 10000.

    maskID = mask.to(torch.int32)
    # mask = ~mask

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])
    # maskMatrix = torch.
    mask_boundary = torch.zeros_like(mask, dtype=torch.int32, device=device)
    for i in range(4):
        # dem_min = torch.min(
        #     dem_min,
        #     dem.roll(int(oppo_direction[i][0]),
        #              int(oppo_direction[i][1])).abs())
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))

    # here we get the dem after considering fill sink
    # dem = torch.where(dem > -1000., torch.max(dem, dem_min), dem)

    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    return dem, mask, mask_boundary, demMeta


# enum BOUNDARY_TYPE = {WALL_NON_SLIP = 3, WALL_SLIP = 4, OPEN = 5, HQ_GIVEN =6};
def setBoundaryEdge(mask,
                    mask_boundary,
                    demMeta,
                    device,
                    boundBox=np.array([]),default_BC=50,
                    bc_type=6):
    # boundBox = [[x_min, y_min, x_max, y_max],[x_min, y_min, x_max, y_max]]
    # it will be great if you can get the box from gis tools
    mask = mask.to(torch.int32)
    bc_dict = {'WALL_NON_SLIP': 3, 'WALL_SLIP': 4, 'OPEN': 5, 'HQ_GIVEN': 6, 'FALL': 9}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = default_BC

    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for i in range(len(boundBox)):
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: WALL_NON_SLIP, WALL_SLIP, OPEN, HQ_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])
            mask_box[rows_min:rows_max, cols_min:cols_max] = True

            mask[mask_boundary & mask_box] = int(str(BC_TYPE) + str(i))
    return mask


# ===========================================================================
# CPU version, no limitation for raster size
# ===========================================================================
def importDEMData_And_BC(DEM_path,
                         device,
                         gauges_position=np.array([]),
                         boundBox=np.array([]), 
                         default_BC=50,
                         bc_type=5
                        ):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    mask = demMasked.mask

    dem = torch.from_numpy(dem).to(device=device)
    # mask = torch.from_numpy(mask).to(device=device)
    mask = torch.from_numpy(mask)
    maskID = mask.to(torch.int32)

    oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])

    mask_boundary = torch.zeros_like(mask, dtype=torch.int32)
    for i in range(4):
        mask_boundary = mask_boundary + maskID.roll(int(oppo_direction[i][0]),
                                                    int(oppo_direction[i][1]))

    mask_boundary[mask] = 0

    mask_boundary = mask_boundary.to(torch.bool)

    mask = ~mask

    mask = mask.to(torch.int32)

    gauge_index_1D = torch.tensor([])

    if gauges_position.size > 0:
        mask_gauge = mask.clone()  # here make a copy of mask values
        rows, cols = rio.transform.rowcol(demMeta['transform'],
                                          gauges_position[:, 0],
                                          gauges_position[:, 1])
        mask_gauge[rows, cols] = 100
        gauge_index_1D = torch.flatten(
            (mask_gauge[mask_gauge > 0] >= 99).nonzero()).type(torch.int64)

        # reorder the gauge_index
        # print(dem[rows, cols])
        # print(mask_gauge[rows, cols])

        # print(gauge_index_1D)

        rows = np.array(rows)
        cols = np.array(cols)
        array = rows * dem.size()[1] + cols

        order = array.argsort()
        ranks = order.argsort()
        # print(array)
        # print(order)

        gauge_index_1D = gauge_index_1D[ranks]

    bc_dict = {'RIGID': 3, 
               'WALL_SLIP': 4, 
               'OPEN': 5, 
               'H_GIVEN': 6, 
               "Q_GIVEN":7,
               'WL_GIVEN': 8,
               'FALL': 9}
    # set the default BC as HQ_GIVEN, and H and Q are 0.0

    mask[mask_boundary] = default_BC
    

    # as the index will start with 0
    bc_count = [-1, -1, -1, -1, -1, -1]
    
    bc_count[list(bc_dict.values()).index(int(str(default_BC)[0]))] += 1

    if boundBox.size > 0:
        mask_box = torch.zeros_like(mask, dtype=torch.bool)
        for i in range(len(boundBox)):
            mask_box[:] = 0
            if type(bc_type[i]) == str:
                try:
                    BC_TYPE = bc_dict[bc_type[i]]
                except KeyError:
                    print(
                        "The keys should be: RIGID, WALL_SLIP, OPEN, H_GIVEN, Q_GIVEN, WL_GIVEN"
                    )
            else:
                BC_TYPE = bc_type[i]
            # src.bounds----> BoundingBox(left=358485.0, bottom=4028985.0, right=590415.0, top=4265115.0)
            boundBox[i][0] = max(src.bounds[0], boundBox[i][0])
            boundBox[i][1] = max(src.bounds[1], boundBox[i][1])
            boundBox[i][2] = min(src.bounds[2], boundBox[i][2])
            boundBox[i][3] = min(src.bounds[3], boundBox[i][3])
            # print(src.bounds)
            # print(boundBox)

            rows_min, cols_min = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][0],
                                                      boundBox[i][1])
            rows_max, cols_max = rio.transform.rowcol(demMeta['transform'],
                                                      boundBox[i][2],
                                                      boundBox[i][3])

            # print(rows_min, cols_min, rows_max, cols_max)

            # mask_box[rows_min:rows_max, cols_min:cols_max] = True
            # The row index is from top to down, so the index is from top to down
            mask_box[rows_max:rows_min, cols_min:cols_max] = True

            bc_count[list(bc_dict.values()).index(BC_TYPE)] += 1

            mask[mask_boundary & mask_box] = int(
                str(BC_TYPE) +
                str(bc_count[list(bc_dict.values()).index(BC_TYPE)]))

    mask_GPU = mask.to(device=device)

    # make sure that the boundary cells are higher than the internal cells
    # dem[mask_GPU == 60] += 2.0
    # dem[mask_GPU == 61] += 2.0

    # print(mask[mask == 80])
    del mask
    torch.cuda.empty_cache()
    return dem, mask_GPU, demMeta, gauge_index_1D


# ===========================================================================
# read the landuse and return the parameters
# ===========================================================================
def importLanduseData(LandUse_path, device, level=0):
    with rio.open(LandUse_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    # landuse = np.ma.filled(demMasked, fill_value=-127)
    if level == 1:
        landuse = np.ma.filled(demMasked, fill_value=11)
    else:
        landuse = np.ma.filled(demMasked, fill_value=1)
    # landuse = demMashed
    mask = demMasked.mask

    if level == 1:
        landuse = (landuse / 10).astype(int) - 1
        landuse_index_class = len(np.unique(landuse)) - 1
        landuse = torch.from_numpy(landuse).to(device=device)
    else:
        landuse_index_class = len(np.unique(landuse)) - 1
        indexes = np.unique(landuse)
        landuse = torch.from_numpy(landuse).to(device=device)
        for i in range(1, len(indexes), 1):
            landuse[landuse == indexes[i]] = i - 1

    landuse_index = np.arange(landuse_index_class)

    return landuse, landuse_index


def importRainStationMask(rainmask_path, device):
    with rio.open(rainmask_path) as src:
        demMashed = src.read(1, masked=True)
        demMeta = src.meta
    rainmask = np.ma.filled(demMashed, fill_value=0)
    rainmask = rainmask.astype(np.int16)
    rainmask = torch.from_numpy(rainmask).to(device=device)
    mask = demMashed.mask
    return rainmask


def lidar_rainfall(lidarFolderPath):
    # time should be the time.tif
    lidarFiles = glob(lidarFolderPath + '*.tif')
    return lidarFiles


def voronoiDiagramGauge_rainfall_source(rainfall_matrix_path):
    rainSource = np.genfromtxt(rainfall_matrix_path)
    return rainSource


def importAscDEM(asc_DEM_path, DEM_path):
    DEM_Data = np.genfromtxt(asc_DEM_path, skip_header=6)
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    print(demMeta)

    mask = ~mask
    mask_cpu = mask.cpu().numpy()
    DEM_Data = ma.masked_array(DEM_Data, mask=mask_cpu)

    nodatavalue = -9999.
    DEM_Data = ma.filled(DEM_Data, fill_value=nodatavalue)

    DATA_meta = demMeta.copy()
    DATA_meta.update({'nodata': nodatavalue})
    topAddress = asc_DEM_path[:asc_DEM_path.rfind('.')]

    outPutName = topAddress + 'tif'
    # print(outPutName)
    with rio.open(outPutName, 'w', **DATA_meta) as outf:
        outf.write(data_cpu, 1)


def gaussian(x, y, x_mu, y_mu, sig):
    return np.exp(-(np.power(x - x_mu, 2.) + np.power(y - y_mu, 2.)) /
                  (2 * np.power(sig, 2.)))


def normalDistMask(DEM_path, device):
    with rio.open(DEM_path) as src:
        demMasked = src.read(1, masked=True)
        demMeta = src.meta
    cantho_mask = demMasked.mask

    rain_mask = cantho_mask.copy()
    cantho_mask = ~cantho_mask

    y = np.arange(cantho_mask.shape[0])
    x = np.arange(cantho_mask.shape[1])

    xv, yv = np.meshgrid(x, y)

    cantho_mask = cantho_mask * gaussian(
        xv, yv, cantho_mask.shape[1], cantho_mask.shape[0],
        (cantho_mask.shape[1] + cantho_mask.shape[0]) / 3.)

    cantho_mask = ma.masked_array(cantho_mask, mask=rain_mask)
    nodatavalue = -9999.
    cantho_mask = ma.filled(cantho_mask, fill_value=nodatavalue)
    DATA_meta = demMeta.copy()
    DATA_meta.update({'nodata': nodatavalue})
    DATA_meta.update({'dtype': np.float64})

    export = False
    if export:
        topAddress = DEM_path[:DEM_path.rfind('/')]
        outPutName = topAddress + "/rainMask.tif"
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(cantho_mask, 1)
    rain_mask_tensor = torch.as_tensor(cantho_mask[cantho_mask > 0.])
    rain_mask_tensor.to(device)
    print(rain_mask_tensor.size())
    return rain_mask_tensor


if __name__ == "__main__":
    # landuse, landindex = importLanduseData(
    #     '/home/cvjz3/Luanhe_case/Luan_Data_90m/Landuse.tif', 2)
    # print(landuse[landuse >= 0])
    # print(landindex)
    # importAscDEM('/home/cvjz3/Luanhe_case/dem.asc',
    #              '/home/cvjz3/Luanhe_case/Luan_Data_90m/DEM.tif')
    normalDistMask('/home/cvjz3/CanTho/dem.tif')