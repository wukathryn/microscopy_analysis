import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

def open_cellregions(imgpath):
    img_file = AICSImage(imgpath, reader=OmeTiffReader)
    img = img_file.data
    # Include only binary thresholded cell region images
    cellregions = img[:, 2:, :, :, :]
    return cellregions


def add_areas_df(cellregions, pixel_area, imgdata, cellreg_ch_d):
    num_timepoints = cellregions.shape[0]

    # Extracts the cell regions channel order
    cellreg_ch_d_keys = cellreg_ch_d.keys()

    imgdata = pd.concat([imgdata]*num_timepoints, ignore_index=True)

    # Calculate areas / timepoint from the cellregions image stack
    regionareas = np.count_nonzero(cellregions, axis=(3, 4)) * pixel_area

    for i, key in enumerate(cellreg_ch_d_keys):
        key = key.replace('region', 'area')
        imgdata[key] = regionareas[:, i]

    imgdata_df = imgdata
    print(imgdata_df)

    return imgdata_df


def compute_region_areas(cellregions, imgdata, cellreg_ch_d, pixel_area):
    print("Computing region metrics...")

    # Parameters
    pixel_area = pixel_area
    imgdata_df = add_areas_df(cellregions, pixel_area, imgdata, cellreg_ch_d)

    return imgdata_df

