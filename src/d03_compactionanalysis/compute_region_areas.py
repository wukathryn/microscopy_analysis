import tifffile as tif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_from_dataframe.plot_from_dataframe
import os

def open_cellregions(imgpath):
    img = tif.imread(imgpath)
    # Include only binary thresholded cell region images
    cellregions = img[:, 2:, :, :]
    return cellregions


def add_areas_df(cellregions, microns_per_pixel, imgdata, cellreg_ch_d):
    num_timepoints = cellregions.shape[0]
    # Extracts the cell regions channel order
    cellreg_ch_d_keys = cellreg_ch_d.keys()

    # Creates a dictionary to store the calculated areas per timepoint
    #areas_keys = [key.replace('regions', 'area') for key in cellreg_ch_d_keys]

    # Calculate areas / timepoint from the cellregions image stack
    regionareas = np.count_nonzero(cellregions, axis=(2, 3)) * (microns_per_pixel ** 2)
    for i, key in enumerate(cellreg_ch_d_keys):
        key = key.replace('regions', 'area')
        imgdata[key] = regionareas[:, i]

    # calculate additional areas / timepoint
    imgdata['CAAX protein-negative area / total area'] = imgdata['CAAX protein-negative cell area'] / imgdata['cell area']

    # Convert dictionary to dataframe
    imgdata_df = pd.DataFrame(imgdata)

    return imgdata_df


def compute_region_areas(withcellregions_path, graphs_path, imgdata, cellreg_ch_d):
    print("Computing region metrics...")

    # Parameters
    # TODO: GET THIS FROM METADATA
    microns_per_pixel = 0.1135

    cellregions = open_cellregions(withcellregions_path)
    imgdata_df = add_areas_df(cellregions, microns_per_pixel, imgdata, cellreg_ch_d)


    return imgdata_df

