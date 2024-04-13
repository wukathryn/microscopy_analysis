from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import numpy.ma as ma
import tifffile as tif
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from skimage.draw import polygon2mask
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

import src.d00_utils.dirnames as dn
import src.d00_utils.utilities as utils


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--img_dirpath', required=True, help="directory containing the dataset")

# Extract ROI masks from ome-tif metadata
def extract_ROIs(metadata, img_shape):
    roi_labels = []
    rois = []

    for roi_data in metadata.rois:

        # Extract roi label
        roi_labels.append(roi_data.union[0].text)

        # Extract polygon coordinates
        coords = roi_data.union[0].points.split(' ')
        for i, coord in enumerate(coords):
            coord = coord.split(',')
            coords[i] = [int(coord[1]), int(coord[0])]

        # Convert polygon coordinates into a mask
        roi = polygon2mask(img_shape, coords)

        rois.append(roi.astype(int))

    return rois, roi_labels

# Mask basked on cell+ regions within the ROI
def mask_img(img, roi, min_size=100000):
    num_ch = img.shape[1]
    # index of the cell membrane channel (2nd to last channel in the stack)
    cell_ch = (num_ch - 2)
    roi = roi.astype(bool)
    cell = img[:, cell_ch, :, :, :].squeeze().astype(bool)
    cellmask = (roi & cell)
    cellmask = remove_small_objects(cellmask, min_size=min_size, connectivity=1)
    cellmask = binary_fill_holes(cellmask)
    cellmask_exp = np.broadcast_to(cellmask, img.shape)
    img_masked = ma.array(img, mask=~cellmask_exp)
    return img_masked, cellmask


def extract_int_and_area(img_masked):
    size_t, size_c = img_masked.shape[:2]

    timepoint = np.repeat(np.arange(size_t), size_c)
    channel = np.tile(np.arange(size_c), size_t)
    area_pixels = np.count_nonzero(img_masked, axis=(3,4)).flatten()
    mean_int = ma.mean(img_masked, axis=(3,4)).flatten()
    std_int = ma.std(img_masked, axis=(3,4)).flatten()
    median_int = ma.median(img_masked, axis=(3,4)).flatten()

    indiv_ch_data = {'timepoint': timepoint, 'channel': channel,
                     'area_pixels': area_pixels, 'mean_int': mean_int,
                     'std_int': std_int, 'median_int': median_int}

    return indiv_ch_data

def compile_info(overall_ch_data, indiv_ch_data, imgname, roi_idx, roi_label, pixelarea):
    if not overall_ch_data:
        overall_ch_data = {'image name':[], 'roi idx':[], 'roi label':[],
                           'timepoint':[], 'channel':[],
                           'area_pixels':[], 'area_microns':[],
                           'mean_int':[],
                           'std_int':[], 'median_int':[]
                           }

    size_t = len(indiv_ch_data['area_pixels'])
    overall_ch_data['image name'].extend([imgname] * size_t)
    overall_ch_data['roi idx'].extend([roi_idx] * size_t)
    overall_ch_data['roi label'].extend([roi_label] * size_t)
    overall_ch_data['timepoint'].extend(indiv_ch_data['timepoint'])
    overall_ch_data['channel'].extend(indiv_ch_data['channel'])
    overall_ch_data['area_pixels'].extend(indiv_ch_data['area_pixels'])
    overall_ch_data['area_microns'].extend(indiv_ch_data['area_pixels']*pixelarea)
    overall_ch_data['mean_int'].extend(indiv_ch_data['mean_int'])
    overall_ch_data['std_int'].extend(indiv_ch_data['std_int'])
    overall_ch_data['median_int'].extend(indiv_ch_data['median_int'])

    return overall_ch_data

def batch_exclusion_analysis(img_dirpath):

    # Create directories to store cell masks and analyses
    analysis_dir = img_dirpath / dn.analyses_dirname
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cellmasks_dir = img_dirpath / dn.cellmask_dirname
    cellmasks_dir.mkdir(parents=True, exist_ok=True)

    # Initiate dataframe to store channel data
    overall_ch_data = {}

    imgpaths = [imgpath for imgpath in img_dirpath.glob('*.ome.tif')]

    for imgpath in imgpaths:
        img_file = AICSImage(imgpath, reader=OmeTiffReader)
        pixelarea = utils.get_pixel_area(img_file.physical_pixel_sizes)

        rois, roi_labels = extract_ROIs(metadata=img_file.metadata, img_shape=img_file.shape[-2:])

        print(f'{len(rois)} ROIs found for {imgpath.name}')
        if len(rois) > 0:
            for i, roi in enumerate(rois):
                img = img_file.data
                size_c = img.shape[1]
                img_masked, cellmask = mask_img(img, roi)
                tif.imwrite(cellmasks_dir / f'{imgpath.name}_roi{i}.tif', cellmask)
                indiv_ch_data = extract_int_and_area(img_masked)
                overall_ch_data = compile_info(overall_ch_data,
                                               indiv_ch_data,
                                               imgname=imgpath.name,
                                               roi_idx=i,
                                               roi_label=roi_labels[i],
                                               pixelarea=pixelarea)

    overall_ch_df = pd.DataFrame(overall_ch_data)
    overall_ch_df = reorganize_df(overall_ch_df).reset_index()

    overall_ch_df.to_csv(analysis_dir / f'ch_analysis.csv', index=False)

def reorganize_df(df):
    # Convert channel info from rows into a column format
    df['channel'] = 'ch' + df['channel'].astype(str)
    num_ch = len(df.channel.unique())
    df = df.pivot_table(index=('image name', 'roi idx', 'timepoint'),
                        columns='channel',
                        values=('area_pixels', 'area_microns',
                                'mean_int', 'std_int', 'median_int'))

    # Extract cell area and % exclusion
    cell_seg_ch = 'ch' + str(num_ch - 2)
    caaxneg_seg_ch = 'ch' + str(num_ch - 1)
    cell_area = df['area_microns'][cell_seg_ch]
    perc_exclusion = df['area_microns'][caaxneg_seg_ch] / cell_area
    df.columns = df.columns.to_flat_index().str.join('_')
    df['cell area (microns)'] = cell_area
    df['% exclusion'] = perc_exclusion

    return df



if __name__ == '__main__':

    args = parser.parse_args()
    img_dirpath = Path(args.img_dirpath)
    print(img_dirpath)

    batch_exclusion_analysis(img_dirpath)
