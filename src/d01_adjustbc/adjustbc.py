import os
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import pandas as pd
import src.d00_utils.utilities as utils


def crop_black_borders(img):
    # get x and y sizes for image
    x_ch, y_ch = (3, 4)
    x_len, y_len = img.shape[x_ch:y_ch + 1]

    # find image corners
    nonzeros = (img != 0)
    xmin = np.max(nonzeros.argmax(axis=x_ch))
    xmax = x_len - np.max(nonzeros[:, :, :, ::-1, :].argmax(axis=x_ch))
    ymin = np.max(nonzeros.argmax(axis=y_ch))
    ymax = y_len - np.max(nonzeros[:, :, :, :, ::-1].argmax(axis=y_ch))

    cropped_img = img[:, :, :, xmin:xmax, ymin:ymax]
    return cropped_img


def clip_max(img, maxclip_percentile):
    [size_t, size_c, size_z, size_x, size_y] = img.shape
    img_flat = img.reshape(size_t, size_c, size_z, size_x * size_y)
    outlier_upperthresh = np.percentile(img_flat, q=maxclip_percentile, axis=3).reshape(size_t, size_c, size_z, 1)
    img_maxclipped = np.clip(img_flat, a_min=None, a_max=outlier_upperthresh)
    img_maxclipped = img_maxclipped.reshape(size_t, size_c, size_z, size_x, size_y)
    return img_maxclipped


def get_otsu_thresholds(img):
    [size_t, size_c, size_z, _, _] = img.shape

    otsu_thresholds = np.zeros([size_t, size_c, size_z])

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                otsu_thresholds[t, c, z] = threshold_otsu(img[t, c, z, :, :], nbins=65536)

    return otsu_thresholds


def subtract_background(img,  otsu_thresholds):
    size_t, size_c, size_z, size_x, size_y = img.shape
    img_flat = img.reshape(size_t, size_c, size_z, size_x * size_y)
    img_bg_subtract = img_flat - np.expand_dims(otsu_thresholds, 3)
    img_bg_subtract = img_bg_subtract.reshape((size_t, size_c, size_z, size_x, size_y))
    img_bg_subtract[img_bg_subtract <= 0] = 0

    return img_bg_subtract


def rescale_to_8bit_range(img):
    size_c = img.shape[1]
    for c in range(size_c):
        max_val = np.maximum(np.max(img[:, c, :, :, :]), 1)
        img[:, c, :, :, :] = img[:, c, :, :, :] * (1/max_val) * 255
    return img


def create_img_dict(imgname, ome_metadata, otsu_thresholds, params):
    size_t, size_c, size_z = otsu_thresholds.shape

    img_processing_var_d = pd.DataFrame()
    timepoint_array = np.arange(size_t)
    img_processing_var_d['Timepoint'] = timepoint_array

    for c in range(size_c):
        for z in range(size_z):
            z_col_label = ''
            if size_z > 1:
                z_col_label = f'_zslice{z}'
            img_processing_var_d[f'Otsu_thresholds_ch{c}{z_col_label}'] = otsu_thresholds[:, c, z].squeeze()

    img_processing_var_d.insert(0, 'Image name', imgname)

    for i, param_key in enumerate(params.keys()):
        img_processing_var_d.insert(i+1, param_key, params[param_key])

    return img_processing_var_d


def reconstruct_ome_metadata(img, channel_names, physical_pixel_sizes, planes):
    ome_metadata = OmeTiffWriter.build_ome([img.shape], [np.dtype(img.dtype)],
                                           channel_names=channel_names,
                                           physical_pixel_sizes=physical_pixel_sizes)
    ome_metadata.images[0].pixels.planes = planes
    return ome_metadata


def process_img(imgpath, bg_sub_dirpath, vis_dirpath, params):

    # Get parameters
    otsu_maxclip_percentile = params['otsu_maxclip_percentile']
    vis_maxclip_percentile = params['vis_maxclip_percentile']
    smooth = params['smooth']
    sigma_smoothing = params['sigma_smoothing']

    # Read image
    img_file = AICSImage(imgpath.path, reader=OmeTiffReader)
    img = img_file.data
    ome_metadata = img_file.ome_metadata

    # Create an adjusted temporary image to optimize otsu thresholding
    temp_img = crop_black_borders(img)
    temp_img = clip_max(temp_img, otsu_maxclip_percentile)
    otsu_thresholds = get_otsu_thresholds(temp_img)

    img_bg_subtract = img

    if smooth is True:
        img_bg_subtract = gaussian_filter(img_bg_subtract, (sigma_smoothing,)*img.ndim)

    img_bg_subtract = subtract_background(img_bg_subtract, otsu_thresholds).astype('uint16')

    channel_names = [img_file.channel_names]
    physical_pixel_sizes = [img_file.physical_pixel_sizes]
    planes = img_file.ome_metadata.images[0].pixels.planes

    bg_subtract_ome_metadata = reconstruct_ome_metadata(img_bg_subtract, channel_names, physical_pixel_sizes, planes)
    OmeTiffWriter.save(img_bg_subtract, os.path.join(bg_sub_dirpath, imgpath.name), ome_xml=bg_subtract_ome_metadata)

    img_vis = clip_max(img_bg_subtract, vis_maxclip_percentile)
    img_vis = rescale_to_8bit_range(img_vis).astype('uint8')
    img_vis_ome_metadata = reconstruct_ome_metadata(img_vis, channel_names, physical_pixel_sizes, planes)
    OmeTiffWriter.save(img_vis, os.path.join(vis_dirpath, imgpath.name), ome_xml=img_vis_ome_metadata)

    img_processing_var_d = create_img_dict(imgpath.name, ome_metadata, otsu_thresholds, params)

    return img_processing_var_d


def process_folder(exp_dir, params):
    raw_ometif_dirpath = os.path.join(exp_dir, '002_raw_ometif_files')
    imgpaths = [file for file in os.scandir(raw_ometif_dirpath) if (os.path.splitext(file)[1] == '.tif')]
    num_imgs = len(imgpaths)
    bg_sub_dirpath = utils.getsavedirpath(exp_dir, '003_bg_subtracted')
    vis_dirpath = utils.getsavedirpath(exp_dir, '004_visualization')
    tables_dirpath = utils.getsavedirpath(exp_dir, 'tables')
    processing_var_df = pd.DataFrame()
    processing_params_path = os.path.join(tables_dirpath, 'processing_params.csv')

    for i, imgpath in enumerate(imgpaths):
        imgname = imgpath.name
        print(f'Processing {imgname} (file {i+1}/{num_imgs})')
        img_processing_var_d = process_img(imgpath, bg_sub_dirpath, vis_dirpath, params)

        processing_var_df = pd.concat([processing_var_df, img_processing_var_d], ignore_index=True)
        if i % 5 == 0:
            processing_var_df.to_csv(processing_params_path)

    processing_var_df.to_csv(processing_params_path)
    print(f'Done! Processing parameters saved to {processing_params_path}')


def check_params(params):
    if params['smooth'] is True:
        assert isinstance(params['sigma_smoothing'], int), 'Please include an integer for sigma smoothing'
    else:
        params['sigma_smoothing'] = None


def main():
    # Parameters to set before processing files
    params = {}
    params['otsu_maxclip_percentile'] = 60
    params['vis_maxclip_percentile'] = 99
    params['smooth'] = False
    params['sigma_smoothing'] = 1

    check_params(params)

    exp_dir = '../../data/CE006'
    process_folder(exp_dir, params)


if __name__ == '__main__':
    main()
