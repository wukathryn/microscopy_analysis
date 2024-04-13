from pathlib import Path
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import pandas as pd
from src.d00_utils.dirnames import bg_sub_dirname, bg_sub_fig_dirname
from src.d00_utils import utilities
from . import vis_and_rescale
from datetime import datetime


def crop_black_borders(img):
    '''
    Crops out any black edges left over from timestiching or alignment processing
    that can throw off the otsu thresholding

    Parameters:
        img (np array): 5-D numpy array to be cropped

    Returns: cropped 5D numpy array
    '''
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


def clip_upper_outliers(img, outlier_percentiles):
    size_c = img.shape[1]
    outlier_thresh = []
    for ch in range(size_c):
        outlier_thresh.append(np.percentile(img[:, ch, :, :, :], q=outlier_percentiles[ch], method='higher', keepdims=False))
    outlier_thresh = np.expand_dims(outlier_thresh, axis=(0, 2, 3, 4))
    img_outliers_clipped = np.clip(img, a_min=None, a_max=outlier_thresh)
    return img_outliers_clipped.astype(img.dtype)


def get_otsu_thresholds(img):

    # flatten 2D images into 1D vectors
    [size_t, size_c, size_z, size_x, size_y] = img.shape
    img.reshape(size_t, size_c, size_z, -1)

    otsu_thresholds = np.zeros([size_t, size_c, size_z])

    for t in range(size_t):
        for ch in range(size_c):
            for z in range(size_z):
                otsu_thresholds[t, ch, z] = threshold_otsu(img[t, ch, z, :], nbins=65536)

    return otsu_thresholds

# Subtract threshold from image

def subtract_background(img, otsu_thresholds):
    otsu_thresholds = np.expand_dims(otsu_thresholds, axis=(3, 4))
    img_bg_subtract = (img.astype('float') - otsu_thresholds).astype('float')
    img_bg_subtract[img_bg_subtract < 0] = 0
    return img_bg_subtract.astype('uint16')


# def rescale_to_fit_dtype_range(img, dtype):
#     scaling_factor = np.iinfo(dtype).max / np.amax(img, axis=(0, 2, 3, 4), keepdims=True)
#     img_rescaled = (img * scaling_factor).astype(dtype)
#     return img_rescaled


def subtract_bg(imgpath, params_list, bs_thresh_d, ch_to_process=None, index=None,
                bgsub_time=None, target_perc_grayval=30, savefreq=5):

    '''
    Estimates background intensity via OTSU thresholding and subtracts it from the image

    Parameters:
        imgpath (str): full path for image
        params_list (dict): dictionary containing parameters for smoothing and clipping

    Returns:
    '''

    # Read image and metadata
    imgpath = Path(imgpath)

    # Make a folder for bg subtracted images if they don't exist yet
    proc_dirpath = utilities.get_proc_dirpath(imgpath)
    ch_str = ''.join(str(ch) for ch in ch_to_process)
    bg_sub_dirpath = proc_dirpath / bg_sub_dirname / 'initial' / (imgpath.parent.name + '_bgsbch' + ch_str + '_init')
    bg_sub_fig_dirpath = bg_sub_dirpath / bg_sub_fig_dirname
    bg_sub_fig_dirpath.mkdir(parents=True, exist_ok=True)

    # Open image
    img_file = AICSImage(imgpath, reader=OmeTiffReader)
    orig_img = img_file.data
    ome_metadata = img_file.ome_metadata
    basename = imgpath.name.split('.')[0]

    # Check processing parameters
    size_t, size_c = orig_img.shape[:2]
    params_list = check_params(params_list, len(ch_to_process))

    # Extract channels to be processed
    img_chsubset = None
    for ch in ch_to_process:
        if img_chsubset is None:
            img_chsubset = orig_img[:, ch, np.newaxis, :, :, :]
        else:
            img_chsubset = np.concatenate((img_chsubset, orig_img[:, ch, np.newaxis, :, :, :]), axis=1)

    #img_chsubset, _ = vis_and_rescale.rescale_img(img_chsubset, target_perc_grayval=target_perc_grayval, conv_to_8bit=False)

    # Preprocess image for OTSU thresholding by cropping black borders
    img_cropped = crop_black_borders(img_chsubset)


    # Apply processing parameters
    for p, params in enumerate(params_list):
        outlier_percentiles = params['outlier_percentiles']
        sigmas_smoothing = params['sigmas_smoothing']

        # Continue to reprocess image for OTSU thresholding via clipping and smoothing
        img_preprocessed = clip_upper_outliers(img_cropped, outlier_percentiles)
        for ch in range(img_preprocessed.shape[1]):
            img_preprocessed[:, ch, :, :, :] = gaussian_filter(img_preprocessed[:, ch, :, :, :], (0, 0, sigmas_smoothing[ch], sigmas_smoothing[ch]))

        # Use OTSU to obtain background thresholds from pre-processed image
        otsu_thresholds = get_otsu_thresholds(img_preprocessed)

        # Smooth the original image and subtract the background thresholds
        bg_sb_chsubset = img_chsubset.copy()
        for ch in range(bg_sb_chsubset.shape[0]):
            bg_sb_chsubset[:, ch, :, :, :] = gaussian_filter(bg_sb_chsubset[:, ch, :, :, :], (0, 0, sigmas_smoothing[ch], sigmas_smoothing[ch]))
        bg_sb_chsubset = subtract_background(bg_sb_chsubset, otsu_thresholds)

        # Save background subtracted image
        bg_sb_img = orig_img.copy()
        for i, ch in enumerate(ch_to_process):
            bg_sb_img[:, ch, :, :, :] = bg_sb_chsubset[:, i, :, :, :]
        bgsub_imgname = f'{basename}_{bgsub_time}_p{p}'
        bs_ome_metadata = utilities.construct_ome_metadata(bg_sb_img, img_file.physical_pixel_sizes)
        OmeTiffWriter.save(bg_sb_img, bg_sub_dirpath / (bgsub_imgname + '.ome.tif'), ome_xml=bs_ome_metadata)

        fig = vis_and_rescale.generate_subtractbg_fig(img_chsubset, bg_sb_chsubset, bgsub_imgname, params,
                                                      target_perc_grayval=target_perc_grayval,
                                                      index=index)
        fig.savefig(bg_sub_fig_dirpath / (bgsub_imgname + '.png'))

        # Update dictionary with params and thresholds
        bs_thresh_d['Image name'].extend([imgpath.name] * size_t)
        bs_thresh_d['Index'].extend([index] * size_t)

        if bgsub_time == None:
            bgsub_time = datetime.now().strftime("%Y%m%d-%I%M%p")
        bs_thresh_d['Bgsub_time'].extend([bgsub_time] * size_t)
        bs_thresh_d['Parameter set'].extend([p] * size_t)
        for i, ch in enumerate(ch_to_process):
            bs_thresh_d[f'outlier perc ch{ch}'].extend([outlier_percentiles[i] * size_t])
            bs_thresh_d[f'sigma smoothing ch{ch}'].extend([sigmas_smoothing[i] * size_t])
            thresh_ch = list(np.atleast_1d(otsu_thresholds[:, i, 0].squeeze()))
            bs_thresh_d[f'thresh ch{ch}'].extend(thresh_ch)

    return bs_thresh_d

def get_params_df(bs_thresh_df, params_keys):
    bs_params_df = bs_thresh_df.groupby('Image name', sort=False).first().reset_index()[params_keys]
    bs_params_df['Keep? Y/N'] = ''
    bs_params_df['Redo? Y/N'] = ''
    bs_params_df['Comments'] = ''
    return bs_params_df


def batch_subtract_bg(input_dirpath, params_list, ch_to_process, subset=None, target_perc_grayval=30, savefreq=5):
    '''
    Performs background subtraction on each image within a given directory

    Parameters:
        input_dirpath (str): full path for the image-containing directory
        params_list (list or dict): list of dictionaries or single dictionary containing parameters for smoothing and clipping
        ch_to_process (list): list of the channels to subtract background from
        subset (list): list containing the indices of images to be processed
        target_perc_grayval (int): target range out of all possible grayvalues represented within the image (used for scaling)
        savefreq (int): determines the number of images processed before updating the output spreadsheet

    Returns:
    '''
    # Get input dirpath, create a directory for bg subtracted images
    input_dirpath = Path(input_dirpath)
    assert input_dirpath.exists(), "Please input a valid directory path"

    proc_dirpath = utilities.get_proc_dirpath(input_dirpath)
    ch_str = ''.join(str(ch) for ch in ch_to_process)
    bg_sub_dirpath = proc_dirpath / bg_sub_dirname / 'initial' / (input_dirpath.name + '_bgsbch' + ch_str + '_init')
    bg_sub_fig_dirpath = bg_sub_dirpath / bg_sub_fig_dirname
    bg_sub_fig_dirpath.mkdir(parents=True, exist_ok=True)

    # Get the list of ome-tiff images to process
    imgpaths = [path for path in input_dirpath.glob('*.ome.tif')]
    imgpaths.sort()
    if subset is None:
        subset = list(np.arange(len(imgpaths)))
    else:
        imgpaths = [imgpaths[i] for i in subset]
    num_imgs = len(imgpaths)

    # Check whether there's an existing list of parameters
    bs_params_path = Path(bg_sub_dirpath) / f'subtractbg_params.csv'
    if bs_params_path.is_file():
        bs_params_df_prev = pd.read_csv(bs_params_path)
    else:
        bs_params_df_prev = pd.DataFrame()

    bs_thresh_path = Path(bg_sub_dirpath) / f'subtractbg_thresholds.csv'

    if bs_thresh_path.is_file():
        bs_thresh_df_prev = pd.read_csv(bs_thresh_path)
    else:
        bs_thresh_df_prev = pd.DataFrame()

    # Create a dictionary to temporarily store threshold data
    bs_thresh_d = {'Image name': [], 'Index': [], 'Bgsub_time': [], 'Parameter set': []}
    params_keys = ['Image name', 'Index', 'Bgsub_time', 'Parameter set']

    bgsub_time = datetime.now().strftime("%Y%m%d-%p%I%M")

    ch_to_process = list(ch_to_process)
    for ch in ch_to_process:
        bs_thresh_d[f'outlier perc ch{ch}'] = []
        params_keys.append(f'outlier perc ch{ch}')
        bs_thresh_d[f'sigma smoothing ch{ch}'] = []
        params_keys.append(f'sigma smoothing ch{ch}')
        bs_thresh_d[f'thresh ch{ch}'] = []

    # Loop through and process each image
    for i, imgpath in enumerate(imgpaths):

        imgname = imgpath.name

        index = subset[i]

        print(f'Processing {imgname}: file {i+1}/{num_imgs}')
        bs_thresh_d = subtract_bg(imgpath, params_list, bs_thresh_d, ch_to_process, index=index, bgsub_time=bgsub_time,
                                  target_perc_grayval=target_perc_grayval, savefreq=savefreq)

        if i % savefreq == 0:
            bs_thresh_df = pd.DataFrame(bs_thresh_d)
            bs_thresh_df = pd.concat([bs_thresh_df_prev, bs_thresh_df], ignore_index=True)
            bs_thresh_df.to_csv(bs_thresh_path, index=False)

    bs_thresh_df = pd.DataFrame(bs_thresh_d)
    bs_params_df = get_params_df(bs_thresh_df, params_keys)
    bs_params_df = pd.concat([bs_params_df_prev, bs_params_df], ignore_index=True)
    bs_params_df.to_csv(bs_params_path, index=False)

    bs_thresh_df = pd.concat([bs_thresh_df_prev, bs_thresh_df], ignore_index=True)
    bs_thresh_df.to_csv(bs_thresh_path, index=False)

    print(f'Finished processing directory!')


def check_params(params_list, size_c):
    # Convert all params_lists to list format
    if not isinstance(params_list, list):
        params_list = [params_list]

    # Check listed params within the params list
    for params in params_list:
        for key, value in params.items():
            # if the params are not in list-format, convert it to a list
            if isinstance(value, (int, float, complex)):
                value = [value]
            assert isinstance(value, list), \
                f'Please include a number or a list of numbers for the {key} parameter'
            if len(value) == 1:
                value = value * size_c
            assert len(value) == size_c, \
                f'For {key}, please list either 1 value to apply to all channels' \
                f' or 1 value for each channel ({size_c} total)'
            params[key] = value
    return params_list
