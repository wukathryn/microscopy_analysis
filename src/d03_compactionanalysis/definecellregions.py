import tifffile as tif
from skimage.filters import gaussian, threshold_otsu
import numpy as np
from collections import OrderedDict

def open_images(imgpaths):
    imgs = []
    for imgpath in imgpaths:
        imgs.append(tif.imread(imgpath))
    return imgs

def mask_img(img, roi_mask):
    [num_timepoints, num_channels] = img.shape[:2]
    roi_mask_binary = roi_mask / np.max(roi_mask)
    roi_mask_binary_exp = np.array([np.array([roi_mask_binary]*num_channels)]*num_timepoints)
    mask_img = roi_mask_binary_exp * img
    return mask_img

def autothresh(img, sigmas_for_filter, channels_d):
    [timepoints, channels, x, y] = img.shape
    smoothed_img = np.zeros_like(img)
    for c in range(channels):
        smoothed_img[:, c, :, :] = gaussian(img[:, c, :, :], sigma=sigmas_for_filter[c], preserve_range=True)

    thresholds = np.zeros([timepoints, channels, 1])

    for t in range(timepoints):
        for c in range(channels):
            thresh = threshold_otsu(smoothed_img[t, c, :, :])
            thresholds[t, c, :] = thresh

    flattened_img_shape = (timepoints, channels, x * y)
    smoothed_img = smoothed_img.reshape(flattened_img_shape)
    img_thresh = np.zeros(flattened_img_shape)
    img_thresh[smoothed_img > thresh] = 1
    img_thresh = img_thresh.reshape((timepoints, channels, x, y))

    caax_ch = channels_d['CAAX protein']
    mem_ch = channels_d['cell membrane']

    return img_thresh, thresholds

def define_regions(img, channels_d):

    img_thresh = img.astype(bool)

    [num_timepoints, num_channels, x, y] = img_thresh.shape
    caax_ch = channels_d['CAAX protein']
    mem_ch = channels_d['cell membrane']

    # Number of additional channels that will be added to the cell regions image stack
    cell_reg = img_thresh[:, mem_ch, :, :]

    # Only include CAAX+ pixels inside the cell
    caax_reg = img_thresh[:, caax_ch, :, :] * cell_reg

    # Pixels representing CAAX protein-negative regions
    mem_noCaax = np.zeros_like(cell_reg)
    mem_noCaax[cell_reg > caax_reg] = 1

    # Pixels where CAAX protein-negative regions) are gained
    mem_noCaax_gained = np.zeros_like(cell_reg)
    (mem_noCaax_gained[1:, :, :])[(mem_noCaax[1:, :, :] > mem_noCaax[:-1, :, :])] = 1

    # Pixels where CAAX protein-negative regions) are lost
    mem_noCaax_lost = np.zeros_like(cell_reg)
    (mem_noCaax_lost[1:, :, :])[(mem_noCaax[1:, :, :] < mem_noCaax[:-1, :, :])] = 1

    new_caax_reg = np.zeros_like(cell_reg)
    (new_caax_reg[1:, :, :])[(caax_reg[1:, :, :] > caax_reg[:-1, :, :])] = 1

    new_hole_mem_noCaax_reg = mem_noCaax_lost * (1 - cell_reg)

    # Creates a dictionary to record the cell regions channels
    cellreg_ch_d = OrderedDict()

    cellreg_ch_d_keys = ['CAAX protein-positive regions',
                         'cell regions',
                         'CAAX protein-negative cell regions',
                         'CAAX protein-negative cell regions gained',
                         'new CAAX protein-positive cell regions',
                         'regions of holes formed in previously CAAX protein-negative regions']

    cellreg_ch_d_values = [caax_reg,
                           cell_reg,
                           mem_noCaax,
                           mem_noCaax_gained,
                           new_caax_reg,
                           new_hole_mem_noCaax_reg]

    # Creates an empty image stack to store defined cell regions
    cellregions = np.zeros((num_timepoints, len(cellreg_ch_d_keys), x, y)).astype('uint8')

    for i, key in enumerate(cellreg_ch_d_keys):
        cellreg_ch_d[key] = i
        cellregions[:, cellreg_ch_d[key], :, :] = cellreg_ch_d_values[i]

    # Set binary masks to the max value
    cellregions = cellregions * 255

    return cellregions, cellreg_ch_d


def saveimgs(img, cellregions, withcellregionspath):
    # Combine the initial ROI-masked images with cell regions images
    imgs_withcellregions = (np.concatenate([img, cellregions], axis=1)).astype('uint8')
    tif.imwrite(withcellregionspath, imgs_withcellregions, imagej=True)


def definecellregions(imgpath, mask_path, withcellregions_path, channels_d):
    # Parameters
    print("Defining cell regions...")

    [img, roi_mask] = open_images([imgpath, mask_path])
    img = mask_img(img, roi_mask)
    img_thresh, thresholds = autothresh(img, sigmas_for_filter, channels_d)
    cellregions, cellreg_ch_d = define_regions(img_thresh, channels_d)
    saveimgs(img, cellregions, withcellregions_path)

    return thresholds, cellreg_ch_d