import tifffile as tif
import numpy as np
from collections import OrderedDict
from skimage.morphology import remove_small_objects
import src.d00_utils.utilities as utils
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

def define_regions(img, channels_d):

    binary_img = (img > 0)
    min_obj_size = 10000
    binary_img = remove_small_objects(binary_img, min_obj_size).astype(int)

    [size_t, _, size_z, size_x, size_y] = binary_img.shape
    caax_ch = channels_d['CAAX protein']
    mem_ch = channels_d['cell membrane']

    # Pixels representing cell regions
    cell_reg = binary_img[:, mem_ch, :, :, :]

    # Only include CAAX+ pixels within the cell
    caax_reg = binary_img[:, caax_ch, :, :, :] * cell_reg

    # Pixels representing CAAX protein-negative regions
    excl_reg = np.zeros_like(cell_reg)
    excl_reg[cell_reg > caax_reg] = 1

    # Pixels where previously CAAX-positive regions became CAAX-excluded
    new_excl = np.zeros_like(cell_reg)
    new_excl[1:, :, :, :][(caax_reg[:-1, :, :, :] > caax_reg[1:, :, :, :])] = 1
    new_excl[cell_reg == 0] = 0

    # Pixels where previously CAAX-excluded regions revert to being CAAX-positive
    new_rev = np.zeros_like(cell_reg)
    new_rev[1:, :, :, :][(excl_reg[:-1, :, :, :] > excl_reg[1:, :, :, :])] = 1
    new_rev[cell_reg == 0] = 0

    # Creates a dictionary to record the cell regions channels
    cellreg_ch_d = OrderedDict()

    cellreg_ch_d_keys = ['CAAX-positive region',
                         'cell region',
                         'CAAX-excluded region',
                         'new exclusion region',
                         'new reverted region']

    cellreg_ch_d_values = [caax_reg,
                           cell_reg,
                           excl_reg,
                           new_excl,
                           new_rev]

    # Creates an empty image stack to store defined cell regions
    cellregions = np.zeros((size_t, len(cellreg_ch_d_keys), size_z, size_x, size_y))

    for i, key in enumerate(cellreg_ch_d_keys):
        cellreg_ch_d[key] = i
        cellregions[:, cellreg_ch_d[key], :, :, :] = cellreg_ch_d_values[i]

    # Set binary masks to the max value
    cellregions = (cellregions * np.iinfo(img.dtype).max).astype('uint8')

    return cellregions, cellreg_ch_d

