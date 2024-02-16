import os, sys
import tifffile as tif
import numpy as np
from collections import OrderedDict
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join('../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import src.d00_utils.utilities as utils

from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, binary_dilation
from skimage.segmentation import watershed
# from skimage.morphology import area_closing, square, binary_closing
from skimage.color import label2rgb
from skimage.feature import peak_local_max

def process_file(imgpath):
    img_file = AICSImage(imgpath.path, reader=OmeTiffReader)
    img = img_file.data
    binary_img = (img > 1).astype('uint8')
def process_folder(dirpath):
    imgpaths = [file for file in os.scandir(dirpath) if (os.path.splitext(file)[1]=='.tif')]
    for imgpath in imgpaths:
        process_file(imgpath)
def main():
    exp_dir = '../../data/CE006'
    vis_dirpath = os.path.join(exp_dir, '004_visualization')
    process_folder(vis_dirpath)

if __name__ == '__main__':
    main()