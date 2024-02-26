import pandas as pd
import numpy as np
from pathlib import Path
from aicsimageio.writers import OmeTiffWriter
from src.d00_utils.dirnames import proc_dirname

# Variables to help parse img file names
exp_search = 'CE'
div_search = 'div'
scene_search = 'sc'
roi_search = 'ROI'
tx_search = 'tx'

def extract_img_info(imgname):
    basename = imgname.split('.')[0]
    name_elems = np.array(basename.split('_'))

    # Extracts info from the image filename
    exp = search_name(name_elems, exp_search)
    div = search_name(name_elems, div_search)
    tx = search_name(name_elems, tx_search)
    scene = search_name(name_elems, scene_search)
    roi = search_name(name_elems, roi_search)

    # Stores info keys and values into lists
    info_df = pd.DataFrame({'Image Name': [imgname],
                            'Experiment': [exp],
                            'DIV': [div],
                            'Tx': [tx],
                            'Scene': [scene],
                            'ROI': [roi],
                            'UID': [scene + '_' + str(roi)]})
    # Only include columns with non-nan values
    info_df.dropna(axis=1)
    return info_df


def search_name(name_elems, searchphrase):
    search_res = np.char.find(name_elems, searchphrase)
    if np.any(search_res!=-1):
        info = name_elems[np.where(search_res != -1)][0]
    else:
        info = np.nan
    return info


def construct_ome_metadata(new_img, physical_pixel_sizes, channel_names=None):
    ome_metadata = OmeTiffWriter.build_ome(data_shapes=[new_img.data.shape], data_types=[np.dtype(new_img.dtype)],
                                           physical_pixel_sizes=[physical_pixel_sizes], channel_names=[channel_names])

    return ome_metadata


def get_pixel_area(physical_pixel_sizes):
    if physical_pixel_sizes.X is not None and physical_pixel_sizes.X is not None:
        pixel_area = physical_pixel_sizes.Y * physical_pixel_sizes.X
    else:
        pixel_area = None
    return pixel_area

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

def generate_df_with_imgpaths(input_dirpath):
    input_dirpath = Path(input_dirpath)
    imgnames = [path.name for path in input_dirpath.glob('*.ome.tif')]
    scenes = []
    for imgname in imgnames:
        info_df = extract_img_info(imgname)
        scenes.append(info_df['Scene'].values[0])
    df = pd.DataFrame({'Image name': imgnames, 'Scene': scenes})
    return df


def get_proc_dirpath(input_dirpath):
    dirname = input_dirpath.name
    if dirname == proc_dirname:
        return input_dirpath
    else:
        return get_proc_dirpath(input_dirpath.parent)


def new_or_overwrite_ok(filepath):
    if filepath.exists():
        overwrite_yn = input(f'\'{filepath}\' already exists. Overwrite? Enter Y/N\n')
        if overwrite_yn.lower() == 'n':
            return False
    return True
