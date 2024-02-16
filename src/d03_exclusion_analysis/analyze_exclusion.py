import os
from pathlib import Path
import pandas as pd

import src.d00_utils.utilities as utils
from . import definecellregions
from . import compute_region_areas
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from src.d00_utils.dirnames import excl_analysis_dirname
from src.d00_utils.utilities import get_pixel_area

def process_timelapse(imgpath, channels_d, excl_analysis_dirpath):
    img_file = AICSImage(imgpath, reader=OmeTiffReader)
    img = img_file.data
    physical_pixel_sizes = img_file.physical_pixel_sizes
    cellregions, cellreg_ch_d = definecellregions.define_regions(img, channels_d)
    ome_metadata = utils.construct_ome_metadata(cellregions, img_file.physical_pixel_sizes)
    OmeTiffWriter.save(cellregions, Path(excl_analysis_dirpath) / imgpath.name, ome_xml=ome_metadata)
    pixel_area = get_pixel_area(physical_pixel_sizes)
    return cellregions, cellreg_ch_d, pixel_area


def batch_exclusion_analysis(input_dir, channels_d):

    # Extract existing directory paths
    input_dir = Path(input_dir)

    # Create save paths
    excl_analysis_dirpath = input_dir.parent / excl_analysis_dirname
    excl_analysis_dirpath.mkdir(parents=True, exist_ok=True)

    overall_df_path = Path(excl_analysis_dirpath) / 'overall_dataframe.csv'

    if os.path.isfile(overall_df_path):
        overall_df = pd.read_csv(overall_df_path)
    else:
        overall_df = pd.DataFrame()

    imgpaths = [path for path in Path(input_dir).glob('*.ome.tif')]
    num_imgs = len(imgpaths)

    for i, imgpath in enumerate(imgpaths):
        imgname = imgpath.name
        progress = f'file {i+1}/{num_imgs}'
        print(f'Processing {imgname}: {progress}')
        cellregions, cellreg_ch_d, pixel_area = process_timelapse(imgpath, channels_d, excl_analysis_dirpath)

        info_d = utils.extract_img_info(imgname)
        imgdata_df = compute_region_areas.compute_region_areas(cellregions, info_d, cellreg_ch_d, pixel_area)
        imgdata_df.insert(0, 'frame', (imgdata_df.index + 1))
        overall_df = pd.concat([overall_df, imgdata_df], ignore_index=True)

    overall_df.to_csv(Path(excl_analysis_dirpath) / 'region_areas_df.csv', index=False)

