import os
import sys
parent_dir = os.path.abspath(os.path.join('..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import src.d00_utils.utilities as utils
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import xml.etree.ElementTree as ET
import xmltodict
from ome_types.model import Plane
from ome_types.model.simple_types import UnitsTime


def get_ome_metadata(img):
    ome_metadata = OmeTiffWriter.build_ome([img.data.shape], [np.dtype('uint16')], channel_names=[img.channel_names],
                                           physical_pixel_sizes=[img.physical_pixel_sizes])

    if img.dims.T > 1:
        ome_metadata = update_ome_timestamps(img.metadata, ome_metadata)

    return ome_metadata


def update_ome_timestamps(metadata, ome_metadata):
    xmlstr = ET.tostring(metadata)
    metadatadict_czi = xmltodict.parse(xmlstr)
    t_increment = float(
        metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['T']['Positions'][
            'Interval']['Increment'])

    size_c = ome_metadata.images[0].pixels.size_c
    size_t = ome_metadata.images[0].pixels.size_t
    for c in range(size_c):
        for t in range(size_t):
            plane = Plane(the_c=c, the_t=t, the_z=0, delta_t=t * t_increment,
                          delta_t_unilt=UnitsTime.SECOND)
            ome_metadata.images[0].pixels.planes.append(plane)
    return ome_metadata


def convert_czi_to_tif(imgpath, raw_tif_dirpath):
    img = AICSImage(imgpath)
    for i, scene in enumerate(img.scenes):
        if len(img.scenes) > 1:
            imgsavename = f'{os.path.splitext(os.path.basename(imgpath))[0]}_{scene}.ome.tif'
            img.set_scene(scene)
        else:
            imgsavename = f'{os.path.splitext(os.path.basename(imgpath))[0]}.ome.tif'
        img_savepath = os.path.join(raw_tif_dirpath, imgsavename)

        ome_metadata = get_ome_metadata(img)

        OmeTiffWriter.save(img.data, img_savepath, ome_xml=ome_metadata)


def batch_convert_czi_to_tif(exp_dir):
    raw_czi_dirpath = os.path.join(exp_dir, '001_raw_czi_files')
    raw_ometif_dirpath = utils.getsavedirpath(exp_dir, '002_raw_ometif_files')

    imgpaths = [file for file in os.scandir(raw_czi_dirpath) if (os.path.splitext(file.name)[1] == '.czi')]
    num_imgs = len(imgpaths)

    for i, imgpath in enumerate(imgpaths):
        print(f'Converting {imgpath.name} to ome-tiff (file {i + 1}/{num_imgs})')
        convert_czi_to_tif(imgpath.path, raw_ometif_dirpath)
    print(f'Done! Files saved to {raw_ometif_dirpath}')


def main():
    exp_dir = '../../data/maya_imgs'
    batch_convert_czi_to_tif(exp_dir)


if __name__ == '__main__':
    main()