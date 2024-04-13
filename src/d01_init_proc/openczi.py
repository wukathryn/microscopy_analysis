import sys
from pathlib import Path
import argparse
src_path = str(Path.cwd().parent)
if src_path not in sys.path:
    sys.path.append(src_path)
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import xml.etree.ElementTree as ET
import xmltodict
from ome_types.model import Plane
from ome_types.model.simple_types import UnitsTime
from src.d00_utils import dirnames as dn

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--exp_dir', required=True, help="directory containing the dataset")


def get_ome_metadata(img):
    ome_metadata = OmeTiffWriter.build_ome([img.shape], [np.dtype(img.dtype)], channel_names=[img.channel_names],
                                           physical_pixel_sizes=[img.physical_pixel_sizes])

    # if img.dims.T > 1:
    #     ome_metadata = update_ome_timestamps(img.metadata, ome_metadata)
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
    n = 0 # cellID counter if there is no scene data

    for i, scene in enumerate(img.scenes):
        if len(img.scenes) > 1:
            if scene is None:
                scene = f'N{n}'
                n = n + 1
            imgsavename = f'{Path(imgpath).name.split(".")[0]}_sc{scene}.ome.tif'
            img.set_scene(scene)
        else:
            imgsavename = f'{Path(imgpath).name.split(".")[0]}.ome.tif'

        img_savepath = Path(raw_tif_dirpath) / imgsavename

        ome_metadata = get_ome_metadata(img)
        OmeTiffWriter.save(img.data, img_savepath, ome_xml=ome_metadata)

def move_files_into_CZI_dir(exp_dir, proc_dir):
    CZI_dirpath = Path(proc_dir) / dn.CZI_dirname
    CZI_dirpath.mkdir(parents=True, exist_ok=True)

    for imgpath in Path(exp_dir).glob('*.czi'):
        imgpath.rename(CZI_dirpath / imgpath.name)

    return CZI_dirpath

def batch_convert_czi_to_tif(exp_dir):
    exp_dir = Path(exp_dir)
    proc_dir = exp_dir / dn.proc_dirname
    CZI_dirpath = move_files_into_CZI_dir(exp_dir, proc_dir)

    imgpaths = [path for path in CZI_dirpath.glob('*.czi')]
    imgpaths.sort()
    imgpaths_total = len(imgpaths)

    raw_ometif_dirpath = proc_dir / dn.raw_ometif_dirname
    raw_ometif_dirpath.mkdir(parents=True, exist_ok=True)

    num_imgs = len(imgpaths)

    if len(imgpaths)==0:
        print('No CZI images found')
    else:
        for i, imgpath in enumerate(imgpaths):
            print(f'Converting {imgpath.name} to ome-tiff (file {i + 1}/{num_imgs})')
            convert_czi_to_tif(imgpath, raw_ometif_dirpath)
        print(f'Done! Ome-tiff files saved to {raw_ometif_dirpath}')


if __name__ == '__main__':

    args = parser.parse_args()
    exp_dir = args.exp_dir

    batch_convert_czi_to_tif(exp_dir)