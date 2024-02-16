import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from matplotlib import pyplot as plt

import sys
src_path = str(Path.cwd().parent)
if src_path not in sys.path:
    sys.path.append(src_path)
import src.d00_utils.dirnames as dn
import src.d00_utils.utilities as utils

tp = -1
z = 0

def apply_mask_files(input_dirpath, mask_subdirname, add_cellmask=False, cell_ch=None):
    '''
    Masks ome-tif images using polygon ROIs and saves these into a new directory

    Parameters:
        input_dirpath (str): string path for folder containing images to be masked
        mask_dirname (str): name of masks directory
        add_cellmask (bool) : if True, will also mask image based on the cell channel (cell_ch variable)
        cell_ch (int) : int representing the channel corresponding to the cell (e.g. membrane dye)

    Returns: None
    '''

    input_dirpath = Path(input_dirpath)

    # Gets directory paths for input masks
    proc_dir = utils.get_proc_dirpath(input_dirpath)
    masks_dirpath = proc_dir / dn.masks_dirname / mask_subdirname

    if add_cellmask:
        assert cell_ch is not None, "Please input the channel corresponding to the full cell"
        cellmasks_dirpath = proc_dir / dn.masks_dirname / dn.cellmask_dirname
        cellmasks_dirpath.mkdir(parents=True, exist_ok=True)
        maskedimgs_subdirname = f'{input_dirpath.name}_masked_{dn.cellmask_dirname}'
    else:
        maskedimgs_subdirname = f'{input_dirpath.name}_masked_{mask_subdirname}'

    masked_imgs_dirpath = proc_dir / dn.masked_imgs_dirname / maskedimgs_subdirname
    masked_imgs_dirpath.mkdir(parents=True, exist_ok=True)


    imgpaths = [path for path in Path(input_dirpath).glob('*.ome.tif')]
    print(f'Applying masks to {len(imgpaths)} images')

    for imgpath in imgpaths:

        # Identifies masks that share the same name as the image
        basename = imgpath.name.split('.ome.tif')[0]
        maskpaths = [path for path in Path(masks_dirpath).glob(f'*{basename}*')]
        print(f"{len(maskpaths)} mask file(s) found for {basename}")

        if len(maskpaths) > 0:
            img_data = AICSImage(imgpath, reader=OmeTiffReader)
            img = img_data.data

            for maskpath in maskpaths:

                mask_file = AICSImage(maskpath)
                mask = mask_file.data.astype(bool)

                img_masked = apply_mask(img, mask)

                if add_cellmask:
                    assert cell_ch is not None, "Please input the channel corresponding to the full cell"
                    cellmask = get_cellmask(img_masked, cell_ch)
                    img_masked = apply_mask(img_masked, cellmask)

                    # Save cell mask
                    cellmask = cellmask.squeeze().astype('uint8') * np.iinfo('uint8').max
                    AICSImage(cellmask).save(cellmasks_dirpath / maskpath.name)

                ome_metadata = utils.construct_ome_metadata(img_masked, img_data.physical_pixel_sizes)
                OmeTiffWriter.save(img_masked, masked_imgs_dirpath / (maskpath.name.split('.tif')[0] + '.ome.tif'),
                                   ome_xml=ome_metadata)
    print('Done!')

def get_cellmask(img, cell_ch, min_size=100000):
    init_cellmask = img[:, cell_ch, np.newaxis, :, :, :]
    init_cellmask = (init_cellmask > 0).astype('bool')
    cellmask = remove_small_objects(init_cellmask, min_size=min_size, connectivity=1)
    cellmask = binary_fill_holes(cellmask)
    return cellmask

def apply_mask(img, mask):
    # Expand mask to the same shape as the image
    size_t, size_c, size_z, _, _ = img.shape
    # mask = np.repeat(mask, size_t, axis=0)
    # mask = np.repeat(mask, size_c, axis=1)
    # mask = np.repeat(mask, size_z, axis=2)

    # Apply mask
    img_masked = img * mask
    return img_masked


def subtract_nuclei_from_cellmasks(nuclei_dirpath):
    nuclei_dirpath = Path(nuclei_dirpath)

    # Get directory path for cell masks
    masks_dirpath = nuclei_dirpath.parent
    cellmask_dirpath = masks_dirpath / dn.cellmask_dirname

    # Check that both nuclei and cell mask directories exist
    assert nuclei_dirpath.exists()
    assert cellmask_dirpath.exists()

    # Create an output path to save cell masks without nuclei
    output_dirpath = masks_dirpath / dn.cellmask_nonuc_dirname
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Iterate through all tif files in the cell masks directory
    cellmaskpaths = [path for path in cellmask_dirpath.glob('*.tif')]
    for cellmaskpath in cellmaskpaths:

        # Find the matching nucleus mask path
        nucleipaths = [path for path in nuclei_dirpath.glob(f'{cellmaskpath.name}*')]
        print(f'{len(nucleipaths)} nuclei found for {cellmaskpath.name}')

        for nucleipath in nucleipaths:
            cellmask_file = AICSImage(cellmaskpath)
            cellmask = cellmask_file.data.squeeze().astype('bool')

            nuclei_file = AICSImage(nucleipath)
            nuclei = nuclei_file.data.squeeze().astype('bool')

            # Create a cell mask without the nucleus
            cellmask_nonuc = cellmask * ~nuclei

            # Rescale and save the cell mask without the nucleus
            cellmask_nonuc = cellmask_nonuc.astype('uint8') * np.iinfo('uint8').max
            AICSImage(cellmask_nonuc).save(output_dirpath / nucleipath.name)