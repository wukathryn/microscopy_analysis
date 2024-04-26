from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from skimage.exposure import match_histograms
import src.d00_utils.utilities as utils

# Might not be useful
def match_histograms_dir(img_dirpath, ref_imgpath):
    ref_img = AICSImage(ref_imgpath, reader=OmeTiffReader).data

    img_dirpath = Path(img_dirpath)
    imgpaths = [p for p in img_dirpath.glob('*.ome.tif')]
    print(f'{len(imgpaths)} images found in {img_dirpath.name}.')

    if len(imgpaths) > 0:
        output_dirpath = img_dirpath.parent / (img_dirpath.name + '_histomatched')
        output_dirpath.mkdir(parents=True, exist_ok=True)

    for p in imgpaths:
        img_file = AICSImage(p, reader=OmeTiffReader)
        img = img_file.data
        matched_img = match_histograms(img, ref_img)
        ome_metadata = utils.construct_ome_metadata(matched_img, img_file.physical_pixel_sizes)
        OmeTiffWriter.save(matched_img, (output_dirpath / p.name), ome_xml=ome_metadata)
