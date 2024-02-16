# Imports
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
import numpy as np
import pandas as pd

from src.d00_utils.dirnames import excl_analysis_dirname
from src.d00_utils import utilities as utils

from matplotlib import pyplot as plt

# Variables
binary_regions_dirname = "binary_regions"
perc_area_dirname = "perc_area"
overlap_dirname = "overlap"

# t and z dimension indices
tp = -1
z = 0

num_digits = 4

def add_exclusion_channel(binary_regions, cell_ch, caax_ch):
    size_c = binary_regions.shape[1]
    caax_ch = check_and_convert_to_list(caax_ch, size_c)
    for ch in caax_ch:
        exclusion_region = binary_regions[tp, cell_ch, z, :, :] & ~binary_regions[tp, caax_live_ch, z, :, :]
        exclusion_region = np.expand_dims(exclusion_region, axis=(0, 1, 2))
        binary_regions = np.concatenate([binary_regions, exclusion_region], axis=1)
    return binary_regions

def check_and_convert_to_list(chs, size_c):
    if chs=='all':
        chs = np.arange(size_c)
    else:
        assert isinstance(chs, (int, list, tuple))
        if isinstance(chs, int):
            chs = [chs]
    return chs


def calc_perc_areas(denom_ch, numerator_ch, binary_regions, saveinfo, idv_perc_area_d):
    (perc_area_dirpath, imgname, ch_labels, ch_save_abbr, physical_pixel_sizes) = saveinfo

    # Mask image by the channel used as the denominator
    denom_ch_mask = np.expand_dims(binary_regions[tp, denom_ch, z, :, :], axis=(0, 1, 2))
    binary_regions_chmask = binary_regions * denom_ch_mask

    # Save channel-masked binary regions
    binary_chmask_dirpath = perc_area_dirpath / f'{binary_regions_dirname}_{ch_save_abbr[denom_ch]}-mask'
    binary_chmask_dirpath.mkdir(parents=True, exist_ok=True)
    ome_metadata = utils.construct_ome_metadata(binary_regions_chmask, physical_pixel_sizes, ch_labels)
    OmeTiffWriter.save(binary_regions_chmask, binary_chmask_dirpath / imgpath.name, ome_xml=ome_metadata)

    # Calculate areas for each channel
    ch_areas = np.count_nonzero(binary_regions_chmask, axis=(3, 4)).squeeze()
    size_c = binary_regions_chmask.shape[1]
    for ch in range(size_c):
        idv_perc_area_d.update({f'{ch_labels[ch]} within {ch_labels[denom_ch]} area (pixels)': ch_areas[ch]})
        # TODO: fix this section
        # if physical_pixel_size is not None:
        #     perc_areas_d.update({f'{ch_labels[ch]} within {ch_labels[denom_ch]} area (um)': ch_areas[ch] * pixel_size})

    numerator_ch = check_and_convert_to_list(numerator_ch, size_c)

    if ch_labels is None:
        ch_labels = {denom_ch: f'ch {denom_ch}'}
        for n_ch in numerator_ch:
            ch_labels.update({n_ch: f'ch {n_ch}'})

    for n_ch in numerator_ch:
        if n_ch != denom_ch:

            # Create directory for this specific analysis
            analysis_dirpath = perc_area_dirpath / (f'perc_areas_{ch_save_abbr[n_ch]}_{ch_save_abbr[denom_ch]}')
            analysis_dirpath.mkdir(parents=True, exist_ok=True)

            # Calculations
            perc_area = round((ch_areas[n_ch] / ch_areas[denom_ch]), num_digits)
            idv_perc_area_d[f'{ch_labels[n_ch]} area / {ch_labels[denom_ch]} area'] = perc_area

            # Create matplotlib figure
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            plt.rcParams.update({'font.size': 7})
            axs[0].imshow(binary_regions_chmask[tp, denom_ch, z, :, :], cmap='gray', interpolation=None)
            axs[0].set_title(f'{ch_labels[denom_ch]}')
            axs[1].imshow(binary_regions_chmask[tp, n_ch, z, :, :], cmap='gray', interpolation=None)
            axs[1].set_title(f'{ch_labels[n_ch]}\nwithin {ch_labels[denom_ch]}')
            axs[2].imshow(binary_regions_chmask[tp, denom_ch, z, :, :], cmap='Greys', interpolation=None)
            axs[2].imshow(binary_regions_chmask[tp, n_ch, z, :, :], cmap='Purples', alpha=0.8, interpolation=None)
            axs[2].set_title(
                f'gray: {ch_labels[denom_ch]}\npurple: {ch_labels[n_ch]} overlap\n{round(perc_area * 100, num_digits)}%')

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            base_imgname = imgname.split('.ome.tif')[0]
            fig.suptitle(imgname)

            # Save matplotlib figure
            fig.savefig(analysis_dirpath / f'{base_imgname}.png')

    return idv_perc_area_d


# Analysis functions

def calc_overlap(ch_1st, ch_2nd, binary_regions, saveinfo, idv_overlap_d):
    (overlap_dirpath, imgname, ch_labels, ch_save_abbr, physical_pixel_sizes) = saveinfo

    binary_ch_1st = binary_regions[-1:, ch_1st, 0:1, :, :]
    binary_ch_2nd = binary_regions[-1:, ch_2nd, 0:1, :, :]

    intersect_img = binary_ch_1st & binary_ch_2nd
    union_img = binary_ch_1st | binary_ch_2nd
    binary_ch_1st_only = binary_ch_1st & ~binary_ch_2nd
    binary_ch_2nd_only = binary_ch_2nd & ~binary_ch_1st

    binaryoverlap = np.stack([intersect_img, union_img, binary_ch_1st_only, binary_ch_2nd_only], axis=1)

    binaryoverlap_dirpath = overlap_dirpath / f'binary_{ch_save_abbr[ch_1st]}_{ch_save_abbr[ch_2nd]}'
    binaryoverlap_dirpath.mkdir(parents=True, exist_ok=True)
    ome_metadata = utils.construct_ome_metadata(binaryoverlap, physical_pixel_sizes, ch_labels)
    OmeTiffWriter.save(binaryoverlap, binaryoverlap_dirpath / imgpath.name, ome_xml=ome_metadata)

    areas = np.count_nonzero(binaryoverlap, axis=(0, 2, 3, 4))

    # area of intersection / area of union
    overlap = round(areas[0] / areas[1], 4)
    idv_overlap_d.update({f'{ch_labels[ch_1st]}_{ch_labels[ch_2nd]}_overlap': overlap})

    # visualize overlap
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(binary_ch_1st.squeeze(), cmap='gray', interpolation=None)
    axs[0].set_title(f'{ch_labels[ch_1st]}')
    axs[1].imshow(binary_ch_2nd.squeeze(), cmap='gray', interpolation=None)
    axs[1].set_title(f'{ch_labels[ch_2nd]}')
    axs[2].imshow(union_img.squeeze(), cmap='Greys', interpolation=None)
    axs[2].imshow(intersect_img.squeeze(), cmap='Blues', alpha=0.5, interpolation=None)
    axs[2].set_title(f'gray: union, \nblue: overlap\n{round(overlap * 100, 4)}%')
    axs[3].imshow(binary_ch_1st_only.squeeze(), cmap='Greens', interpolation=None)
    axs[3].imshow(binary_ch_2nd_only.squeeze(), cmap='Purples', alpha=0.5, interpolation=None)
    axs[3].set_title(f'green: {ch_labels[ch_1st]} only, \npurple: {ch_labels[ch_2nd]} only')
    for ax in axs:
        ax.axis('off')
    plt.rcParams.update({'font.size': 7})
    plt.tight_layout()
    plt.show()

    base_imgname = imgname.split('.ome.tif')[0]
    fig.suptitle(imgname)

    # Save matplotlib figure
    fig_dirpath = overlap_dirpath / f'fig_{ch_save_abbr[ch_1st]}_{ch_save_abbr[ch_2nd]}'
    fig_dirpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dirpath / f'{base_imgname}.png')

    return idv_overlap_d