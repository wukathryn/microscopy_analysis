from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
import numpy as np
from numpy.fft import fft2, ifft2
import pandas as pd
from src.d00_utils.utilities import crop_black_borders, construct_ome_metadata, extract_img_info
from src.d01_init_proc import vis_and_rescale
from src.d00_utils.dirnames import proc_dirname, raw_ometif_dirname, ch_aligned_dirname
import argparse
import shutil
from datetime import datetime

# Global variables
aligned_dirname = 'aligned'
orig_unaligned_dirname = 'orig_unaligned'
alignment_csv_name = 'alignment_info.csv'
error_csv_name = 'alignment_errors.csv'

# User-input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--multiexp_dir', required=True, help="multi-experiment directory path")
parser.add_argument('-c1', '--img1_ch_align', required=True, help="channel to align")
parser.add_argument('-c2', '--img2_ch_align', required=True, help="channel to align")
parser.add_argument('-s', '--ch_subset', required=True, help="subset of channels to keep")

tp_align = -1
z_align = 0


def preprocess_for_alignment(img1, img2, img1_ch_align, img2_ch_align):
    # rescale images
    img1_proc, _ = vis_and_rescale.rescale_img(img1, target_perc_grayval=50, standardize_scaling=False,
                                                           scaling_fact=None, conv_to_8bit=True)
    img2_proc, _ = vis_and_rescale.rescale_img(img2, target_perc_grayval=50, standardize_scaling=False,
                                                           scaling_fact=None, conv_to_8bit=True)

    # Crop images to the same dimensions
    min_y_size = np.minimum(img1.shape[3], img2.shape[3])
    min_x_size = np.minimum(img1.shape[4], img2.shape[4])

    img1_proc = img1_proc[tp_align, img1_ch_align, z_align, :min_y_size, :min_x_size].squeeze()
    img2_proc = img2_proc[tp_align, img2_ch_align, z_align, :min_y_size, :min_x_size].squeeze()

    return img1_proc, img2_proc

# paper: http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
def get_translation_coords(img1, img2):
    # Calculate translation coordinates
    shape = img1.shape
    f1 = fft2(img1)
    f2 = fft2(img2)
    ir = abs(ifft2((f1 * f2.conjugate()) / (abs(f1) * abs(f2))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return t0, t1


def align_and_stack(img1, img2, t0, t1):
    [img1_size_y, img1_size_x] = img1.shape[3:]

    pad_y_top = 0
    pad_y_bottom = 0
    pad_x_top = 0
    pad_x_bottom = 0

    if t0 < 0:
        pad_y_bottom = np.abs(t0)
    else:
        pad_y_top = t0 * 2
    if t1 < 0:
        pad_x_bottom = np.abs(t1)
    else:
        pad_x_top = t1 * 2

    img2_padded = np.pad(img2, ((0, 0), (0, 0), (0, 0), (pad_y_top, pad_y_bottom), (pad_x_top, pad_x_bottom)))
    img2_align = img2_padded[:, :, :, np.abs(t0): (img1_size_y + np.abs(t0)), np.abs(t1):(img1_size_x + np.abs(t1))]

    aligned_stack = np.concatenate((img1[-1:, :, :, :, :], img2_align[-1:, :, :, :, :]), axis=1)
    aligned_stack = crop_black_borders(aligned_stack)

    return aligned_stack

## NOTE: doesn't work very well yet
def align_all_ch(imgpath, ch_aligned_dirpath=None):
    imgpath = Path(imgpath)

    # Create output directory if not given
    if ch_aligned_dirpath==None:
        ch_aligned = imgpath.parent / ch_aligned_dirname
        ch_aligned.mkdir(parents=True, exist_ok=True)

    # Open image
    img_file = AICSImage(imgpath, reader=OmeTiffReader)
    img = img_file.data

    size_c = img.shape[1]

    # Set img1 as the first channel
    img1 = img[:, 0, np.newaxis, :, :, :]

    # align and stack the remaining channels
    for img2_ch in range(1, size_c):

        # Set img2 as the next channel
        img2 = img[:, img2_ch, np.newaxis, :, :, :]

        img1_slice, img2_slice = preprocess_for_alignment(img1, img2, img1_ch_align=0, img2_ch_align=0)

        t0, t1 = get_translation_coords(img1_slice, img2_slice)

        # Stack img1 and img2 and save this stack as img1
        img1 = align_and_stack(img1, img2, t0, t1)

    ome_metadata = construct_ome_metadata(img1, img_file.physical_pixel_sizes)
    OmeTiffWriter.save(img1, ch_aligned_dirpath / imgpath.name, ome_xml=ome_metadata)

## NOTE: doesn't work very well yet
def batch_align_ch(input_dir):
    # Create output directory
    input_dir = Path(input_dir)
    ch_aligned_dirpath = input_dir.parent / ch_aligned_dirname
    ch_aligned_dirpath.mkdir(parents=True, exist_ok=True)

    imgpaths = [path for path in Path(input_dir).glob('*.ome.tif')]
    imgpaths.sort()
    num_imgs = len(imgpaths)
    for i, imgpath in enumerate(imgpaths):
        print(f'Aligning {i}/{num_imgs}')
        align_all_ch(imgpath, ch_aligned_dirpath)

    return ch_aligned_dirpath

def align_2imgs(imgpath1, imgpath2, aligned_dirpath, img1_ch_align=0, img2_ch_align=0,
                img1_ch_subset=None, img2_ch_subset=None):

    # Open both images and crop out black borders
    img_file1 = AICSImage(imgpath1, reader=OmeTiffReader)
    img1 = img_file1.data
    img1 = crop_black_borders(img1)

    img_file2 = AICSImage(imgpath2, reader=OmeTiffReader)
    img2 = img_file2.data
    img2 = crop_black_borders(img2)

    # Check if the channels in the channel subset will exist in the merged image
    img1_sizeC = img1.shape[1]
    img2_sizeC = img2.shape[1]
    if img1_ch_subset is not None:
        assert (np.array(img1_ch_subset) < img1_sizeC).all(), \
            'Image 1 is missing some of the channels within the requested subset'
    if img2_ch_subset is not None:
        assert (np.array(img2_ch_subset) < img2_sizeC).all(), \
            'Image 2 is missing some of the channels within the requested subset'

    img1_slice, img2_slice = preprocess_for_alignment(img1, img2, img1_ch_align, img2_ch_align)

    # Get translation coordinates for alignment
    [t0, t1] = get_translation_coords(img1_slice, img2_slice)

    aligned_stack = align_and_stack(img1, img2, t0, t1)
    ome_metadata = construct_ome_metadata(aligned_stack, img_file1.physical_pixel_sizes)

    aligned_pathname = f'{Path(imgpath1).name.split(".")[0]}_aligned.ome.tif'
    aligned_path = Path(aligned_dirpath) / aligned_pathname

    # Save aligned stack with all image channels
    OmeTiffWriter.save(aligned_stack, aligned_path, ome_xml=ome_metadata)

    align_df = pd.DataFrame({'img1': [Path(imgpath1).name], 'img2': [Path(imgpath2).name], 't0': [t0], 't1': [t1]})

    if img1_ch_subset is not None or img2_ch_subset is not None:
        # Create merged channel subset
        if img1_ch_subset is None:
            img1_ch_subset = np.arange(0, img1_sizeC)
        else:
            img1_ch_subset = np.array(img1_ch_subset)
        if img2_ch_subset is None:
            img2_ch_subset = np.arange(0, img2_sizeC)
        else:
            img2_ch_subset = np.array(img2_ch_subset)
        stack_img2_ch_subset = img2_ch_subset + img1_sizeC
        stack_ch_subset = np.concatenate([img1_ch_subset, stack_img2_ch_subset])

        chsubset, chsubset_path = create_ch_subset(aligned_stack, aligned_path, stack_ch_subset)
        subset_ome_metadata = construct_ome_metadata(chsubset, img_file1.physical_pixel_sizes)
        OmeTiffWriter.save(chsubset, chsubset_path, ome_xml=subset_ome_metadata)

    return align_df

def create_ch_subset(aligned_stack, aligned_path, ch_subset):
    chsubset_dirpath = aligned_path.parent.parent / (aligned_path.parent.name + '_chsubset')
    chsubset_dirpath.mkdir(parents=True, exist_ok=True)
    chsubset_path = chsubset_dirpath / aligned_path.name

    img_chs = [aligned_stack[:, ch, np.newaxis, :, :, :] for ch in ch_subset]
    chsubset = np.concatenate(img_chs, axis=1)

    return chsubset, chsubset_path

def batch_align_2imgs(aligndir, img1_ch_align=0, img2_ch_align=0, img1_ch_subset=None, img2_ch_subset=None):
    # get directories of images to align
    dirs = [Path(d) for d in aligndir.iterdir() if d.is_dir()]
    assert (len(dirs) == 2), 'Please only include 2 directories to align'
    # Sorts directories by number, alphabetical order
    dirs.sort()

    assert dirs[0].exists(), "dir1 does not exist, please input a valid path"
    assert dirs[1].exists(), "dir2 does not exist, please input a valid path"

    dir1_raw_dirpath = Path(dirs[0]) / proc_dirname / raw_ometif_dirname
    assert dir1_raw_dirpath.exists(), f'Please process {dirs[0].name} into raw ome-tiffs'
    dir2_raw_dirpath = Path(dirs[1]) / proc_dirname / raw_ometif_dirname
    assert dir2_raw_dirpath.exists(), f'Please process {dirs[1].name} into raw ome-tiffs'

    # Get imgnames from each of the directories to be aligned
    imgnames1 = [path.name for path in Path(dir1_raw_dirpath).glob('*.ome.tif')]
    imgnames1.sort()
    imgnames1_totalcount = len(imgnames1)
    imgnames2 = [path.name for path in Path(dir2_raw_dirpath).glob('*.ome.tif')]
    imgnames2.sort()

    # Check for and exclude already aligned images
    multiexp_dirpath = aligndir.parent
    aligned_dirpath = multiexp_dirpath / aligned_dirname / f'{dirs[0].name}_and_{dirs[1].name}_aligned'\
                      / proc_dirname / raw_ometif_dirname
    if aligned_dirpath.exists():

        # Extracts the img1 directory name from the aligned file (relies on the naming convention staying the same)
        aligned_imgnames = [(path.name[:path.name.find('_aligned')] + '.ome.tif') for path in aligned_dirpath.glob('*.ome.tif')]
        imgnames1 = [imgname1 for imgname1 in imgnames1 if imgname1 not in aligned_imgnames]
        print(f'Skipping {imgnames1_totalcount - len(imgnames1)} previously aligned images')

        # Looks for and loads any existing alignment or error csv file
        aligned_csv_path = aligned_dirpath / alignment_csv_name
        if aligned_csv_path.is_file():
            align_df = pd.read_csv(aligned_csv_path)
            # Drop any rows that are missing t0 and t1 values
            align_df = align_df.dropna(axis=0, subset=['t0', 't1'])
        else:
            align_df = pd.DataFrame()
        error_csv_path = aligned_dirpath / error_csv_name
        if error_csv_path.is_file():
            error_df = pd.read_csv(error_csv_path)
        else:
            error_df = pd.DataFrame({'image name': [], 'error time': []})

    else:
        aligned_dirpath.mkdir(parents=True, exist_ok=True)

    # Save info about images in dataframes
    imgs1_df = pd.DataFrame()
    for imgname1 in imgnames1:
        info_df = extract_img_info(imgname1)
        imgs1_df = pd.concat([imgs1_df, info_df], ignore_index=True)

    imgs2_df = pd.DataFrame()
    for imgname2 in imgnames2:
        info_df = extract_img_info(imgname2)
        imgs2_df = pd.concat([imgs2_df, info_df], ignore_index=True)

    # Match up images based on the scene information
    align_df_new = imgs1_df.merge(imgs2_df, how='inner', on='Scene')
    align_df_new['t0'] = ''
    align_df_new['t1'] = ''

    # Get starting index for new images when previous and new dataframes are merged
    start_idx = align_df.shape[0]

    # Merge previous and new dataframes
    align_df = pd.concat([align_df, align_df_new], ignore_index=True)
    errors = []
    errtime = []

    for i, row in (align_df.loc[start_idx:]).iterrows():
        imgpath1 = Path(dir1_raw_dirpath) / row['Image Name_x']
        imgpath2 = Path(dir2_raw_dirpath) / row['Image Name_y']

        single_align_df = align_2imgs(str(imgpath1), str(imgpath2), aligned_dirpath,
                                      img1_ch_align, img2_ch_align, img1_ch_subset, img2_ch_subset)
        align_df.at[i, 't0'] = single_align_df.at[0, 't0']
        align_df.at[i, 't1'] = single_align_df.at[0, 't1']

        # Saves alignment info
        align_df.to_csv(aligned_csv_path, index=False)
        error_df.to_csv(error_csv_path, index=False)

    # Move original, unaligned images to a new folder
    unaligned_parent_dirpath = multiexp_dirpath / orig_unaligned_dirname
    unaligned_parent_dirpath.mkdir(parents=True, exist_ok=True)
    shutil.move(aligndir, unaligned_parent_dirpath)

    print(f'Done with experimental directory! Images saved to {aligned_dirpath}')
    return

def multiexp_align(multiexp_dir, img1_ch_align=0, img2_ch_align=0, img1_ch_subset=None, img2_ch_subset=None):
    # Finds directories within the multiexperiment directory
    # (excluding the directories that contain already aligned or
    # original, unaligned images corresponding to already aligned images)
    aligndirs = [aligndir for aligndir in multiexp_dir.iterdir() if (aligndir.is_dir() &
                                                                     (aligndir.name not in [orig_unaligned_dirname,
                                                                                            aligned_dirname]))]
    num_dirs = len(aligndirs)

    # Create directories for aligned, unaligned images
    aligned_dirpath = Path(multiexp_dir) / aligned_dirname
    aligned_dirpath.mkdir(parents=True, exist_ok=True)
    orig_unaligned_dirpath = Path(multiexp_dir) / orig_unaligned_dirname
    orig_unaligned_dirpath.mkdir(parents=True, exist_ok=True)

    for i, aligndir in enumerate(aligndirs):
        print(f'Starting to align {aligndir} ({i + 1}/{num_dirs})')
        batch_align_2imgs(aligndir, img1_ch_align=img1_ch_align, img2_ch_align=img2_ch_align,
                          img1_ch_subset=img1_ch_subset, img2_ch_subset=img2_ch_subset)

    print('Done with multi-exp align!')


if __name__ == '__main__':
    args = parser.parse_args()
    img1_ch_align = int(img1_ch_align)
    img2_ch_align = int(img2_ch_align)

    ch_subset = args.ch_subset
    multiexp_dir = args.multiexp_dir

    multiexp_align(multiexp_dir, img1_ch_align, img2_ch_align, ch_subset)
