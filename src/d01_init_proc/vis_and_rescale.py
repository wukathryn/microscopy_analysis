from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter
from src.d00_utils.dirnames import bg_sub_dirname, bg_sub_fig_dirname, init_rescale_dirname

#TODO: maybe move to utilities?
def get_img_and_metadata(imgpath):
    img_file = AICSImage(imgpath, reader=OmeTiffReader)
    img = img_file.data
    img = img.astype('uint16')

    metadata_d = {}
    metadata_d['ome_metadata'] = img_file.ome_metadata
    metadata_d['channel_names'] = img_file.channel_names
    metadata_d['physical_pixel_sizes'] = img_file.physical_pixel_sizes

    return img, metadata_d


# Obtain scaling factor for each image based on target percent grayvalue for the 99th percentile pixel
def get_scaling_fact(img, target_perc_grayval, percentile=70):
    target_grayval = (target_perc_grayval / 100) * np.iinfo(img.dtype).max
    img = img.astype('float')
    img[img == 0] = 'nan'
    bg_sub_percentile = np.nanpercentile(img, q=percentile, axis=(0, 2, 3, 4))

    scaling_fact = target_grayval / bg_sub_percentile
    scaling_fact = np.expand_dims(scaling_fact, axis=(0, 2, 3, 4))
    return scaling_fact


# Rescale image by scaling factor. Any values exceeding the max value for the image type will be clipped to the max value
def multiply_by_scaling_fact(img, scaling_fact):
    dtype = img.dtype
    img_rescaled = (img * scaling_fact)
    img_rescaled = np.clip(img_rescaled, a_min=None, a_max=np.iinfo(dtype).max)
    return img_rescaled.astype(dtype)


# Converts img from uint16 to uint8, scaling pixel intensities accordingly
def convert_to_8bit(img):
    assert str(img.dtype) == 'uint16', f'Can only convert uint16 images to uint8'
    scaling_factor = np.iinfo('uint8').max / np.iinfo('uint16').max
    img_rescaled = (img * scaling_factor)
    img_rescaled = np.clip(img_rescaled, a_min=None, a_max=np.iinfo('uint8').max)
    return img_rescaled.astype('uint8')


def rescale_img(img, target_perc_grayval=30,
                standardize_scaling=False, scaling_fact=None, conv_to_8bit=True):
    if (standardize_scaling is False or scaling_fact is None):
        scaling_fact = get_scaling_fact(img, target_perc_grayval)
    img_vis = multiply_by_scaling_fact(img, scaling_fact)
    if conv_to_8bit is True:
        img_vis = convert_to_8bit(img_vis)
    return img_vis, scaling_fact

def generate_subtractbg_fig(orig_img, bgsb_img, bgsub_imgname, params, target_perc_grayval=30, index=None):
    size_t, size_c, _, _, _ = orig_img.shape

    rescaled_orig_img, scaling_fact = rescale_img(orig_img, target_perc_grayval=target_perc_grayval,
                                                  standardize_scaling=False, conv_to_8bit=True)
    binary_bgsb_img = (bgsb_img > 0).astype('uint8')

    num_imgs = 2
    fig = plt.figure(figsize=(2 + num_imgs * size_c * 3, size_t * 3.5), constrained_layout=True)
    plt.rcParams.update({'font.size': 8})

    subfigs = fig.subfigures(1, size_c, squeeze=False)

    for c in range(size_c):
        subfig = subfigs[0, c]
        subfig.suptitle(f'Channel {c + 1}')

        subfigs_imgtype = subfig.subfigures(1, num_imgs)
        subfigs_imgtype[0].suptitle('Rescaled orig image')
        subfigs_imgtype[1].suptitle('Bg subtracted binary')

        axs_img = subfigs_imgtype[0].subplots(size_t, squeeze=False)
        axs_binary = subfigs_imgtype[1].subplots(size_t, squeeze=False)


        for t in range(size_t):
            axs_img[t, 0].imshow(rescaled_orig_img[t, c, 0, :, :], cmap='gray', interpolation=None)
            axs_img[t, 0].axis('off')
            axs_img[t, 0].set_title(f'Timepoint: {t + 1}')
            axs_binary[t, 0].imshow(binary_bgsb_img[t, c, 0, :, :], cmap='gray', interpolation=None)
            axs_binary[t, 0].axis('off')
            axs_binary[t, 0].set_title(f'Timepoint: {t + 1}')

    fig.suptitle(f'Idx: {index}, {bgsub_imgname}\n{str(params)}')
    plt.show()

    return fig

def batch_rescale_imgs(input_dirpath):
    input_dirpath = Path(input_dirpath)
    assert input_dirpath.exists(), "Please input a valid directory path"
    vis_dirpath = input_dirpath.parent / init_rescale_dirname
    vis_dirpath.mkdir(parents=True, exist_ok=True)

    imgpaths = [path for path in input_dirpath.glob('*.ome.tif')]
    if len(imgpaths)==0:
        print("No images found.")
    print(f'Rescaling {len(imgpaths)} images')

    rescale_params_df = pd.DataFrame()
    scaling_fact = None
    for i, imgpath in enumerate(imgpaths):
        img_file = AICSImage(imgpath, reader=OmeTiffReader)
        img = img_file.data
        ome_metadata = img_file.ome_metadata
        img_vis, scaling_fact = rescale_img(img, target_perc_grayval=50, standardize_scaling=False,
                                                            scaling_fact=scaling_fact, conv_to_8bit=False)
        OmeTiffWriter.save(img_vis, Path(vis_dirpath) / imgpath.name, ome_xml=ome_metadata)

        img_rescale_params = pd.DataFrame({'Image name': [imgpath.name]})

        size_c = scaling_fact.shape[1]
        for c in range(size_c):
            img_rescale_params[f'Scaling_fact_ch{c}'] = [scaling_fact[:, c, :, :, :].squeeze()]
        rescale_params_df = pd.concat([rescale_params_df, img_rescale_params], ignore_index=True)

    rescale_params_df.to_csv(Path(vis_dirpath) / 'rescale_params.csv')
    print("Done!")

    return vis_dirpath