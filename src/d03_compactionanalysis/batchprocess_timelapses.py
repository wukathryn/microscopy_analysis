import os
import pandas as pd
import numpy as np
import tifffile as tif
import definecellregions
#from compute_region_areas import compute_region_areas
import src.d00_utils.utilities as utils

# TODO: Only threshold once per image and save image stack
# TODO: Figure out best way to use masks (e.g. at compute region stage)?

def checkformasks(imgname, dir):
    base_imgname = os.path.splitext(os.path.basename(imgname))[0]
    mask_dir = os.path.join(dir, 'mask')
    mask_paths = [os.path.join(mask_dir, mask.name) for mask in os.scandir(mask_dir) if (base_imgname in mask.name)]
    return mask_paths

def extract_img_info(imgname, info):

    # Variables to help parse img file names
    exp_code_searchphrase = 'CE'
    exp_code_len = 5
    day_searchphrase = "_d"
    seriesnum_searchphrase = '_timestitched_'


    # Extracts info from the image filename
    exp = imgname[imgname.find(exp_code_searchphrase):(imgname.find(exp_code_searchphrase) + exp_code_len)]
    day = imgname[imgname.find(day_searchphrase) + len(day_searchphrase)]
    seriesnum = imgname[imgname.find(seriesnum_searchphrase) + len(seriesnum_searchphrase)]

    # Stores info keys and values into lists
    addl_info = {'image name': imgname, 'experiment': exp, 'day': day, 'series num': seriesnum}
    info.update(addl_info)

    return info

def process_timelapse(imgpath, channels_d, analysis_dir_paths):
    img = tif.imread(imgpath)
    cellregions, cellreg_ch_d = definecellregions.define_regions(img, channels_d)
    definecellregions.saveimgs(img, cellregions, os.path.join(analysis_dir_paths['withcellregions'], os.path.basename(imgpath)))

def batch_process_timelapses(exp_dir, channels_d):
    imgdir = os.path.join(exp_dir, 'bg_subtracted')

    analysis_dir_names = ['withcellregions', 'graphs', 'tables']
    analysis_dir_paths = {}
    for analysis in analysis_dir_names:
        analysis_dir_paths[analysis] = utils.getsavedirpath(exp_dir, analysis)

    overall_df_path = os.path.join(analysis_dir_paths['tables'], f'overall_dataframe.csv')
    if os.path.isfile(overall_df_path):
        overall_df = pd.read_csv(os.path.join(overall_df_path))
    else:
        overall_df = pd.DataFrame()

    imgnames = [file.name for file in os.scandir(imgdir) if (os.path.splitext(file)[1]=='.tif')]
    num_imgs = len(imgnames)

    for i, imgname in enumerate(imgnames):
        print(f'Processing {imgname} (file {i}/{num_imgs})')
        imgpath = os.path.join(imgdir, imgname)
        process_timelapse(imgpath, channels_d, analysis_dir_paths)

def main():
    exp_dir = '../../data/CE006'
    channels_d = {'CAAX protein': 0, 'cell membrane': 1}
    batch_process_timelapses(exp_dir, channels_d)


    #     for j, mask_path in enumerate(mask_paths):
    #         cellnum = j+1
    #         print(f'Processing {imgname} (file {i}/{num_imgs}, cell {cellnum}/{num_cells})')
    #
    #         # Only add new cell data to the pandas dataframe
    #         unique_cell_idx = os.path.splitext(imgname)[0] + f'_cell{cellnum}'
    #         if overall_df.empty or unique_cell_idx not in overall_df['unique cell index']:
    #             imgdata = {}
    #
    #             info = {'unique cell index': unique_cell_idx}
    #             info = extract_img_info(imgname, info)
    #             info.update({'cell num': cellnum, 'orig. ch order': channels_d,
    #                                           'cell regions path': withcellregions_paths[j], 'CAAX protein': CAAX_protein})
    #
    #             # Creates an image stack with defined cell regions
    #             thresholds, cellreg_ch_d = definecellregions(imgpath, mask_path, withcellregions_paths[j], channels_d)
    #
    #             num_timepoints = thresholds.shape[0]
    #             timepoints = np.arange(num_timepoints)
    #
    #             for (key, val) in info.items():
    #                 imgdata[key] = [val] * num_timepoints
    #
    #             imgdata['timepoint'] = timepoints
    #             # Separate out OTSU thresholds by channels
    #             for key in channels_d.keys():
    #                 thresh_keys = (f'OTSU threshold: {key}')
    #                 thresh_vals = np.squeeze(thresholds[:, channels_d[key]])
    #                 imgdata[thresh_keys] = thresh_vals
    #
    #             # Computes region metrics and returns it in a dataframe
    #             imgdata_df = compute_region_areas(withcellregions_paths[j], graphs_paths[j], imgdata, cellreg_ch_d)
    #
    #             # Add the individual cell dataframe to the overall dataframe
    #             overall_df = pd.concat([overall_df, imgdata_df], ignore_index=True)
    #
    #             # Save the dataframe after processing a defined number of files
    #             num_files_btw_save_df = 1
    #             if (i % num_files_btw_save_df)==0:
    #                 overall_df.to_csv(os.path.join(graphs_dir, 'overall_dataframe' + processing_var + '.csv'), index=False)
    #                 print('Saving overall dataframe')
    #
    # overall_df.to_csv(os.path.join(graphs_dir, 'overall_dataframe' + processing_var + '.csv'), index=False)

if __name__ == '__main__':
    main()