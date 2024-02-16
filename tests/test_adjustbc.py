import numpy as np
import sys
from pathlib import Path
import src.d01_init_proc as adjustbc

src_path = str(Path.cwd().parent)

print(src_path)
if src_path not in sys.path:
    sys.path.append(src_path)

import src.d01_init_proc.subtractbg as adjustbc

params = {}

# 100 percentile is equivalent to no clipping at all
params['outlier_percentiles'] = [83, 30]

# 0 sigma_smoothing = no smoothing
params['sigmas_smoothing'] = [0.6, 0.6]

thresholds = np.ones((3, 2, 1, 1, 1))
imgname = "CE012_I1_d3_A2_40x_22hrs_timelapse_multiplecells_timestitched_cidP21-A2.ome.tif"

adjustbc.create_img_dict(imgname, thresholds, params)