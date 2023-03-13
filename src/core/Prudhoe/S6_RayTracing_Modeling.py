"""
:module: RayTracing_Modeling.py
:purpose: Conduct ray-tracing for each pick in a given spread using the best-fit velocity
			model from Step 5 (Sensitivity testing)

:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

"""

import sys
import os
import pandas as pd
import numpy as np
from glob import glob
sys.path.append(os.path.join('..','..'))
import util.Dix_1D_Raytrace_Analysis as d1d


### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks_O2_idsw_v5.csv')

# Load pick data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
#





