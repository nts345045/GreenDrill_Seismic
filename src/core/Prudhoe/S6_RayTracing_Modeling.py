"""
:module: RayTracing_Modeling.py
:purpose: Conduct ray-tracing for each pick in a given spread using the best-fit velocity
			model from Step 5 (Sensitivity testing)

:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

:: TODO ::
WORK IN PROGRESS - MUST FILL IN METHODS FOR RAY-PATH LENGTHS

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
# Filter down to phase picks of diving (P), primary reflections (S), and multiples (R)
df_picks = df_picks[df_picks['phz'].isin(['P','S','R'])]

# Create dictionary for velocity model files
mod_dict = {'Full':{'WHB':'Full_v8_WHB_ODR_LHSn100.csv',\
					'RT':'Ice_Thickness_1DRT_v8_K2_FINE_WIDE_Full_mean_GridSearch.csv'}}
for sp_ in ['NS01','NS02','NS03','WE01','WE02','WE03']:
	mod_dict.update({sp_:{'WHB':'Spread_%s_v8_GeoRod_WHB_ODR_LHSn100.csv'%(sp_),\
						  'RT':'Ice_Thickness_1DRT_v8_D2_FINE_%s_mean_DepthSweep.csv'%(sp_)}})


# Iterate across unique combinations
df_p_update = pd.DataFrame()


## Process Node Raytracing ##
idf_picks = df_picks[df_picks['itype']=='Node']
df_WHB = pd.read_csv(os.path.join(OROOT,mod_dict['Full']['WHB']))
df_RH = 


for SP_ in df_picks['spread'].unique():
	df_WHB = pd.read_csv(mod_dict)


for PH_,IT_,SP_ in df_picks[['phz','itype','spread']].unique():
	# Subset data
	idf_picks = df_picks[(df_picks['phz']==PH_)&\
						(df_picks['itype']==IT_)&\
						(df_picks['spread']==SP_)]
	# If pick is on a node, use the 'Full' ensemble models
	if IT_ == 'Node':
		P_mod = 
		SR_mod = 

	# If pick is on a GeoRod, use the spread-specific models
	elif IT_ == 'GeoRod':



	


	# ...if diving wave...
	if PH_ == 'P':
		# ...picked on a node...
		if IT_ == 'Node':
			# ...use ensemble WHB model
		else:
			# ...use spread specific WHB model
	# ...if primary reflection...	
	elif PH_ == 'S':
		# ...picked on a node...
		if IT_ == 'Node':
			#...use ensemble mean ice-thickness model
		else:
			#...use spread-specific ice-thickness model
	#...if reflection multiple...
	elif PH_ == 'R':
		if IT_ == 'Node':
			#...









