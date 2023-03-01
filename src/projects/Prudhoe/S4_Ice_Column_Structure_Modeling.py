"""
:module: S4_Ice_Column_Structure_Modeling.py
:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

:synopsis:
	:inputs: Shallow structure velocity models, reflected arrival travel times
	:tasks: 
		1) Calculate initial estimate of <V,H>_N under the assumption that a NMO with V_RMS
		2) Conduct ray-tracing with ~some type~ of perturbation of <V,H>_N to optimize
			--> this may require building a chunky wrapper that includes ray-tracing and 
				providing a pre-compiled 1-2 D model
		3) Use the best-fit V_N to inform Kohnen firn density calculation
		3.b) Also do Robin firn density calculation.


:: TODO ::

This is going to get merged into S3 so each realization of 

"""
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.join('..','..'))
import core.RayTracing1D as rt1
import core.Dix_Conversion as dix

### CORE PROCESSES ###

def gridsearch_dix(xx,tt,Uv,ZNv,dx=10):
	"""
	Conduct a grid-search over specified slownesses (Uv) and 
	bottom-
	"""
	ixx = np.arange(np.min(xx),np.max(xx)+dx,dx)


	for uN_ in Uv:
		# Downsample overburden velocity profile
		imod_OB = dix.resample_WHB(i_zDv['z m'],i_zDv['uD ms/m']*1e-3)
		# Generate 1D velocity model
		iMOD = v1d.generate_layercake_vel(Vv=[1e3/imod_OB['uRMS'],Zv=np.append([imod_OB['Ztop'],4000]))
		# Iterate across source-depths and model travel times
		for zN_ in zNv:
			# Calculate travel-times for 10m grid
			ixx = np.arange(np.min(xx),np.max(xx)+10,10)
			itt = v1d.raytrace_explicit(iMOD,ixx,zN_)
			itt_hat = np.interp(xx,ixx,itt)


##### ACTUAL PROCESSING #####
### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks.csv')


df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
D_ = df_picks[(df_picks['phz']=='S')&(df_picks['kind']==2) & (df_picks['SRoff m'].notna())]

# Iterate over spreads for overburden model
for i_,SP_ in enumerate(D_['spread'].unique())
	iD_ = D_[D_['spread']==SP_]
	# Load relevant WHB models
	
	# Downsample WHB
	WHB_disc = dix.resample_WHB()

	# Iterate over shots to calculate <H,V>_N
	for j_,SH_ in enumerate(iD_['shot #'].unique()):
		ijD_ = iD_[iD_['shot #']==SH_]

		# Conduct Coarse Gridsearch
		df_CGS = dix.hyperbolic_fit_simple(ijD_['SRoff m'].values,ijD_['tt sec'].values,\
										   )




