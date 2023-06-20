"""
:module: S4_Ex1_Average_Site_Structure
:purpose: Conduct coarse grid-search analysis to determine best single glacier ice thickness and
		  glacier ice velocity pair for the layer underlying shallow structure from WHB analysis
		  in Step 3 (S3). Then conduct fine grid-search around coarse best-fit to increase solution
		  precison.

:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

:Processing Notes: 
Propagate data filtering decisions from STEP3:
+ For full dataset use all data, but use kind==2 (polarity defining pick) for reflected arrivals as
there are more observations of these arrivals compared to kind==1

:: TODO ::

"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join('..','..','..'))
import util.Dix_1D_Raytrace_Analysis as d1d


##### PROCESSING #####

### USER DEFINED UNCERTAINTIES ###
# Node location uncertainty in meters
Node_xSig = 6.
# Georod uncertainty in meters
GeoRod_xSig = 1.
# Phase pick time uncertainties in seconds
tt_sig = 1e-3
# Parameter sweep coefficients
COARSE_dHN = 50 # [m] +/- (half) range to scan over for Nth layer thickness in coarse grid-searches
COARSE_NODES= 21 # [#] grid nodes for HN and VN sweeps in fine grid-searches
FINE_dVN = 5 # [m/s] +/- (half) range to scan over for Nth layer velocity in fine grid-searches
FINE_dZN = 5 # [m] +/- (half) range to scan over for the Nth layer thickness in fine grid-searches
FINE_NODES= 21 # [#] grid nodes for HN and VN sweeps in fine grid-searches
# Runtime controls
issave = True
isplot = True
apply_DT = False

######## DATA LOADING SECTION ########
### MAP FILE STRUCTURE ###
# Main Directory
ROOT = os.path.join('..','..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# Model Sub-Directory
MROOT = os.path.join(ROOT,'velocity_models','structure_experiments')
# Phase Data File
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')
# Wiechert-Herglotz-Bateman Reference Model(s)
UDAT = os.path.join(ROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_WHB_ODR_LHSn100.csv')
# Reference KB79 Model
CDAT = os.path.join(ROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_KB79_ODR.csv')

### Load Phase Pick Data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')

### Load KB79 Model
df_cov = pd.read_csv(CDAT)
if apply_DT:
	KB_DT = df_cov['mean'].values[-1]
else:
	KB_DT = 0.

### Load WHB Model
df_MOD = pd.read_csv(UDAT)
# Subset full data
sD_ = df_picks[(df_picks['phz']=='S')&\
			   (df_picks['SRoff m'].notna())&\
			   (df_picks['kind'].isin([2]))]

### Pull data vectors
xx = sD_['SRoff m'].values
tt = sD_['tt sec'].values - 1e-3*KB_DT
# Populate instrument-type specific location uncertainties
xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
# Populate pick-time uncertainties
tsig = np.ones(tt.shape)*tt_sig

######## INITIAL PARAMETER ESTIMATION SECTION ########
### Do coarse grid-search for best-fit H_N and V_N
# Get bottom-WHB velocity
VN = 1e3/df_MOD['median u(z)'].values[-1]
# Conduct Dix estimate of ice-thickness for initial H_N parameter guess
ODR_Dix = d1d.hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[400,VN])
Hrms = ODR_Dix.beta[0]
Vrms = ODR_Dix.beta[1]

######## GRID-SEARCH SECTION ########
for fld_ in ['mean','Q025','Q975']:
	print('Running %s (firn model: %s)'%('Site Average Structure',fld_))
	Uwhb = df_MOD['%s u(z)'%(fld_)].values
	Zwhb = df_MOD['%s z'%(fld_)].values
	# Set ZN scan vector
	Z_Nv = np.linspace(Hrms - COARSE_dHN,Hrms + COARSE_dHN,COARSE_NODES) 
	# Set VN scan vector
	V_Nv = np.linspace(VN - (3850 - VN),3850,COARSE_NODES)
	print('Coarse Grid Search Starting')
	### RUN COARSE GRID SEARCH ###
	df_GSc, res_GSc = d1d.raytracing_gridsearch(xx,tt,V_Nv,Z_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,full=True)

	### SAVE COARSE MODEL SUMMARY TO DISK ###
	if issave:
		df_GSc.to_csv(os.path.join(MROOT,'S5A_COARSE_%s_Average_Firn_Model_Average_Deep_Structure.csv'%(fld_)),header=True,index=False)

	# Fetch best-fit model in the L-2 norm minimization sense
	IBEST = df_GSc['res L2']==df_GSc['res L2'].min()
	V_Nc = df_GSc[IBEST]['VN m/s'].values[0]
	Z_Nc = df_GSc[IBEST]['Z m'].values[0]
	# Compose fine grid-search vectors 
	V_Nvc = np.linspace(V_Nc - FINE_dVN, V_Nc + FINE_dVN,FINE_NODES)
	Z_Nvc = np.linspace(Z_Nc - FINE_dZN, Z_Nc + FINE_dZN,FINE_NODES)
	print('Fine Grid Search Starting')

	### RUN FINE GRID SEARCH ###
	df_GSf, res_GSf = d1d.raytracing_gridsearch(xx,tt,V_Nvc,Z_Nvc,Uwhb,Zwhb,full=True)
			
	### SAVE FINE MODEL SUMMARY TO DISK ###
	if issave:
		df_GSf.to_csv(os.path.join(MROOT,'S5A_FINE_%s_Average_Firn_Model_Average_Deep_Structure.csv'%(fld_)),header=True,index=False)







