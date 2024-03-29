"""
:module: S4B_Ex2_Spread_Firn_Average_VN.py
:purpose: Use the site-average velocity structure as a reference model and perturb bottom-layer thickness
		(H_N) to improve data-model fit on a spread-wise (Ex1) and shot-wise (Ex2) basis.

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
from glob import glob


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
# Saving control
issave = True

apply_DT = False

# Parameter sweep coefficient
full_sweep = True
if full_sweep:
	COARSE_dZN = 30 # [m] +/- (half) range to scan over for Nth layer thickness in coarse (spread-wise) grid-searches
	COARSE_NODES= 13 # [#] grid nodes for ZN sweeps in fine grid-searches
	FINE_dZN = 15 # [m] +/- (half) range to scan over for the Nth layer thickness in fine (spread-wise) grid-searches
	FINE_NODES= 31 # [#] grid nodes for ZN sweeps in fine grid-searches
	VFINE_dZN = 15 # [m] +/- (half) range to scan over for the Nth layer thickness in very fine (shot-wise) grid-searches
	VFINE_NODES = 31 # [#]grid nodes for ZN sweeps in very fine grid-searches
else:
	COARSE_dZN = 30 # [m] +/- (half) range to scan over for Nth layer thickness in coarse (spread-wise) grid-searches
	COARSE_NODES= 5 # [#] grid nodes for ZN sweeps in fine grid-searches
	FINE_dZN = 15 # [m] +/- (half) range to scan over for the Nth layer thickness in fine (spread-wise) grid-searches
	FINE_NODES= 5 # [#] grid nodes for ZN sweeps in fine grid-searches
	VFINE_dZN = 15 # [m] +/- (half) range to scan over for the Nth layer thickness in very fine (shot-wise) grid-searches
	VFINE_NODES = 5 # [#]grid nodes for ZN sweeps in very fine grid-searches



######## DATA LOADING SECTION ########
### MAP FILE STRUCTURE ###
# Main Directory
ROOT = os.path.join('..','..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Inglefield_Land')
# Model Sub-Directory
MROOT = os.path.join(ROOT,'velocity_models','structure_experiments')
# Phase Data File
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')
# Wiechert-Herglotz-Bateman Reference Model(s)
UFMT = os.path.join(ROOT,'velocity_models','Spread_{SP}_v5_ele_MK2_ptO3_sutured_GeoRod_WHB_ODR_LHSn100.csv')
# Reference KB79 Model
CFMT = os.path.join(ROOT,'velocity_models','Spread_{SP}_v5_ele_MK2_ptO3_sutured_GeoRod_KB79_ODR.csv')

### Load Phase Pick Data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')


# Iterate across spread-specific firn models
for SP_ in ['NS01','NS02','NS03','WE01','WE02','WE03']:
	### Load WHB Model for Average Firn Structure
	df_MOD = pd.read_csv(UFMT.format(SP=SP_))
	df_COV = pd.read_csv(CFMT.format(SP=SP_))
	if apply_DT:
		KB_DT = df_COV['mean'].values[-1]
	else:
		KB_DT = 0.
	# Iterate across firn perturbation types
	for fld_ in ['mean','Q025','Q975']:
		for KD_ in [1,2]:
			print('Running %s (%s firn model) for Z_N (K: %d)'%(SP_,fld_,KD_))
			### Load relevant deep model from ensemble average
			df_VZN = pd.read_csv(os.path.join(MROOT,'S5A_FINE_%s_Average_Firn_Model_Average_Deep_Structure.csv'%(fld_)))

			# Get best-fit ZN and VN values from ensemble as starting guesses
			IBEST = df_VZN['res L2'] == df_VZN['res L2'].min()
			VNo = df_VZN[IBEST]['VN m/s'].values[0]
			ZNo = df_VZN[IBEST]['Z m'].values[0]

			### Pull relevant shallow model fields for Average Firn Model
			Uwhb = df_MOD['%s u(z)'%(fld_)].values
			Zwhb = df_MOD['%s z'%(fld_)].values

			#### DATA SUBSETTING SECTION ####
			# Subset Observations
			sD_ = df_picks[(df_picks['phz']=='S')&\
						   (df_picks['SRoff m'].notna())&\
						   (df_picks['kind']==KD_)&\
						   (df_picks['itype']=='GeoRod')&\
						   (df_picks['spread']==SP_)]

			### Pull data vectors
			xx = sD_['SRoff m'].values
			tt = sD_['tt sec'].values + 1e-3*KB_DT
	
			# Populate instrument-type specific location uncertainties
			xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
			# Populate pick-time uncertainties
			tsig = np.ones(tt.shape)*tt_sig

			######## GRID-SEARCH SECTION ########

			# Set coarse ZN scan vector
			Z_Ncv = np.linspace(ZNo - COARSE_dZN,ZNo + COARSE_dZN,COARSE_NODES) 

			print('Coarse Parameter Search Starting')
			### RUN COARSE GRID SEARCH ###
			df_ZSc, res_ZSc = d1d.raytracing_Zsearch(xx,tt,Z_Ncv,Uwhb,Zwhb,VN=VNo,full=True)

			### SAVE COARSE MODEL SUMMARY TO DISK ###
			if issave:
				df_ZSc.to_csv(os.path.join(MROOT,'S5B_Ex2_COARSE_{pert}_{sp}_Firn_Model_{sp}_Depth_Fit_K{kd}.csv'.format(pert=fld_,sp=SP_,kd=KD_)),header=True,index=False)

			# Fetch best-fit model in the L-2 norm minimization sense
			IBEST = df_ZSc['res L2']==df_ZSc['res L2'].min()

			Z_Nc = df_ZSc[IBEST]['Z m'].values[0]
			# Compose fine parameter sweep vector
			Z_Nfv = np.linspace(Z_Nc - FINE_dZN, Z_Nc + FINE_dZN,FINE_NODES)
			print('Fine Parameter Search Starting')

			### RUN FINE GRID SEARCH ###
			df_ZSf, res_ZSf = d1d.raytracing_Zsearch(xx,tt,Z_Nfv,Uwhb,Zwhb,VN=VNo,full=True)
			### SAVE FINE MODEL SUMMARY TO DISK ###
			if issave:
				df_ZSf.to_csv(os.path.join(MROOT,'S5B_Ex2_FINE_{pert}_{sp}_Firn_Model_{sp}_Depth_Fit_K{kd}.csv'.format(pert=fld_,sp=SP_,kd=KD_)),header=True,index=False)

			# Get best-fit thickness estimate for given spread/perturbation iteration
			JBEST = df_ZSf['res L2']==df_ZSf['res L2'].min()
			Z_Nf = df_ZSf[JBEST]['Z m'].values[0]

			######## ITERATE ACROSS SHOTS #######
			for SH_ in sD_['shot #'].unique():
				print('Running very fine parameter sweep on spread %s, shot %s'%(SP_,SH_))
				ixx = sD_[sD_['shot #']==SH_]['SRoff m'].values
				itt = sD_[sD_['shot #']==SH_]['tt sec'].values
				Z_Nvfv = np.linspace(Z_Nf - VFINE_dZN, Z_Nf + VFINE_dZN,VFINE_NODES)
				df_ZSvf, res_ZSvf = d1d.raytracing_Zsearch(ixx,itt,Z_Nvfv,Uwhb,Zwhb,VN=VNo,full=True)
				if issave:
					df_ZSvf.to_csv(os.path.join(MROOT,'S5B_Ex2_VFINE_{pert}_{sp}_Firn_Model_{sp}_shot_{sh}_Depth_Fit_K{kd}.csv'.format(pert=fld_,sp=SP_,kd=KD_,sh=SH_)),header=True,index=False)


