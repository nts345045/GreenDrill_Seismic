"""
:module: S4_Ice_Thickness_Raytracing_Coarse.py
:purpose: Conduct coarse grid-search analysis to determine best single glacier ice thickness and
		  glacier ice velocity pair for the layer underlying shallow structure from WHB analysis
		  in Step 3 (S3). This site-averaged best-fit is saved to disk for use in sensitivity
		  testing in Step 5 (S5), where velocity model uncertainties and spread-specific models
		  are considered.
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



##### SUPPORTING METHODS #####


##### PROCESSING #####

### USER DEFINED UNCERTAINTIES ###
# Node location uncertainty in meters
Node_xSig = 6.
# Georod uncertainty in meters
GeoRod_xSig = 1.
# Phase pick time uncertainties in seconds
tt_sig = 1e-3
# Parameter sweep coefficients
rHN = 50 	# [m] range of H_N values to scan, centered on Hrms (from Dix equation)
dHN = 5 	# [m] increment to scan across H_N values
##################################

### MAP FILE STRUCTURE ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks_O2_idsw_v5.csv')
UDAT = os.path.join(OROOT,'Full*_v7*LHSn100.csv')
##########################


# Get desired WHB model filename
flist = glob(UDAT)


### Load Phase Pick Data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Subset primary reflection arrivals of interest

# Run full dataset first to enable coarse grid-search for mean solution

for f_ in flist:

	# In the case of full data-set
	if 'Full' in f_:
		# Load WHB Model
		df_MOD = pd.read_csv(f_)
		# Subset full data
		sD_ = df_picks[(df_picks['phz']=='S')&\
					   (df_picks['SRoff m'].notna())&\
					   (df_picks['kind'].isin([1,2]))]

		# Pull data vectors
		xx = sD_['SRoff m'].values
		tt = sD_['tt sec'].values
		# Populate instrument-type specific location uncertainties
		xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
		# Populate pick-time uncertainties
		tsig = np.ones(tt.shape)*tt_sig


		# Do coarse grid-search for best-fit H_N and V_N
		VN = 1e3/df_MOD['median u(z)'].values[-1]
		# Conduct Dix estimate of ice-thickness for initial parameter guess
		ODR_Dix = d1d.hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[400,VN])
		Hrms = ODR_Dix.beta[0]
		Vrms = ODR_Dix.beta[1]

		# Set ZN scan vector
		Z_Nv = np.linspace(Hrms - 25,Hrms + 25,11) 
		# Set VN scan vector
		V_Nv = np.linspace(VN - (3850 - VN),3850,11)
		df_GS, res_GS = d1d.raytracing_gridsearch(xx,tt,V_Nv,Z_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
												  full=True)


		# # Conduct VN parameter sweep
		# df_VNout, res_VN = d1d.raytracing_Vsearch(xx,tt,V_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
		# 										  Hrms,full=True)


		# # Conduct parameter sweep
		# df_RT, res_block = d1d.raytracing_zsearch(xx,tt,Z_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
		# 										VN=VN,full=True)


		# Do fine grid-search

### MOVE THIS TO S5
# for f_ in flist:
# 	# Load WHB Model
# 	df_MOD = pd.read_csv(f_)
# 	# In the case of full data-set
# 	if 'Full' not in f_:

# 		# Subset spread-specific data
# 		SP_ = os.path.split(f_)[-1].split('_')[1]
# 		sD_ = df_picks[(df_picks['phz']=='S')&\
# 					   (df_picks['SRoff m'].notna())&\
# 					   (df_picks['kind'].isin([1,2]))&\
# 					   (df_picks['spread']==SP_)]
# 		# Pull data vectors
# 		xx = sD_['SRoff m'].values
# 		tt = sD_['tt sec'].values
# 		# Populate instrument-type specific location uncertainties
# 		xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
# 		# Populate pick-time uncertainties
# 		tsig = np.ones(tt.shape)*tt_sig

# 		### PLACEHOLDER - ITERATE ACROSS PARAMETRIC REPRESENTATIONS
# 		# for par_ in ('mean','median','Q10','Q90','p1sig','m1sig'):
# 		#	if par_ == 'p1sig':
# 			# elif par_ == 'm1sig':
# 		# else:
# 		VN = 1e3/df_MOD['median u(z)'].values[-1]
# 		# Conduct Dix estimate of ice-thickness for initial parameter guess
# 		ODR_Dix = d1d.hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[400,VN])
# 		Hrms = ODR_Dix.beta[0]
# 		Vrms = ODR_Dix.beta[1]

# 		# Set VN scan vector
# 		V_Nv = np.linspace(VN,3950,10)
# 		# Conduct VN parameter sweep
# 		df_VNout, res_VN = d1d.raytracing_Vsearch(xx,tt,V_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
# 												  Hrms,full=True)

# 		# Set ZN scan vector
# 		Z_Nv = np.arange(Hrms - 0.5*rHN,Hrms + 0.5*rHN + dHN,dHN) 
# 		# Conduct parameter sweep
# 		df_RT, res_block = d1d.raytracing_zsearch(xx,tt,Z_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
# 												VN=VN,full=True)




