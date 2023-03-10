"""
:module: S5_Ice_Thickness_Sensitivity_Testing.py
:purpose: Conduct fine grid-search analysis to determine best single glacier ice thickness and
		  glacier ice velocity pair for the layer underlying shallow structure from WHB analysis
		  in Step 3 (S3). 

		  NOTE: This takes a while to run


:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

:: TODO ::
Parallelization may be worthwhile in future versions

"""
import sys
import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

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

# Runtime controls
issave = True
isplot = True

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

### Load coarse grid-search results
df_GSC = pd.read_csv(os.path.join(OROOT,'Ice_Thickness_1DRT_v7_COARSE_GridSearch.csv'))

# Iterate across 
for f_ in flist:
	# Load WHB Model
	df_MOD = pd.read_csv(f_)
	# In the case of full data-set
	if 'Full' in f_:
		# Subset full data
		sD_ = df_picks[(df_picks['phz']=='S')&\
					   (df_picks['SRoff m'].notna())&\
					   (df_picks['kind'].isin([1,2]))]
		NAME = 'Full'
	# In the case of spread-specific data
	else:
		# Subset spread-specific data
		SP_ = os.path.split(f_)[-1].split('_')[1]
		sD_ = df_picks[(df_picks['phz']=='S')&\
					   (df_picks['SRoff m'].notna())&\
					   (df_picks['kind'].isin([1,2]))&\
					   (df_picks['spread']==SP_)]
		NAME = SP_

	### Pull data vectors ###
	xx = sD_['SRoff m'].values
	tt = sD_['tt sec'].values
	# Populate instrument-type specific location uncertainties
	xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
	# Populate pick-time uncertainties
	tsig = np.ones(tt.shape)*tt_sig

	### PREPARE FOR FINE GRID-SEARCH ###
	# Pull best-fit VN,ZN from coarse data
	df_CB = df_GSC[df_GSC['res L2']==df_GSC['res L2'].min()]
	V_Nc = df_CB['VN m/s'].values[0]
	Z_Nc = df_CB['Z m'].values[0]

	for fld_ in ['mean','Q10','Q90']:
		print('Running %s (%s)'%(NAME,fld_))
		# Get bottom-most values from WHB model
		Uwhb = df_MOD['%s u(z)'%(fld_)].values
		Zwhb = df_MOD['%s z'%(fld_)].values
		# Get WHB maximum velocity
		V_Nwhb = 1e3/Uwhb[-1]
		# Get Hrms
		ODR_Dix = d1d.hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[400,V_Nwhb])
		Hrms = ODR_Dix.beta[0]
		# Define scaling of grid-search based off of differences
		dV_N = np.abs(V_Nwhb - V_Nc)
		if dV_N < 0.5:
			dV_N = 0.5
		dZ_N = np.abs(Hrms - Z_Nc)
		if dZ_N < 0.5:
			dZ_N = 0.5

		V_Nvc = np.arange(V_Nc - 5.*dV_N,V_Nc + 5.*dV_N + 0.5,0.5)
		Z_Nvc = np.arange(Z_Nc - 1.1*dZ_N,Z_Nc + 1.1*dZ_N + 0.5,0.5)

		print('Consisting of %d velocity x %d depth iterations'%(len(V_Nvc),len(Z_Nvc)))
		# Z_Nvc = np.linspace(df_l2rm['Z m'].values - dV,df_l2rm['Z m'].values + dV,41)
		# V_Nvc = np.linspace(df_l2rm['VN m/s'].values - dZ,df_l2rm['VN m/s'].values + dZ,5)

		### RUN GRID SEARCH ###
		df_GSf, res_GSf = d1d.raytracing_gridsearch(xx,tt,V_Nvc,Z_Nvc,Uwhb,Zwhb,full=True)
				
		### SAVE FINE MODEL SUMMARY TO DISK ###
		if issave:
			df_GSf.to_csv(os.path.join(OROOT,'test_Ice_Thickness_1DRT_v5_FINE_%s_%s_GridSearch.csv'%(NAME,fld_)),header=True,index=False)


		# Get best-fit model by provided metrices
		df_l2rm = df_GSf[df_GSf['res L2'] == df_GSf['res L2'].min()]
		df_urm = df_GSf[np.abs(df_GSf['res mean']) == np.abs(df_GSf['res mean']).min()]
		df_srm = df_GSf[df_GSf['res std'] == df_GSf['res std'].min()]
		# Get best-fit 
		res_l2best = res_GSf[df_GSf['res L2']==df_GSf['res L2'].min()]
		# Plot coarse model results
		Vm = np.reshape(df_GSf['VN m/s'].values,(len(V_Nvc),len(Z_Nvc)))
		Zm = np.reshape(df_GSf['Z m'].values,(len(V_Nvc),len(Z_Nvc)))
		l2rm = np.reshape(df_GSf['res L2'].values,(len(V_Nvc),len(Z_Nvc)))

		urm = np.reshape(df_GSf['res mean'].values,(len(V_Nvc),len(Z_Nvc)))

		srm = np.reshape(df_GSf['res std'].values,(len(V_Nvc),len(Z_Nvc)))

		if isplot:
			plt.figure()
			plt.subplot(311)
			plt.plot(xx,tt,'k.',label='data')
			plt.plot(xx,tt - res_l2best[0,:],'r.',alpha=0.1,label='best-fit %s (%s) fine model'%(NAME,fld_))
			plt.legend()
			plt.xlabel('Source-Receiver Offset [m]')
			plt.ylabel('Two-way Travel Time [sec]')
			plt.title('Data model comparison - Fine Resolution')
			plt.subplot(323)
			plt.pcolor(Vm,Zm,l2rm)
			plt.plot(df_l2rm['VN m/s'],df_l2rm['Z m'],'r*',ms=16)
			plt.colorbar()
			plt.title('$|| t - \\hat{t}||_2$ [sec]')
			plt.subplot(324)
			plt.pcolor(Vm,Zm,np.abs(urm))
			plt.plot(df_urm['VN m/s'],df_urm['Z m'],'c*')
			plt.colorbar()
			plt.title(' |mean($t - \\hat{t}$)| [sec]')
			plt.subplot(325)
			plt.pcolor(Vm,Zm,srm)
			plt.plot(df_srm['VN m/s'],df_srm['Z m'],'m*')
			plt.colorbar()
			plt.title('std($t - \\hat{t}$) [sec]')
			plt.subplot(326)
			plt.pcolor(Vm,Zm,(srm - np.min(srm))/(np.max(srm) - np.min(srm)) * \
							 (urm - np.min(urm))/(np.max(urm) - np.min(urm)) * \
							 (l2rm - np.min(l2rm))/(np.max(l2rm) - np.min(l2rm)))
			plt.plot(df_l2rm['VN m/s'],df_l2rm['Z m'],'r*',ms=16)
			plt.plot(df_urm['VN m/s'],df_urm['Z m'],'c*')
			plt.plot(df_srm['VN m/s'],df_srm['Z m'],'m*')
			plt.title('Unit-scalar product of L-2 norm $\\mu\\sigma$')
			plt.colorbar()


			plt.show()



