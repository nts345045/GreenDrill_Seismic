"""
:module: S5_Ice_Thickness_Raytracing_Coarse.py
:purpose: Conduct coarse grid-search analysis to determine best single glacier ice thickness and
		  glacier ice velocity pair for the layer underlying shallow structure from WHB analysis
		  in Step 3 (S3). This site-averaged best-fit is saved to disk for use in sensitivity
		  testing in Step 5 (S5), where velocity model uncertainties and spread-specific models
		  are considered.
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
# Parameter sweep coefficients
rHN = 50 	# [m] range of H_N values to scan, centered on Hrms (from Dix equation)
dHN = 5 	# [m] increment to scan across H_N values

# Runtime controls
issave = True
isplot = True

##################################

### MAP FILE STRUCTURE ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks_O2_idsw_v6.csv')
UDAT = os.path.join(OROOT,'Full_v8_WHB_ODR_LHSn100.csv')
##########################


# Get desired WHB model filename
# flist = glob(UDAT)


### Load Phase Pick Data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')

# In the case of full data-set
# if 'Full' in f_:
# Load WHB Model
# df_MOD = pd.read_csv(f_)
df_MOD = pd.read_csv(UDAT)
# Subset full data
sD_ = df_picks[(df_picks['phz']=='S')&\
			   (df_picks['SRoff m'].notna())&\
			   (df_picks['kind'].isin([2]))]

# Pull data vectors
xx = sD_['SRoff m'].values
tt = sD_['tt sec'].values
# Populate instrument-type specific location uncertainties
xsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
# Populate pick-time uncertainties
tsig = np.ones(tt.shape)*tt_sig


# Do coarse grid-search for best-fit H_N and V_N
# Get bottom-WHB velocity
VN = 1e3/df_MOD['median u(z)'].values[-1]
# Conduct Dix estimate of ice-thickness for initial parameter guess
ODR_Dix = d1d.hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[400,VN])
Hrms = ODR_Dix.beta[0]
Vrms = ODR_Dix.beta[1]

# Set ZN scan vector
Z_Nv = np.linspace(Hrms - 30,Hrms + 30,21) 
# Set VN scan vector
V_Nv = np.linspace(VN - (3850 - VN),3850,21)

### RUN GRID SEARCH ###
df_GSc, res_GSc = d1d.raytracing_gridsearch(xx,tt,V_Nv,Z_Nv,df_MOD['median u(z)'].values,df_MOD['median z'].values,\
										  full=True)

### SAVE COARSE MODEL SUMMARY TO DISK ###
if issave:
	df_GSc.to_csv(os.path.join(OROOT,'Ice_Thickness_K2_1DRT_v8_COARSE_GridSearch.csv'),header=True,index=False)

# Get best-fit model by provided metrices
df_l2rm = df_GSc[df_GSc['res L2'] == df_GSc['res L2'].min()]
df_urm = df_GSc[np.abs(df_GSc['res mean']) == np.abs(df_GSc['res mean']).min()]
df_srm = df_GSc[df_GSc['res std'] == df_GSc['res std'].min()]
# Get best-fit 
res_l2best = res_GSc[df_GSc['res L2']==df_GSc['res L2'].min()]
# Plot coarse model results
Vm = np.reshape(df_GSc['VN m/s'].values,(len(V_Nv),len(Z_Nv)))
Zm = np.reshape(df_GSc['Z m'].values,(len(V_Nv),len(Z_Nv)))
l2rm = np.reshape(df_GSc['res L2'].values,(len(V_Nv),len(Z_Nv)))

urm = np.reshape(df_GSc['res mean'].values,(len(V_Nv),len(Z_Nv)))

srm = np.reshape(df_GSc['res std'].values,(len(V_Nv),len(Z_Nv)))

if isplot:
	plt.figure()
	plt.subplot(311)
	plt.plot(xx,tt,'k.',label='data')
	plt.plot(xx,tt - res_l2best[0,:],'r.',alpha=0.1,label='best-fit coarse model')
	plt.legend()
	plt.xlabel('Source-Receiver Offset [m]')
	plt.ylabel('Two-Way Travel Time [sec]')
	plt.title('Data model comparison - Coarse Resolution')
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
	plt.pcolor(Vm,Zm,(1/3)*((srm - np.min(srm))/(np.max(srm) - np.min(srm)) + \
					 (urm - np.min(urm))/(np.max(urm) - np.min(urm)) + \
					 (l2rm - np.min(l2rm))/(np.max(l2rm) - np.min(l2rm))))
	plt.plot(df_l2rm['VN m/s'],df_l2rm['Z m'],'r*',ms=16)
	plt.plot(df_urm['VN m/s'],df_urm['Z m'],'c*')
	plt.plot(df_srm['VN m/s'],df_srm['Z m'],'m*')
	plt.title('Unit-scalar sum of L-2 norm $\\mu\\sigma$')
	plt.colorbar()



for fld_ in ['mean','Q10','Q90']:
	print('Running %s (%s)'%(NAME,fld_))
	Uwhb = df_MOD['%s u(z)'%(fld_)].values
	Zwhb = df_MOD['%s z'%(fld_)].values
	# Conduct fine-scale grid-search for ensemble
	V_Nvc = np.linspace(V_Nc - 5, V_Nc + 5,21)
	Z_Nvc = np.linspace(Z_Nc - 5, Z_Nc + 5,21)
	print('Consisting of %d velocity x %d depth iterations'%(len(V_Nvc),len(Z_Nvc)))

	### RUN GRID SEARCH ###
	df_GSf, res_GSf = d1d.raytracing_gridsearch(xx,tt,V_Nvc,Z_Nvc,Uwhb,Zwhb,full=True)
			
	### SAVE FINE MODEL SUMMARY TO DISK ###
	if issave:
		df_GSf.to_csv(os.path.join(OROOT,'Ice_Thickness_1DRT_v8_K2_FINE_WIDE_%s_%s_GridSearch.csv'%(NAME,fld_)),header=True,index=False)

	### GET UPDATED BEST-FIT VELOCITY & ICE THICKNESS ###
	iV_Nc = df_GSf[df_GSf['res L2']==df_GSf['res L2'].min()]['VN m/s'].values[0]
	iZ_Nc = df_GSf[df_GSf['res L2']==df_GSf['res L2'].min()]['Z m'].values[0]

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
		plt.plot(xx,tt - res_l2best[0,:],'r.',alpha=0.5,label='best-fit fine model')
		plt.legend()
		plt.xlabel('Source-Receiver Offset [m]')
		plt.ylabel('Two-Way Travel Time [sec]')
		plt.title('Data model comparison - Fine Resolution (%s)'%(fld_))
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
		plt.pcolor(Vm,Zm,(1/3)*((srm - np.min(srm))/(np.max(srm) - np.min(srm)) + \
						 (urm - np.min(urm))/(np.max(urm) - np.min(urm)) + \
						 (l2rm - np.min(l2rm))/(np.max(l2rm) - np.min(l2rm))))
		plt.plot(df_l2rm['VN m/s'],df_l2rm['Z m'],'r*',ms=16)
		plt.plot(df_urm['VN m/s'],df_urm['Z m'],'c*')
		plt.plot(df_srm['VN m/s'],df_srm['Z m'],'m*')
		plt.title('Unit-scalar sum of L-2 norm $\\mu\\sigma$')
		plt.colorbar()



	plt.show()


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




