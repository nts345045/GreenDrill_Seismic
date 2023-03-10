"""
:module: S3_Shallow_Ice_Structure_MCMC.py
:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu
:Synopsis:
    Inputs: Compiled picks (from S1)
    Tasks: Conduct KB79 fitting and WHB inversion (with uncertainty quant) on:
            1) Whole data-set gather
            2) Spread gathers
            3) Shot gathers
           Estimate ice-column structure using WHB profiles, propagating uncertainties with a
           bootstrap approach informed by a latin hypercube sampler.
    Outputs: Gather-specific KB79 fits and vertical ice-structure models, model summary index


"""
import pandas as pd
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Add repository root to path & get repo modules of use
sys.path.append(os.path.join('..','..'))
import util.KB_WHB_Inversion as kwi
import util.Firn_Density as fpz
import util.InvTools as inv



##### SUPPORTING METHODS #####

def run_WHB_lhs(xx,tt,xsig,tsig,fit_type=0,sum_rule='trap',n_draw=100,min_sig2_mag = 1e-12):
	"""
	Run a Wichert-Herglotz-Bateman inversion on supplied travel-time vs offset data 
	considering data uncertainties and propagating uncertainties through the inversion
	using a Bayesian approach with a latin hypercube samplier (lhs)
	
	:: INPUTS ::
	:param xx: source-receiver offsets in meters
	:param tt: travel-times of direct/diving wave arrivals in seconds
	:param xsig: source-receiver offset uncertainties in meters
	:param tsig: travel-time uncertainties in seconds
	:param fit_type: solution control for scipy.odr.ODR - 	0 = ODR
															1 = ODR + full output
															2 = LSQ
	:param sum_rule: summation rule to approximate the WHB integral
	:param n_draw: number of samples to draw from the prior distribution
	:param min_sig2_mag: Minimum acceptable magnitude

	:: OUTPUTS ::
	:return output: output from ODR fitting 
	:return uDn: [m,n_draw] array of slowness values from WHB realizations
	:return zn: [m,n_draw] array of depth values from WHB realizations
	"""
	# Do unweighted inversion first to get first estimate of KB79 parameters
	try:
		beta1,cov_beta1 = kwi.curvefit_KB79(xx,tt)
	except RuntimeError:
		breakpoint()
	print('(LSQ) beta0 -- %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4]))
	# Do inverse-variance-weighted Orthogonal Distance Regression estimate for KB79
	output = kwi.ODR_KB79(xx,tt,xsig,tsig,beta0=beta1,fit_type=fit_type)
	beta2 = output.beta
	print('(ODR) beta1 -- %.2e %.2e %.2e %.2e %.2e'%(beta2[0],beta2[1],beta2[2],beta2[3],beta2[4]))
	# Create LHS realizations from model parameters
	# Filter for 0-valued covariances
	cIND = np.abs(output.sd_beta) >= min_sig2_mag
	# Create crunched mean vector and covariance matrix for LHS sampling
	um_crunch = output.beta[cIND]
	Cm_crunch = output.cov_beta[cIND,:][:,cIND]
	# breakpoint()
	samps = inv.norm_lhs(um_crunch,Cm_crunch,n_samps=n_draw)
	print('LHS: %d samples drawn'%(n_draw))
	# Iterate over LHS samples and run WHB
	z_mods = [];u_mods = []; mod_ind = np.arange(0,n_draw)
	for i_ in tqdm(range(n_draw)):
		# Create perturbed parameter holder
		i_samp = np.zeros(5,)
		# Add in perturbed values
		i_samp[cIND] += samps[i_,:]
		# Add in values that had ~0-valued covariances
		i_samp[~cIND] += output.beta[~cIND]
		# Sanity check that i_samp contains no 0-valued entries
		i_samp[i_samp < min_sig2_mag] = min_sig2_mag
		# breakpoint() # Check that i_samp is (5,) or (5,1) or (1,5)
		i_zDv = kwi.loop_WHB_int(np.nanmax(xx)+1.,abcde=i_samp,sig_rule=sum_rule)
		z_mods.append(i_zDv['z m'])
		u_mods.append(i_zDv['uD ms/m'])

	# breakpoint() # Check dimensionality of z_mods and u_mods before putting in index/columns
	df_uD = pd.DataFrame(u_mods,index=mod_ind)
	df_z = pd.DataFrame(z_mods,index=mod_ind)

	return output,df_uD,df_z


def PP_WHB_write_outputs(OROOT,FN_start,output,df_uD,df_z,n_draw,full=False):
	"""
	Conduct post-processing and output file writing for MCMC simulation outputs

	:: INPUTS ::
	:param OROOT: Output ROOT directory (where to save file(s))
	:param FN_start: File-Name start
	:param output: 'output' object containing summary of ODR fitting for KB'79 model
	:param df_uD: Data Frame containing slowness models [in msec/m]
	:param df_z: DataFrame containing depth models [in m BGS]
	:param n_draw: explicit statement of number of simulations
	:param full: [BOOL] save all MCMC simulations?

	:: OUTPUTS ::
	:return df_MOD: pandas.DataFrame containing statistical representations of the velocity structure
					Parameters											Statistics
					u(z) = WHB slowness values [msec/m]		 			mean
					z = WHB depth values [m Below Glacier Surface]		std
																		median
																		10th quantile (Q10)
																		90th quantile (Q90)
	:return df_beta: pandas.DataFrame containing a summary of the KB'79 model fit


	"""


	# Write all 
	if full:
		df_uD.T.to_csv(os.path.join(OROOT,'%s_uD_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
		df_z.T.to_csv(os.path.join(OROOT,'%s_z_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
	# Write ODR model for KB79 to file
	df_beta = pd.DataFrame({'mean':output.beta,'a0':output.cov_beta[:,0],'a1':output.cov_beta[:,1],\
							'a2':output.cov_beta[:,2],'a3':output.cov_beta[:,3],'a4':output.cov_beta[:,4]},\
						    index=['a0','a1','a2','a3','a4'])
	df_beta.to_csv(os.path.join(OROOT,'%s_KB79_ODR.csv'%(FN_start)),header=True,index=True)
	# Get stats representations of each (Q25,Q50,Q75,mean,std)
	df_MOD = pd.DataFrame({'mean u(z)':df_uD.mean(axis=0).values,'mean z':df_z.mean(axis=0).values,\
						   'std u(z)':df_uD.std(axis=0).values,'std z':df_z.std(axis=0).values,\
						   'median u(z)':df_uD.median(axis=0).values,'median z':df_z.median(axis=0).values,\
						   'Q10 u(z)':df_uD.quantile(.1,axis=0).values,'Q10 z':df_z.quantile(.1,axis=0).values,\
						   'Q90 u(z)':df_uD.quantile(.9,axis=0).values,'Q90 z':df_z.quantile(.9,axis=0).values})
	df_MOD.to_csv(os.path.join(OROOT,'%s_WHB_ODR_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
	return df_MOD,df_beta


##### ACTUAL PROCESSING #####

#### PROCESSING CONTROLS ####
# Node location uncertainty in meters
Node_xSig = 6.
# Georod uncertainty in meters
GeoRod_xSig = 1.
# Phase pick time uncertainties in seconds
tt_sig = 1e-3
# Number of MCMC draws to conduct
n_draw = 100
ms2m=1e-12
sig_rule='trap'
write_MCMC = False

# Render plots?
isplot = False

### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks_O2_idsw_v5.csv')

### Load data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Subset diving-wave arrivals of interest
pD_ = df_picks[(df_picks['phz']=='P')&(df_picks['SRoff m'].notna())&(df_picks['kind']==1)&(df_picks['SRoff m'] > 3)]


### RUN PROCESSING ON ENSEMBLE DATA ###
xx = pD_['SRoff m'].values
tt = pD_['tt sec'].values*1000. # Put into Milliseconds
# Create coordinate standard deviation based on travel-time pic
xsig = Node_xSig*(pD_['itype']=='Node').values**2 + GeoRod_xSig*(pD_['itype']=='GeoRod').values**2
tsig = np.ones(tt.shape)*tt_sig


# Get shallow structure model
output,df_uD,df_z = run_WHB_lhs(xx,tt,xsig,tsig,n_draw=n_draw)
# Conduct post-processing and write shallow structure model to disk
df_MOD,df_beta = PP_WHB_write_outputs(OROOT,'Full_v7',output,df_uD,df_z,n_draw)

plt.figure()
plt.subplot(211)
plt.plot(xx,tt/1e3,'k.',label='Full')
plt.xlabel('Source receiver offset [m]')
plt.ylabel('Travel time [sec]')

k_ = 0
for fmt,X_,Y_ in [('k-','median u(z)','median z'),('k:','Q10 u(z)','Q10 z'),('k:','Q90 u(z)','Q90 z')]:

	if k_ == 0:
		plt.subplot(223)
		plt.plot(1e3/df_MOD[X_].values,df_MOD[Y_].values,fmt,label='Full')
		plt.subplot(224)
		plt.plot(fpz.rho_robin(1e3/df_MOD[X_].values),df_MOD[Y_].values,fmt,label='Full')

	else:
		plt.subplot(223)
		plt.plot(1e3/df_MOD[X_].values,df_MOD[Y_].values,fmt)
		plt.subplot(224)
		plt.plot(fpz.rho_robin(1e3/df_MOD[X_].values),df_MOD[Y_].values,fmt)

	k_ += 1



# Iterate across spreads
cid = ['blue','red','m','dodgerblue','g','orange']
SP_Sort = df_picks['spread'].unique()
SP_Sort.sort()
for i_,SP_ in enumerate(SP_Sort):
	# Subset diving-wave arrivals of interest
	pD_ = df_picks[(df_picks['phz']=='P')&(df_picks['SRoff m'].notna())&(df_picks['kind']==1)&(df_picks['SRoff m'] > 3)&(df_picks['spread']==SP_)]

	### RUN PROCESSING ON SPREAD DATA ###
	ixx = pD_['SRoff m'].values
	itt = pD_['tt sec'].values*1000. # Put into Milliseconds for KB79 inversion (provides stability)
	# Create coordinate standard deviation based on travel-time pic
	ixsig = Node_xSig*(pD_['itype']=='Node').values**2 + GeoRod_xSig*(pD_['itype']=='GeoRod').values**2
	itsig = np.ones(itt.shape)*tt_sig
	
	plt.subplot(211)
	plt.plot(ixx,itt/1e3,'.',color=cid[i_],label=SP_)

	# Get shallow structure model
	outputi,df_uDi,df_zi = run_WHB_lhs(ixx,itt,ixsig,itsig,n_draw=n_draw)
	# Conduct post-processing and write shallow structure model to disk
	idf_MOD,idf_beta = PP_WHB_write_outputs(OROOT,'Spread_%s_v7'%(SP_),outputi,df_uDi,df_zi,n_draw)


	k_ = 0
	for fmt,X_,Y_ in [('-','median u(z)','median z'),(':','Q10 u(z)','Q10 z'),(':','Q90 u(z)','Q90 z')]:
		if k_ == 0:
			plt.subplot(223)
			plt.plot(1e3/idf_MOD[X_].values,idf_MOD[Y_].values,fmt,label=SP_,color=cid[i_])
			plt.subplot(224)
			plt.plot(fpz.rho_robin(1e3/idf_MOD[X_].values),idf_MOD[Y_].values,fmt,label=SP_,color=cid[i_])

		else:
			plt.subplot(223)
			plt.plot(1e3/idf_MOD[X_].values,idf_MOD[Y_].values,fmt,color=cid[i_])
			plt.subplot(224)
			plt.plot(fpz.rho_robin(1e3/idf_MOD[X_].values),idf_MOD[Y_].values,fmt,color=cid[i_])
		k_ += 1
plt.subplot(223)
plt.ylim([100,-10])
plt.xlabel('Compressional Velocity [m/sec]')
plt.ylabel('Depth [mBGS]')
plt.legend()
plt.subplot(224)
plt.ylim([100,-10])
plt.xlabel('Seismic Density [kg $m^{-3}$]')
plt.ylabel('Depth [mBGS]')
plt.legend()
plt.show()


