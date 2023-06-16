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
           Estimate ice-column structure using WHB profiles, propagating uncertainties from
           fitting data with a Kirchner & Bentley (1979) double-exponential model with
           Monte Carlo Markov Chain simulations informed by a latin hypercube sampler.
    Outputs: Gather-specific KB79 fits and vertical ice-structure models, model summary index

Data input notes:
For full dataset KB79/WHB inversion, use all kind==1 (first break) data to get an aerial average
representation of the shallow structure.

For spread-specific data slices, further sub-set to just GeoRods to maintain linear sampling along
the transect. Excluding Node data is a quick & dirty way to remove off-axis observations.

:: TODO ::
Output fitting to t0 done with curve_fit() into PP_write_outputs()

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

def run_WHB_lhs(xx,tt,xsig,tsig,fit_type=0,sum_rule='trap',n_draw=100,min_sig2_mag = 1e-12,bounds=(0,np.inf),KB79_ext=False):
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
	:return df_uD: [m,n_draw] array of slowness values from WHB realizations housed in a pandas.DataFrame
	:return df_z: [m,n_draw] array of depth values from WHB realizations housed in a pandas.DataFrame
	:return df_X: [m,n_draw] array of upper integration bounds for WHB inversions housed in a pandas.DataFrame
	"""
	# Do unweighted inversion first to get first estimate of KB79 parameters

	if not KB79_ext:
		print('Fitting Info -- a1   a2   a3   a4   a5  ')
		try:
			beta1,cov_beta1 = kwi.curvefit_KB79(xx,tt,bounds=bounds)
			print('(LSQ) beta0 -- %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4]))

		except RuntimeError:
			beta1 = np.array([10,30,30,0.01,0.1])
			cov_beta1 = np.zeros((5,5))
			print('(LSQ) failure -- defaulting -- %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4]))

	else:
		print('Fitting Info -- a1   a2   a3   a4   a5   t0')
		try:
			beta1,cov_beta1 = kwi.curvefit_KB79_ext(xx,tt,bounds=bounds)
			print('(LSQ) beta0 -- %.2e %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4],beta1[5]))

		except RuntimeError:
			beta1 = np.array([10,30,30,0.01,0.1,0])
			cov_beta1 = np.zeros((6,6))
			print('(LSQ) failure -- defaulting -- %.2e %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4],beta1[5]))



	# Do inverse-variance-weighted Orthogonal Distance Regression estimate for KB79
	if not KB79_ext:
		output = kwi.ODR_KB79(xx,tt,xsig,tsig,beta0=beta1,fit_type=fit_type)
		beta2 = output.beta
		print('(ODR) beta1 -- %.2e %.2e %.2e %.2e %.2e'%(beta2[0],beta2[1],beta2[2],beta2[3],beta2[4]))
	else:
		output = kwi.ODR_KB79(xx,tt - beta1[-1],xsig,tsig,beta0=beta1[:5],fit_type=fit_type)
		beta2 = output.beta
		print('(ODR) beta1 -- %.2e %.2e %.2e %.2e %.2e (input tt - dt applied. dt: %.2e)'%(beta2[0],beta2[1],beta2[2],beta2[3],beta2[4],beta1[5]))		
	# Create LHS realizations from model parameters
	# Filter for 0-valued covariances (and exclude t0 ranks - not used in WHB)
	# breakpoint()
	cIND = np.abs(output.sd_beta) >= min_sig2_mag
	# Create crunched mean vector and covariance matrix for LHS sampling
	um_crunch = output.beta[cIND]
	Cm_crunch = output.cov_beta[cIND,:][:,cIND]
	# breakpoint()
	samps = inv.norm_lhs(um_crunch,Cm_crunch,n_samps=n_draw)
	print('LHS: %d samples drawn'%(n_draw))
	# Iterate over LHS samples and run WHB
	z_mods = [];u_mods = []; x_int = []; mod_ind = np.arange(0,n_draw)
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
		x_int.append(i_zDv['X'])

	# breakpoint() # Check dimensionality of z_mods and u_mods before putting in index/columns
	df_uD = pd.DataFrame(u_mods,index=mod_ind)
	df_z = pd.DataFrame(z_mods,index=mod_ind)
	df_X = pd.DataFrame(x_int,index=mod_ind)
	if KB79_ext:
		out = {'output':output,'dt':beta1[-1],'covdtdt':cov_beta1[-1,-1]}
	else:
		out = output
	return out,df_uD,df_z,df_X


def PP_WHB_write_outputs(OROOT,FN_start,output,df_uD,df_z,df_X,n_draw,dt=0,covdtdt=0,full=False,KB79_ext=False):
	"""
	Conduct post-processing and output file writing for MCMC simulation outputs

	:: INPUTS ::
	:param OROOT: Output ROOT directory (where to save file(s))
	:param FN_start: File-Name start
	:param output: 'output' object containing summary of ODR fitting for KB'79 model
	:param df_uD: DataFrame containing slowness models [in msec/m]
	:param df_z: DataFrame containing depth models [in m BGS]
	:param df_X: DataFrame containing lateral distance integration bounds for a given WHB inversion [in m]
	:param n_draw: explicit statement of number of simulations
	:param full: [BOOL] save all MCMC simulations?

	:: OUTPUTS ::
	:return df_MOD: pandas.DataFrame containing statistical representations of the velocity structure
					Parameters											Statistics
					u(z) = WHB slowness values [msec/m]		 			mean
					z = WHB depth values [m Below Glacier Surface]		std
																		median
																		2.5th quantile (Q025)
																		97.5th quantile (Q975)
	:return df_beta: pandas.DataFrame containing a summary of the KB'79 model fit


	"""


	# Write all 
	if full:
		df_uD.T.to_csv(os.path.join(OROOT,'%s_uD_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
		df_z.T.to_csv(os.path.join(OROOT,'%s_z_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
		df_X.T.to_csv(os.path.join(OROOT,'%s_X_int_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
 	# Splice in the t0 and covt0t0 from the LSQ fitting
	if KB79_ext:
		# Write ODR model for KB79 to file
		df_beta = pd.DataFrame({'mean':np.r_[output.beta,dt],'a0':np.r_[output.cov_beta[:,0],0],\
								'a1':np.r_[output.cov_beta[:,1],0],'a2':np.r_[output.cov_beta[:,2],0],\
								'a3':np.r_[output.cov_beta[:,3],0],'a4':np.r_[output.cov_beta[:,4],0],\
								'dt':np.r_[np.zeros(5,),covdtdt]},\
							    index=['a0','a1','a2','a3','a4','t0'])
	else:
		# Write ODR model for KB79 to file
		df_beta = pd.DataFrame({'mean':output.beta,'a0':output.cov_beta[:,0],'a1':output.cov_beta[:,1],\
								'a2':output.cov_beta[:,2],'a3':output.cov_beta[:,3],'a4':output.cov_beta[:,4]},\
							    index=['a0','a1','a2','a3','a4'])


	df_beta.to_csv(os.path.join(OROOT,'%s_KB79_ODR.csv'%(FN_start)),header=True,index=True)
	# Get stats representations of each (Q25,Q50,Q75,mean,std)
	df_MOD = pd.DataFrame({'mean u(z)':df_uD.mean(axis=0).values,'mean z':df_z.mean(axis=0).values,'mean X':df_X.mean(axis=0),\
						   'std u(z)':df_uD.std(axis=0).values,'std z':df_z.std(axis=0).values,'std X':df_X.std(axis=0).values,\
						   'median u(z)':df_uD.median(axis=0).values,'median z':df_z.median(axis=0).values,'median X':df_X.median(axis=0).values,\
						   'Q025 u(z)':df_uD.quantile(.025,axis=0).values,'Q025 z':df_z.quantile(.025,axis=0).values,'Q025 X':df_X.quantile(.025,axis=0).values,\
						   'Q975 u(z)':df_uD.quantile(.975,axis=0).values,'Q975 z':df_z.quantile(.975,axis=0).values,'Q975 X':df_X.quantile(.95,axis=0).values})
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
# Write out all MCMC models (True), or just summary (False)?
write_MCMC = False
# Include a t0 shift for KB79 fitting?
KB79_ext_bool = False
# Exclude 0-offset shot-receiver gather data?
no_0_offset_data = False
# Render intermediate plots?
isplot = False

### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Inglefield_Land')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')

### Load data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Subset diving-wave arrivals of interest
pD_ = df_picks[(df_picks['phz']=='P')&(df_picks['SRoff m'].notna())&\
			   (df_picks['kind'].isin([1,2]))&(df_picks['SRoff m'] > 10)]


# NS01 fails to settle, filter out for now
# df_picks = df_picks[df_picks['spread']!='NS01']

if no_0_offset_data:
	pD_ = pD_[pD_['offset code'] > 0]
			   # &\
			   # (df_picks['itype']=='GeoRod')]


### RUN PROCESSING ON ENSEMBLE DATA ###
xx = pD_['SRoff m'].values
tt = pD_['tt sec'].values*1000. # Put into Milliseconds
# Create coordinate standard deviation based on travel-time pic
xsig = Node_xSig*(pD_['itype']=='Node').values**2 + GeoRod_xSig*(pD_['itype']=='GeoRod').values**2
tsig = np.ones(tt.shape)*tt_sig


if KB79_ext_bool:
	bounds = ((0,0,0,0,0,-30),(np.inf,np.inf,np.inf,np.inf,np.inf,30))
else:
	bounds = (0.,np.inf)

# Get shallow structure model
print('Running FULL')
output,df_uD,df_z,df_X = run_WHB_lhs(xx,tt,xsig,tsig,n_draw=n_draw,KB79_ext=KB79_ext_bool,bounds=bounds)
# breakpoint()
# Conduct post-processing and write shallow structure model to disk
if KB79_ext_bool:
	df_MOD,df_beta = PP_WHB_write_outputs(OROOT,'Full_v5_ele_MK2_ptO3_KB_ext',output['output'],df_uD,df_z,df_X,n_draw,\
										  KB79_ext=KB79_ext_bool,dt=output['dt'],covdtdt=output['covdtdt'])
	plot_dt_msec = df_beta['mean'].values[-1]
else:
	df_MOD,df_beta = PP_WHB_write_outputs(OROOT,'Full_v5_ele_MK2_ptO3_sutured',output,df_uD,df_z,df_X,n_draw,KB79_ext=KB79_ext_bool)
	plot_dt_msec = 0

plt.figure()
plt.subplot(211)
plt.plot(xx,(tt - plot_dt_msec)/1e3,'k.',label='Full')
plt.xlabel('Source receiver offset [m]')
plt.ylabel('Travel time [sec]')

k_ = 0
for fmt,X_,Y_ in [('k-','median u(z)','median z'),('k:','Q025 u(z)','Q025 z'),('k:','Q975 u(z)','Q975 z')]:

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
	print("Running {}".format(SP_))
	# Subset diving-wave arrivals of interest
	pD_ = df_picks[(df_picks['phz']=='P')&(df_picks['SRoff m'].notna())&\
				   (df_picks['kind']==1)&(df_picks['SRoff m'] > 3)&(df_picks['spread']==SP_)&\
				   (df_picks['itype']=='GeoRod')]
	# breakpoint()

	if no_0_offset_data:
		pD_ = pD_[pD_['offset code'] > 0]

	### RUN PROCESSING ON SPREAD DATA ###
	ixx = pD_['SRoff m'].values
	itt = pD_['tt sec'].values*1000. # Put into Milliseconds for KB79 inversion (provides stability)
	# Create coordinate standard deviation based on travel-time pic
	ixsig = Node_xSig*(pD_['itype']=='Node').values**2 + GeoRod_xSig*(pD_['itype']=='GeoRod').values**2
	itsig = np.ones(itt.shape)*tt_sig
	
	plt.subplot(211)
	plt.plot(ixx,itt/1e3,'.',color=cid[i_],label=SP_)

	# Get shallow structure model
	outputi,df_uDi,df_zi,df_Xi = run_WHB_lhs(ixx,itt,ixsig,itsig,n_draw=n_draw,KB79_ext=KB79_ext_bool,bounds=bounds)
	# Conduct post-processing and write shallow structure model to disk

	if KB79_ext_bool:
		idf_MOD,idf_beta = PP_WHB_write_outputs(OROOT,'Spread_%s_v5_ele_MK2_ptO3_sutured_GeoRod_KB_ext'%(SP_),\
												outputi['output'],df_uDi,df_zi,df_Xi,n_draw,KB79_ext=KB79_ext_bool,\
										 		dt=outputi['dt'],covdtdt=outputi['covdtdt'])
	else:
		idf_MOD,idf_beta = PP_WHB_write_outputs(OROOT,'Spread_%s_v5_ele_MK2_ptO3_sutured_GeoRod'%(SP_),\
												outputi,df_uDi,df_zi,df_Xi,n_draw,KB79_ext=KB79_ext_bool)
	# if SP_ == 'NS01':
	# 	breakpoint()
	k_ = 0
	for fmt,X_,Y_ in [('-','median u(z)','median z'),(':','Q025 u(z)','Q025 z'),(':','Q975 u(z)','Q975 z')]:
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


