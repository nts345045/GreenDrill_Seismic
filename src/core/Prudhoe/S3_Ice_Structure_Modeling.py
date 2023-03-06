"""
:module: S3_Vertical_Slowness_Modeling.py
:auth: Nathan T. Stevens
:Synopsis:
    Inputs: Compiled picks (from S1)
    Tasks: Conduct KB79 fitting and WHB inversion (with uncertainty quant) on:
            1) Whole data-set gather
            2) Spread gathers
            3) Shot gathers
           Estimate ice-column structure using WHB profiles
    Outputs: Gather-specific KB79 fits and vertical ice-structure models, model summary index

:: TODO ::

"""
import pandas as pd




# Add repository root to path & get repo modules of use
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.join('..','..'))
import util.KB_WHB_Inversion as kwi
import util.Firn_Density as fpz
import util.InvTools as inv
import util.Dix_1D_Raytrace_Analysis as d1d



##### SUPPORTING PROCESSES #####

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


def PP_write_outputs(OROOT,FN_start,output,df_uD,df_z,n_draw,full=False):
	# Write all 
	if full:
		df_uD.T.to_csv(os.path.join(OROOT,'%s_uD_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
		df_z.T.to_csv(os.path.join(OROOT,'%s_z_models_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
	# Write ODR model for KB79 to file
	df_beta = pd.DataFrame({'mean':output_full.beta,'a0':output_full.cov_beta[:,0],'a1':output_full.cov_beta[:,1],\
							'a2':output_full.cov_beta[:,2],'a3':output_full.cov_beta[:,3],'a4':output_full.cov_beta[:,4]},\
						    index=['a0','a1','a2','a3','a4'])
	df_beta.to_csv(os.path.join(OROOT,'%s_KB79_ODR.csv'%(FN_start)),header=True,index=True)
	# Get stats representations of each (Q25,Q50,Q75,mean,std)
	df_sum = pd.DataFrame({'mean u(z)':df_uD.mean(axis=0).values,'mean z':df_z.mean(axis=0).values,\
						   'std u(z)':df_uD.std(axis=0).values,'std z':df_z.std(axis=0).values,\
						   'median u(z)':df_uD.median(axis=0).values,'median z':df_z.median(axis=0).values,\
						   'Q10 u(z)':df_uD.quantile(.1,axis=0).values,'Q10 z':df_z.quantile(.1,axis=0).values,\
						   'Q90 u(z)':df_uD.quantile(.9,axis=0).values,'Q90 z':df_z.quantile(.9,axis=0).values})
	df_sum.to_csv(os.path.join(OROOT,'%s_WHB_ODR_LHSn%d.csv'%(FN_start,n_draw)),header=True,index=False)
	return df_sum,df_beta


def WHB_DIX_lhs(beta_a,beta_d,cov_a,cov_d,n_draw=100,min_sig2_mag=1e-12):
	"""
	Given parameter fits for diving wave data with the KB79 equation,
	fits for reflected wave data with a hyperbolic moveout equationc,
	and their covariance matrices, create a series of perturbed models for
	shallow velocity structure using the WHB inversion, and deep layer 
	interval velocity estimation using Dix conversion. Prior distributions
	are sampled with a Latin Hypercube Scheme informed by the mean vectors 
	(beta_*) and covariance matrices (cov_*) from model inputs. 

	Note:
	In this implementation model parameters from diving waves (*_a) and reflected waves
	(*_d) are assumed to be independent (i.e., 0-valued off-diagonals for the)
	full covariance matrix used for the LHS sampler.


	:: INPUTS ::
	:param beta_a: values of coefficients a1 -- a5 for Kirchner & Bentley (1979) double-exponential
	:param beta_d: values of ice thickness and RMS velocity for a hyperbolic reflector
	:param cov_a: covariance matrix estimated from KB79 fitting
	:param cov_d: covariance matrix estiamted from hyperbolic fitting
	:param n_draw: number of MCMC simulations to conduct
	:param min_sig2_mag: minimum magnitude of covariance matrix entries to be considered non-zero
						Rows/columns containing zero-valued diagonal values are removed and 
						average values of these parameters (from beta_a and beta_d) are passed
						to MCMC

	:: OUTPUTS ::
	:return Zm: Depth values for interval midpoints, shape = (n_draw,m) 
	:return Hm: Layer thickness values for each interval, shape = (n_draw,m)
	:return Um: Slowness values for each interval, shape = (n_draw,m)

	"""
	# Create full mean vector
	mv = np.append(beat_a,beta_d)
	# Create full covariance matrix - method developed with assistance from ChatGPT
	Cm =  np.block([[cov_a,np.zeros((5,2))],[np.zeros((2,5)),cov_d]])
	# Filter for near-0 valued covariances
	cIND = np.diag(np.abs(Cm)) >= min_sig2_mag
	# Create reduced model vector (guided by covariance matrix)
	mv_crunch = mv[cIND]
	# Create reduced model covariance matrix
	Cm_crunch = Cm[cIND,:][:,cIND]
	# DO ACTUAL LHS WORK (implementation from Stevens and others, 2022,J. Glac)
	samps = inv.norm_lhs(mv_crunch,Cm_crunch,n_samps=n_draw)
	print('Samples Drawn: %d'%(n_draw))
	# Create holders for outputs
	Zm,Hm,Um,=[],[],[]
	for i_ in tqdm(range(n_draw)):
		# Create perturbed parameter holder
		i_ref = np.zeros(7,)
		# Add in perturbed values
		i_ref[cIND] += samps[i_,:]
		# Add in 0-value locked values
		i_ref[~cIND] += mv[~cIND]
		# Run sanity check
		i_ref[i_ref < min_sig2_mag] = min_sig2_mag
		# Run WHB for shallow structure (short version - next v. provide control on kwargs beyond beta_)
		i_zDv = kwi.loop_WHB_int(4000,abcde=i_ref[:5])
		# Run (inverse) vRMS analysis (Dix conversion) using WHB structure as the layer-1 definition
		Vn = @HOLDER_FUN()


def run_full_WHB_DIX_lhs(df_picks,pfilts={'SRoff m':})





##### ACTUAL PROCESSING #####

#### PROCESSING CONTROLS ####
# Node location uncertainty in meters
Node_xSig = 6.
# Georod uncertainty in meters
GeoRod_xSig = 1.
# Phase pick time uncertainties in seconds
tt_sig = 1e-3
# Number of MCMC draws to conduct
n_draw = 10
# Array of ice-thicknesses to assess
Zv = np.arange(450,600,1)
dV, Vmax = 10, 3850
NMOkwargs = {'dV':dV,'Vmax':Vmax,'dx':10,'n_ref':1}
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
# Subset primary reflection arrivals of interest
sD_ = df_picks[(df_picks['phz']=='S')&(df_picks['SRoff m'].notna())&(df_picks['kind'].isin([1,2]))]


### RUN PROCESSING ON ENSEMBLE DATA ###
pxx = pD_['SRoff m'].values
sxx = sD_['SRoff m'].values
ptt = pD_['tt sec'].values*1000. # Put into Milliseconds
stt = sD_['tt sec'].values
# Create coordinate standard deviation based on travel-time pic
pxsig = Node_xSig*(pD_['itype']=='Node').values**2 + GeoRod_xSig*(pD_['itype']=='GeoRod').values**2
sxsig = Node_xSig*(sD_['itype']=='Node').values**2 + GeoRod_xSig*(sD_['itype']=='GeoRod').values**2
ptsig = np.ones(ptt.shape)*tt_sig
stsig = np.ones(stt.shape)*tt_sig
# Run WHB analysis 
output_full,df_uD,df_z = run_WHB_lhs(pxx,ptt,pxsig,ptsig,n_draw=n_draw)
# Write outputs from WHB
df_sum_full, df_beta_full = PP_write_outputs(OROOT,'Full_Data_v5',output_full,df_uD,df_z,n_draw)
# Conduct reflection grid-search with Dix to narrow target values
df_DIX_Q50 = d1d.hyperbolic_fitting(sxx,stt,Zv,df_sum_full['median u(z)'].values,df_sum_full['median z'].values,dV=1,Vmax=Vmax)
# Find best-fit model
IBEST50 = df_DIX_Q50['res L2']==df_DIX_Q50['res L2'].min()
S_DIX_Q50 = df_DIX_Q50[IBEST50]
# Compare to ODR analysis
output = d1d.hyperbolic_ODR(sxx,stt,sxsig,stsig)

breakpoint()

# Iterate across spreads
cid = ['blue','red','m','dodgerblue','g','orange']
if isplot:
	plt.figure()
	plt.subplot(222)
	plt.plot(1000*df_sum_full['Q10 u(z)'].values**-1,df_sum_full['Q10 z'],'k:')
	plt.plot(1000*df_sum_full['Q90 u(z)'].values**-1,df_sum_full['Q90 z'],'k:')
	plt.plot(1000*df_sum_full['median u(z)'].values**-1,df_sum_full['median z'],'k-',label='Full Data')
	plt.xlabel('WHB Compressional Velocity ($m/s$)')
	plt.ylabel('WHB Depth (mBGS)')
	plt.subplot(223)
	plt.plot(fpz.rho_robin(1000*df_sum_full['median u(z)'].values**-1),df_sum_full['median z'],'k-',label='Full Data')
	plt.plot(np.ones(2)*870,[-5,100],'k:')
	plt.xlabel('Robin Method Density ($kg/m^3$)')
	plt.ylabel('WHB Depth (mBGS)')
	plt.subplot(224)
	plt.plot(fpz.rho_kohnen(1000*df_sum_full['median u(z)'].values**-1,1000*df_sum_full['median u(z)'].values[-1]**-1),df_sum_full['median z'],'k-.',\
							label='Full v=%d m/s'%(1000*df_sum_full['median u(z)'].values[-1]**-1))
	plt.plot(fpz.rho_kohnen(1000*df_sum_full['median u(z)'].values**-1),df_sum_full['median z'],'k--',label='Full v=3850 m/s')
	plt.plot(np.ones(2)*870,[-5,100],'k:')
	plt.xlabel('Kohnen Method Density ($kg/m^3$)')
	plt.ylabel('WHB Depth (mBGS)')

for i_,SP_ in enumerate(pD_['spread'].unique()):
	# Plot Group Model

	print(SP_)
	# Subset data and uncertainties
	pIND = pD_['spread']==SP_
	sIND = sD_['spread']==iS_
	ipD_ = pD_[pIND]; ipxsig = xsig[pIND]; iptsig = tsig[pIND]
	isD_ = sD_[sIND]; isxsig = xsig[sIND]; istsig = tsig[sIND]

	ipxx = ipD_['SRoff m'].values
	iptt = ipD_['tt sec'].values*1000. # Put into Milliseconds
	if isplot:
		plt.subplot(221)
		plt.plot(ipxx,iptt,'.',color=cid[i_],label=SP_,ms=1)#,alpha=0.25)
		plt.plot(isxx,istt,'.',color=cid[i_],ms=1)

	### Generate spread-specific model
	# Get shallow structure model
	outputi,df_uDi,df_zi = run_WHB_lhs(ipxx,iptt,ipxsig,iptsig,n_draw=n_draw)
	# Conduct post-processing and write shallow structure model to disk
	df_sum_i,df_beta_i = PP_write_outputs(OROOT,'Spread_%s'%(SP_),outputi,df_uDi,df_zi,n_draw)
	# Conduct reflection grid-search with Dix to narrow target values
	idf_DIX_Q50 = d1d.hyperbolic_fitting(isxx,istt,Zv,df_sum_full['median u(z)'].values,df_sum_full['median z'].values,dV=1,Vmax=Vmax)
	# Find best-fit model
	IBEST50 = df_DIX_Q50['res L2']==df_DIX_Q50['res L2'].min()
	iS_DIX_Q50 = idf_DIX_Q50[IBEST50]

	# Plotting stuff
	vpi = 1e3/df_sum_i['median u(z)'].values
	zi = df_sum_i['median z'].values
	if isplot:
		plt.subplot(222)
		plt.plot(1000*df_sum_i['Q10 u(z)'].values**-1,df_sum_i['Q10 z'],':',color=cid[i_])
		plt.plot(1000*df_sum_i['Q90 u(z)'].values**-1,df_sum_i['Q90 z'],':',color=cid[i_])
		plt.plot(vpi,zi,'-',color=cid[i_],label='Spread %s'%(SP_))


		plt.subplot(223)
		plt.plot(fpz.rho_robin(vpi),zi,'-',color=cid[i_],label='Spread %s'%(SP_))

		

		plt.subplot(224)
		plt.plot(fpz.rho_kohnen(1000*df_sum_i['median u(z)'].values**-1,1000*df_sum_i['median u(z)'].values[-1]**-1),zi,'-.',\
								color=cid[i_],label='%s v=%d m/s'%(SP_,1000*df_sum_i['median u(z)'].values[-1]**-1))
		plt.plot(fpz.rho_kohnen(1000*df_sum_i['median u(z)'].values**-1,),zi,'--',label='%s v=3850 m/s'%(SP_),color=cid[i_])


if isplot:
	for i_ in [222,223,224]:
		plt.subplot(i_)
		plt.legend()
		plt.ylim([100,-5])


plt.show()


# ##### END #####



# def run_WHB_DC_lhs(xx,tt,xsig,tsig,fit_type=0,sum_rule='trap',n_draw=100,min_sig2_mag = 1e-12):
# 	"""
# 	Run a Wichert-Herglotz-Bateman inversion on supplied travel-time vs offset data 
# 	considering data uncertainties and propagating uncertainties through the inversion
# 	using a Bayesian approach with a latin hypercube samplier (lhs)
	
# 	:: INPUTS ::
# 	:param xx: source-receiver offsets in meters
# 	:param tt: travel-times of direct/diving wave arrivals in seconds
# 	:param xsig: source-receiver offset uncertainties in meters
# 	:param tsig: travel-time uncertainties in seconds
# 	:param fit_type: solution control for scipy.odr.ODR - 	0 = ODR
# 															1 = ODR + full output
# 															2 = LSQ
# 	:param sum_rule: summation rule to approximate the WHB integral
# 	:param n_draw: number of samples to draw from the prior distribution
# 	:param min_sig2_mag: Minimum acceptable magnitude

# 	:: OUTPUTS ::
# 	:return output: output from ODR fitting 
# 	:return uDn: [m,n_draw] array of slowness values from WHB realizations
# 	:return zn: [m,n_draw] array of depth values from WHB realizations
# 	"""
# 	# Do unweighted inversion first to get first estimate of KB79 parameters
# 	try:
# 		beta1,cov_beta1 = kwi.curvefit_KB79(xx,tt)
# 	except RuntimeError:
# 		breakpoint()
# 	print('(LSQ) beta0 -- %.2e %.2e %.2e %.2e %.2e'%(beta1[0],beta1[1],beta1[2],beta1[3],beta1[4]))
# 	# Do inverse-variance-weighted Orthogonal Distance Regression estimate for KB79
# 	output = kwi.ODR_KB79(xx,tt,xsig,tsig,beta0=beta1,fit_type=fit_type)
# 	beta2 = output.beta
# 	print('(ODR) beta1 -- %.2e %.2e %.2e %.2e %.2e'%(beta2[0],beta2[1],beta2[2],beta2[3],beta2[4]))
# 	# Create LHS realizations from model parameters
# 	# Filter for 0-valued covariances
# 	cIND = np.abs(output.sd_beta) >= min_sig2_mag
# 	# Create crunched mean vector and covariance matrix for LHS sampling
# 	um_crunch = output.beta[cIND]
# 	Cm_crunch = output.cov_beta[cIND,:][:,cIND]
# 	# breakpoint()
# 	samps = inv.norm_lhs(um_crunch,Cm_crunch,n_samps=n_draw)
# 	print('LHS: %d samples drawn'%(n_draw))
# 	# Iterate over LHS samples and run WHB
# 	z_mods = [];u_mods = []; mod_ind = np.arange(0,n_draw)
# 	for i_ in tqdm(range(n_draw)):
# 		# Create perturbed parameter holder
# 		i_samp = np.zeros(5,)
# 		# Add in perturbed values
# 		i_samp[cIND] += samps[i_,:]
# 		# Add in values that had ~0-valued covariances
# 		i_samp[~cIND] += output.beta[~cIND]
# 		# Sanity check that i_samp contains no 0-valued entries
# 		i_samp[i_samp < min_sig2_mag] = min_sig2_mag
# 		# breakpoint() # Check that i_samp is (5,) or (5,1) or (1,5)
# 		i_zDv = kwi.loop_WHB_int(np.nanmax(xx)+1.,abcde=i_samp,sig_rule=sum_rule)
# 		# Get bottom-profile velocity
# 		vNm1 = 1e3/i_zDv['uD ms/m'][-1]
# 		# Iterate across ice-column slownesses
# 		for uN_ in np.arange(vNm1*0.9,3950,dVN):
# 			# Downsample overburden velocity profile
# 			imod_OB = dix.resample_WHB(i_zDv['z m'],i_zDv['uD ms/m']*1e-3)
# 			# Generate 1D velocity model
# 			iMOD = v1d.generate_layercake_vel(Vv=[1e3/imod_OB['uRMS'],Zv=np.append([imod_OB['Ztop'],4000]))
# 			# Iterate across source-depths and model travel times
# 			for zN_ in zNv:
# 				# Calculate travel-times for 10m grid
# 				ixx = np.arange(np.min(xx),np.max(xx)+10,10)
# 				itt = v1d.raytrace_explicit(iMOD,ixx,zN_)
# 				itt_hat = np.interp(xx,ixx,itt)
# 				# Calculate residuals
				

# 				# Append to 2-layer holder arrays ::TODO::



	# Iterate across shots
	# for j_,SH_ in enumerate(iD_['shot #'].unique()):
	# 	JND = iD_['shot #']==SH_
	# 	jD_ = iD_[JND]
	# 	jxx = jD_['SRoff m'].values
	# 	jtt = jD_['tt sec'].values
	# 	outputj, df_uDj, df_zj = run_WHB_lhs(jxx,jtt,ixsig[JND],itsig[JND],n_draw=n_draw)
	# 	df_sum_j,df_beta_j = PP_write_outputs(OROOT,'Spread_%s_Shot_%d'%(SP_,SH_),outputj,df_uDj,df_zj,n_draw)
	# 	if isplot:
	# 		plt.plot(df_sum_j['Q25 u(z)'].values**-1,df_sum_j['Q25 z'],':',color=cid[i_])
	# 		plt.plot(df_sum_j['Q75 u(z)'].values**-1,df_sum_j['Q75 z'],':',color=cid[i_])
	# 		plt.plot(df_sum_j['median u(z)'].values**-1,df_sum_j['median z'],'-',label='Shot %s'%(SH_))
	# except RuntimeError:
		# breakpoint()

# 		z_mods.append(i_zDv['z m'])
# 		u_mods.append(i_zDv['uD ms/m'])

# 	# breakpoint() # Check dimensionality of z_mods and u_mods before putting in index/columns
# 	df_uD = pd.DataFrame(u_mods,index=mod_ind)
# 	df_z = pd.DataFrame(z_mods,index=mod_ind)

# 	return output,df_uD,df_z



# df_uD.T.to_csv(os.path.join(OROOT,'Full_Data_uD_models_LHSn%d.csv')%(n_draw),header=True,index=False)
# df_z.T.to_csv(os.path.join(OROOT,'Full_Data_z_models_LHSn%d.csv')%(n_draw),header=True,index=False)
# df_beta = pd.DataFrame({'mean':output_full.beta,'a0':output_full.cov_beta[:,0],'a1':output_full.cov_beta[:,1],\
# 						'a2':output_full.cov_beta[:,2],'a3':output_full.cov_beta[:,3],'a4':output_full.cov_beta[:,4]},\
# 					    index=['a0','a1','a2','a3','a4'])
# df_beta.






# ### LOAD PHASE DATA, CLEAN, & SUBSET BY PHASE ###
# df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# D_ = df_picks[(df_picks['kind']==1) & (df_picks['SRoff m'].notna())]
# pD_ = D_[D_['phz']=='P'] # Direct-arriving waves





# PD_instructions = (df_PD,'Prudhoe_Dome','PD',300,600)

# # Create Holder for output models
# MODS = []
# #### ITERATE ACROSS SPREAD ####
# for i_,SP_ in enumerate(df_picks['spread'].unique()):

	
# 	### ITERATE ACROSS SHOT GATHERS ###
# 	for j_,SH_ in enumerate(df_picks['shot #'].unique()):
# 		### Slice DataFrame for subset

# 		### CONDUCT KB'79, 






# breakpoint()

# df_DIX_Q10 = d1d.hyperbolic_fitting(sxx,stt,Zv,df_sum_full['Q10 u(z)'].values,df_sum_full['Q10 z'].values,dV=1,Vmax=Vmax)
# df_DIX_Q90 = d1d.hyperbolic_fitting(sxx,stt,Zv,df_sum_full['Q90 u(z)'].values,df_sum_full['Q90 z'].values,dV=1,Vmax=Vmax)
# Get best-fit model

# # IBEST10 = df_DIX_Q10['res L2']==df_DIX_Q10['res L2'].min()
# # S_DIX_Q10 = df_DIX_Q10[IBEST10]
# # IBEST90 = df_DIX_Q90['res L2']==df_DIX_Q90['res L2'].min()
# # S_DIX_Q90 = df_DIX_Q90[IBEST90]
# plt.figure()
# plt.plot(sxx,stt,'ko',label='data')
# plt.plot(sxx,d1d.hyperbolic_tt(sxx,S_DIX_Q50['Z m'].values,S_DIX_Q50['Vrms'].values),'r.',label='model')
# # plt.plot(sxx,d1d.hyperbolic_tt(sxx,S_DIX_Q10['Z m'].values,S_DIX_Q10['Vrms'].values),'r.',label='model')
# # plt.plot(sxx,d1d.hyperbolic_tt(sxx,S_DIX_Q90['Z m'].values,S_DIX_Q90['Vrms'].values),'r.',label='model')
# plt.show()

# breakpoint()
# # Refine depth estimate using ray-tracing
# df_NMO_Q10 = d1d.raytracing_gridsearch(sxx,stt,Zv,df_sum_full['Q10 u(z)'].values,df_sum_full['Q10 z'].values,**NMOkwargs)
# df_NMO_Q50 = d1d.raytracing_gridsearch(sxx,stt,Zv,df_sum_full['median u(z)'].values,df_sum_full['median z'].values,**NMOkwargs)
# df_NMO_Q90 = d1d.raytracing_gridsearch(sxx,stt,Zv,df_sum_full['Q90 u(z)'].values,df_sum_full['Q90 z'].values,**NMOkwargs)



# def run_WHB_analysis(df_picks,XX=5e3,dx=1.,sig_rule='trap',p0=np.ones(5),bounds=(0,np.inf),output='model only'):
# 	"""
# 	Run whole Wiechert-Herglotz-Bateman analysis including Kirchner & Bently (1979) model
# 	fitting

# 	:: INPUTS ::
# 	:param df_picks: DataFrame containing analyst picks with columns 'SRoff m','tt sec', and 'kind'
# 	:param XX: Maximum Source-Receiver offset at which to include data and evaluate with WHB
# 	:param dx: mesh discretization for WHB integral approximation
# 	:param sig_rule: integral approximation rule to use ('trap','left','right')
# 	:param p0: initial parameter guesses for KB79 model fitting
# 	:param bounds: bounds for KB79 parameter estimation
# 	:param output: output type 	'model only': returns just the vertical slowness model
# 								'full': returns all outputs as a dictionary
# 	:: OUTPUTS ::
# 	:return model: dictionary with fields 'X' (max offset), 'z' (turning depth), 'uD' apparent slowness at depth
# 	:return IND: Filtering index for df_picks
# 	:return popt: parameter fits for the KB79 model
# 	:return pcov: parameter covariance matrix for the KB79 model
# 	:return pres: data - model residual vector 
# 	"""
# 	df_ = df_picks.copy().sort_values('SRoff m')
# 	IND = (df_['SRoff m'].notna()) &\
# 		  (df_['SRoff m'] <= XX)
# 	# Extract desired values
# 	xx = df_[IND]['SRoff m'].values
# 	tt = df_[IND]['tt sec'].values
# 	# Conduct initial model fitting without uncertainties
# 	abcde,cov_abcde = kwi.curvefit_KB79(xx,tt,p0=p0,bounds=bounds)
# 	# Conduct ODR fitting that incorporates uncertainties
# 	output = kwi.ODR_KB79(xx,tt,xsig,tsig,beta0=abcde)
# 	# Get Model Residuals
# 	res = tt - kwi.KB79_exp_fun(xx,*output[])
# 	z_uDv = whb.loop_WHB_int(XX,dx=dx,abcde=abcde,sig_rule=sig_rule)
# 	if output.lower() == 'model only':
# 		return z_uDv
# 	elif output.lower() == 'full':
# 		return {'model':z_uDv,'IND':IND,'popt':abcde,'pcov':cov_abcde,'pres':res}


# ##### DRIVERS #####
# def vertical_profile_driver(df_picks,Zv):
# 	"""
# 	Given a pre-filtered slice of the pick data (df_picks) and a vector of 

# 	:: INPUTS ::
# 	:param df_picks: pre-filtered (sliced) DataFrame of picks to be considered for analysis
# 	:param Zv: vector of ice-thickness values to test for grid-search of layer N
# 	:param uv: vector of slowness values to test for grid-search of layer N

# 	:: OUTPUTS ::
	


# 	"""
# 	# Run KB'79 model fitting and 
# 	WHB_out = run_WHB_analysis(df_picks,XX=5e3,dx=1.,output='full')

# 	vmod_rs = dix.resample_WHB(WHB_out['model']['z m'],WHB_out['model']['uD sec/m'],\
# 								method='incremental increase',scalar=1.3)
# 	xx = df_picks['SRoff'].values
# 	tt = df_picks['tt sec'].values
# 	df_DIX = dix.hyperbolic_fit(xx,tt,Zv,df_WHB)