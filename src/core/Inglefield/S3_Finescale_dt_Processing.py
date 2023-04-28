"""
:module: S3_Finescale_dt_Processing.py
:purpose: Use overlaps in spread/shot source-receiver offsets to provide additional origin time corrections.
		  General strategy: Starting from longest offset (N = 4)
		  	1) Find the apparent slowness and intercept time of diving waves for offset code N
		  	2) Find the apparent slowness and intercept time of diving waves for offset code N - 1
		  	3) Apply the difference in the Sx + t0 models as a time-correction factor for dataset N - 1
		  	4) Repeat 1-3 with N = 3 (using time-corrected version), and N = 2
		  	5) For offset codes 0 and 1 we have to deal with nonlinear moveouts from shallow structure
		  		Use a quadratic or cubic model for 
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add repository root to path & get repo modules of use
sys.path.append(os.path.join('..','..'))
import util.InvTools as inv
import util.KB_WHB_Inversion as kwi


### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Inglefield_Land')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3.csv')
OFILE = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')
issave = True
isplot = True
# Read in Phase Picks
df = pd.read_csv(DPHZ,parse_dates=['time'])
# Subset to GeoRods for use in analyses
df_G = df[df['itype']=='GeoRod']
# Subset out Node entries for concatenation at the end
df_N = df[df['itype']=='Node']

# Define Geo-Rod Location & Pick Uncertainties
GeoRod_xSig = 1.
tt_sig = 1e-3

# Define model to use for suturing data
stitch_mod = inv.lin_fun
beta0 = [1./3850,0]

# Kirchner & Bentley (1979) Model Fitting Parameters
bounds = ((0,0,0,0,0,-30),(np.inf,np.inf,np.inf,np.inf,np.inf,30))

plotlevel = 5

# Create DataFrame holder for completely updated travel-times
df_OUT = pd.DataFrame()
UNPROCESSED = []
## Iterate Across Spreads for geo-rods
for SP_ in df_G['spread'].unique():
	# Create holder for spread-specific corrected data
	df_SP = pd.DataFrame()
	# Subset down to spread-level data
	df_J = df_G[df_G['spread']==SP_]
	# # Make sets for iterating over
	# sets = df_J[['offset code','shot #']].value_counts().sort_index(ascending=False).index
	# Create indexer for which pairwise iteration this is
	I_ = 0
	# Iterate across unique pairs
	for OC_ in df_J.sort_values('offset code',ascending=False)['offset code'].unique():
	# for OC_,SH_ in sets:
		# If first pairwise comparison, define data subset
		if I_ == 0:
			idf = df_J[df_J['offset code']==OC_]
			# Append this "reference" subset of data to the spread-wise summary 
			df_SP = pd.concat([df_SP,idf],axis=0,ignore_index=False)
			# print('df_SP is %d long'%(len(df_SP)))

		# Otherwise, define idf from the last iteration
		elif I_ > 0 and len(jdf) > 0:
			idf = jdf.copy()
		# else:
		# 	breakpoint()
		# Now subset data to be updated
		jdf = df_J[df_J['offset code']==(OC_-1)]
		if len(jdf['shot #'].unique()) > 1:
			breakpoint()
		# Sanity check that we have data
		if len(idf) > 0 and len(jdf) > 0:
			# Sanity check for multiple shots with the same offset code
			# and iterate each pair
			for SH_ in jdf['shot #'].unique():
				print('{} {} {} {}'.format(I_,SP_,OC_,SH_))
				# Pull subset for unique shot #'s
				jdf_ = jdf[jdf['shot #']==SH_]
				# Subset diving wave offsets to suture zone
				xi = idf[(idf['kind']==1) & (idf['phz']=='P') &\
						 (idf['SRoff m'] <= jdf['SRoff m'].max())]\
						 ['SRoff m'].values
				xj = jdf[(jdf['kind']==1) & (jdf['phz']=='P') &\
						 (jdf['SRoff m'] >= idf['SRoff m'].min())]\
						 ['SRoff m'].values
				# Subset diving wave travel times to suture zone
				ti = idf[(idf['kind']==1) & (idf['phz']=='P') &\
						 (idf['SRoff m'] <= jdf['SRoff m'].max())]\
						 ['tt sec'].values
				tj = jdf[(jdf['kind']==1) & (jdf['phz']=='P') &\
						 (jdf['SRoff m'] >= idf['SRoff m'].min())]\
						 ['tt sec'].values
				# Compose uncertaintiy vectors
				xisig = np.ones(xi.shape)*GeoRod_xSig**2
				tisig = np.ones(ti.shape)*tt_sig
				xjsig = np.ones(xj.shape)*GeoRod_xSig**2
				tjsig = np.ones(tj.shape)*tt_sig
				# Fit model to suture-zone 
				out_j = inv.curve_fit_2Derr(stitch_mod,xj,tj,xjsig,tjsig,beta0=beta0)
				# Find model values at reference offsets
				ti_hat = stitch_mod(out_j.beta,xi)
				# Calculate average DT value for suture
				dt_hat = np.mean(ti - ti_hat)
				# Failsafe switch if the dt is unrealistically large
				if np.abs(dt_hat) > 0.1:
					breakpoint()
				if isplot and plotlevel > 1:
					plt.figure()
					plt.title('{} {} {}-{} {}:{}'.format(SP_,SH_,OC_,OC_-1,idf['shot #'].unique(),jdf['shot #'].unique()))
					plt.plot(idf['SRoff m'],idf['tt sec'],'b.',label=idf['shot #'].unique()[0])
					plt.plot(jdf['SRoff m'],jdf['tt sec'],'k.',label=jdf['shot #'].unique()[0])
					# plt.plot(xi,ti,'bo')
					# plt.plot(xj,tj,'k*')
					plt.plot(xj,tj + dt_hat,'rv',label='GeoRod,Shifted Tiepoints')

				# Update times for jdf
				if np.isfinite(dt_hat):
					jdf.loc[jdf['shot #']==SH_,'tt sec'] = jdf.loc[jdf['shot #']==SH_,'tt sec'].values + dt_hat
				else:
					breakpoint()
				if isplot and plotlevel > 1:
					plt.plot(jdf['SRoff m'],jdf['tt sec'],'r.',label='GeoRod,Updated')
					plt.legend()
			df_SP = pd.concat([df_SP,jdf],axis=0,ignore_index=False)
			# print('df_SP is %d long'%(len(df_SP)))
			I_ += 1
		elif OC_ > 1:
			UNPROCESSED.append((SP_,OC_-2))

### Conduct an (extended) Kirchner & Bentley (1979) fitting to sutured
### diving wave t(x) data to help re-align data with Nodes
	df_I = df_N[(df_N['spread']==SP_)&(df_N['phz']=='P')&(df_N['kind']==1)&(df_N['SRoff m']>3)]
	XI = df_I['SRoff m'].values
	TI = df_I['tt sec'].values
	df_J = df_SP[(df_SP['phz']=='P')&(df_SP['kind']==1)]
	XJ = df_J['SRoff m'].values
	TJ = df_J['tt sec'].values

	poptJ,pcovJ = kwi.curvefit_KB79_ext(XJ,TJ,bounds=bounds)
	TI_HAT = kwi.KB79_exp_ext_fun(XI,*poptJ)
	T0DT = np.mean(TI - TI_HAT)
	if isplot:
		plt.figure()
		plt.title('Spread {} | dt: {} sec'.format(SP_,T0DT))
		plt.plot(df_I['SRoff m'],df_I['tt sec'],'b.',label='Node,Raw')
		plt.plot(df_J['SRoff m'],df_J['tt sec'],'k.',label='GeoRod,Raw')
		# plt.plot(XI,TI,'bo',label='Node,Tiepoints')
		# plt.plot(XJ,TJ,'k*',)
		# plt.plot(XJ,TJ + T0DT,'rv')

	df_SP.loc[:,'tt sec'] += T0DT

	if isplot:
		plt.plot(df_SP['SRoff m'],df_SP['tt sec'],'r.',label='GeoRod,Shifted')
		plt.legend()
	# Put everything together
	df_OUT = pd.concat([df_OUT,df_SP],axis=0,ignore_index=False)	
	print('OUT is %d long'%(len(df_OUT)))
	

# Stick in "unprocessable" data from NS01 and stick with previous corrections done in S2_Time_Geom_Processing.py	
for SP_,OC_ in UNPROCESSED:
	df_U = df_G[(df_G['spread']==SP_)&(df_G['offset code']==OC_)]
	df_OUT = pd.concat([df_OUT,df_U],axis=0,ignore_index=False)
	print('OUT is %d long'%(len(df_OUT)))




#### SAVE IT ####
if issave:
	df_OUT.to_csv(OFILE,header=True,index=False)


if isplot:
	plt.show()




# breakpoint()







# # If Offset Code > 2, use linear models (+ sanity checks)
# if OC_ > 2 and len(jdf_) > 0:
# 	out_j = inv.curve_fit_2Derr(inv.lin_fun,xj,tj,xjsig,tjsig,beta0=[3850**-1,0])
# 	ti_hat = inv.lin_fun(out_j.beta,xi)
# 	dt_hat = np.mean(ti - ti_hat)
# # If offset Code <= 2, use cubic models
# elif OC_ <= 2 and len(jdf_) > 0:
# 	popt,pcov = kwi.curvefit_KB79_ext(xj,tj)
# 	ti_hat = kwi.KB79_ext_odr_fun(xi,*popt)
# 	dt_hat = np.mean(ti - ti_hat)


####
# If Offset Code <= 2, use cubic models




	# ### Conduct an (extended) Kirchner & Bentley (1979) fitting to sutured
	# ### diving wave t(x) data to help re-align data with Nodes
	# df_I = df_N[(df_N['spread']==SP_)&(df_N['phz']=='P')&(df_N['kind']==1)&(df_N['SRoff m']>3)]
	# XI = df_I['SRoff m'].values
	# TI = df_I['tt sec'].values
	# df_J = df_SP[(df_SP['phz']=='P')&(df_SP['kind']==1)]
	# XJ = df_J['SRoff m'].values
	# TJ = df_J['tt sec'].values

	# poptJ,pcovJ = kwi.curvefit_KB79_ext(XJ,TJ,bounds=bounds)
	# TI_HAT = kwi.KB79_exp_ext_fun(XI,*poptJ)
	# T0DT = np.mean(TI - TI_HAT)
	# if plotlevel > 0:
	# 	plt.figure()
	# 	plt.title('Spread {} | dt: {} sec'.format(SP_,T0DT))
	# 	plt.plot(df_I['SRoff m'],df_I['tt sec'],'b.',label='Node,Raw')
	# 	plt.plot(df_J['SRoff m'],df_J['tt sec'],'k.',label='GeoRod,Raw')
	# 	# plt.plot(XI,TI,'bo',label='Node,Tiepoints')
	# 	# plt.plot(XJ,TJ,'k*',)
	# 	# plt.plot(XJ,TJ + T0DT,'rv')

	# df_SP.loc[:,'tt sec'] += T0DT

	# if plotlevel > 0:
	# 	plt.plot(df_SP['SRoff m'],df_SP['tt sec'],'r.',label='GeoRod,Corrected')
	# 	plt.legend()
	# 	plt.xlabel('Source-Receiver Offset [m]')
	# 	plt.ylabel('Travel Time [sec]')
	# # Put everything together