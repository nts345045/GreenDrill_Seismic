"""
:module: S5_PostProcess_Vel_Structure_Tests.py
:prupose: Create a summary table of all vertical velocity structure experiments and incorporate 
			shot-receiver geometry measures from STEP1 to create pseudo-sections of the 
			bed reflector topography

:TODO:
Bring in Phase picks and create a geometric representation of sampled parts of the bed from 
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


### SUBROUTINES ###
def read_ZN_experiment(fname):
	"""
	Extract key experiment hyperparameters from ZN inversion file names
	generated in STEP4 and read the file into a pandas.DataFrame

	:: INPUTS ::
	:param fname: path/filename [str]

	:: OUTPUTS ::
	:return df_mod: [pandas.DataFrame] contents of fname
	:return dmeta: [dict] dictionary with key experimental hyperparameters
		--- KEYS ---
		FM = Firn Model [Average, or Spread]
		FMQ = Firn Model perturbation (Quantile sample from STEP3)
		KD = Kind of phase data [1 = first-break, 2 = phase-defining pick]
		RES = Resolution of grid-search mesh [COARSE, FINE, VFINE]
		DS = Data Slice ['all', Spread, or Shot #]
		SI = Experimental STEP# (S4A, S4B)
		EX = Experiment # (Ex0 for S4A, Ex1, Ex2)

	"""
	df_mod = pd.read_csv(fname)
	nps = os.path.split(fname)[-1].split('_')
	if nps[0] == 'S4A':
		# Scrape experiment information from experiment file-name
		FM = nps[3] # Firn Model type
		FMQ = nps[2] # Firn Model Quantile perturbation used
		KD = 'all' # Data KinD used
		RES = nps[1] # Grid/Parameter Search RESolution
		DS = nps[6] # Data Slice
		SI = nps[0] # Step Number
		EX = 'Ex0' # Experiment Number
	elif nps[0] == 'S4B':
		# Scrape experiment information from experiment file-name
		FM = nps[5] # Firn Model type
		FMQ = nps[4] # Firn Model Quantile perturbation used
		KD = nps[-1][1] # Data KinD used
		try:
			KD = int(KD)
		except:
			pass
		RES = nps[2] # Grid/Parameter Search RESolution
		DS = nps[-4] # Data Slice
		# Attempt to turn DS into an INT
		try:
			DS = int(DS)
		except:
			pass
		SI = nps[0] # Step Number
		EX = nps[1] # Experiment Number

	dmeta = {'FM':FM,'FMQ':FMQ,'KD':KD,'RES':RES,'DS':DS,'SI':SI,'EX':EX,'fname':os.path.split(fname)[-1]}
	return df_mod, dmeta

def find_bestfit(df_mod,dmeta,method='res L2'):
	df_ = df_mod[df_mod[method]==df_mod[method].min()]
	dout = dmeta.copy()
	# TODO: have ZN Min/Max BOOL to say if the optimum is an edge effect or an actual cost minimum
	for k_ in df_.columns:
		dout.update({k_:df_[k_].values[0]})
	return dout


# ROOT DIRECTORY
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# glob STRINGS
GSTR1 = os.path.join(ROOT,'velocity_models','structure_experiments','S4A*.csv')
GSTR2 = os.path.join(ROOT,'velocity_models','structure_experiments','S4B*K?.csv')
# Phase Travel-time v Offset data
DPHZ = os.path.join(ROOT,'VelCorrected_Phase_Picks_O2_idsw_v5.csv')
# SmartSolo station locations

# Handheld GPS Tracks

### DATA EXTRACTION ###

# Extract Aerial Average Modeling Results
flist1 = glob(GSTR1)

table1 = []
for f_ in flist1:
	df_mod, dmeta = read_ZN_experiment(f_)
	dout = find_bestfit(df_mod,dmeta,method='res L2')
	line = list(dout.values())
	table1.append(line)

df_SUM1 = pd.DataFrame(table1,columns=dout.keys())

# Extract Spread/Shot specific Modeling Results
flist2 = glob(GSTR2)
table2 = []
for f_ in flist2:
	df_mod, dmeta = read_ZN_experiment(f_)
	dout = find_bestfit(df_mod,dmeta,method='res L2')
	line = list(dout.values())
	table2.append(line)

df_SUM2 = pd.DataFrame(table2,columns=dout.keys()) 

# Load Phase Data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
df_picks = df_picks[df_picks['SRoff m'].notna()]
# Compile Model Summaries
df_SUM = pd.concat([df_SUM1,df_SUM2],axis=0,ignore_index=True)

# Iterate across each model and estimate the area of the bed reflection
# based on shot-receiver midpoint coordinates calculated in STEP1
CMP_stats = []; IND = []
for i_ in range(len(df_SUM)):
	mS_ = df_SUM.loc[i_,:]
	IND.append(mS_.name)
	# Subset Phase Data...
	# ...if using all data...
	if mS_['DS']=='Average':
		pD_ = df_picks[df_picks['phz']=='S']

	# ...if using a whole-spread data slice...
	elif mS_['DS'] in ['NS01','NS02','NS03','WE01','WE02','WE03']:	
		pD_ = df_picks[(df_picks['spread']==mS_['DS'])&\
					   (df_picks['phz']=='S')&\
					   (df_picks['itype']=='GeoRod')]

	# ...if using a shot-specific data slice
	elif isinstance(mS_['DS'],int):
		pD_ = df_picks[(df_picks['shot #']==mS_['DS'])&\
					   (df_picks['phz']=='S')&\
					   (df_picks['itype']=='GeoRod')]

	# ...if there is pick-kind specific
	if mS_['KD'] in [1,2]:
		pD_ = pD_[pD_['kind']==mS_['KD']]

	# Calculate covariance matrix of included CMP coordinates
	CMPcov = pD_[['CMP mE','CMP mN']].cov().values
	# Compile output geometry line
	line = [pD_['CMP mE'].mean(),pD_['CMP mN'].mean(),\
			pD_['CMP mE'].min(),pD_['CMP mN'].min(),\
			pD_['CMP mE'].max(),pD_['CMP mN'].max(),\
			pD_['CMP mE'].quantile(.5),pD_['CMP mN'].quantile(.5),\
			pD_['CMP mE'].quantile(.1),pD_['CMP mN'].quantile(.1),\
			pD_['CMP mE'].quantile(.9),pD_['CMP mN'].quantile(.9),\
			CMPcov[0,0],CMPcov[1,1],CMPcov[0,1]]

	CMP_stats.append(line)

df_CMP = pd.DataFrame(CMP_stats,columns=['mE mean','mN mean','mE max','mN max','mE min','mN min',\
										 'mE med','mN med','mE Q10','mN Q10','mE Q90','mN Q90',\
										 'mE var','mN var','mEmN cov'],\
					  index=IND)

df = pd.concat([df_SUM,df_CMP],axis=1,ignore_index=False)


plt.figure()
plt.subplot(121)
plt.errorbar(df['mE mean'],df['Z m'],xerr=df['mE var'].values**0.5,fmt='.')
plt.xlabel('Easting (Cross-Line) [m]')
plt.ylabel('Ice Thickness [m]')
plt.ylim([df['Z m'].max() + 10,df['Z m'].min() - 10])

plt.subplot(122)
plt.errorbar(df['mN mean'],df['Z m'],xerr=df['mN var'].values**0.5,fmt='.')
plt.xlabel('Northing (In-Line) [m]')
plt.ylabel('Ice Thickness [m]')
plt.ylim([df['Z m'].max() + 10,df['Z m'].min() - 10])




