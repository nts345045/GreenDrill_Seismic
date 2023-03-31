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
from tqdm import tqdm

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
	if nps[0][-1] == 'A':
		# Scrape experiment information from experiment file-name
		FM = nps[3] # Firn Model type
		FMQ = nps[2] # Firn Model Quantile perturbation used
		KD = 'all' # Data KinD used
		RES = nps[1] # Grid/Parameter Search RESolution
		DS = nps[6] # Data Slice
		SI = nps[0] # Step Number
		EX = 'Ex0' # Experiment Number
	elif nps[0][-1] == 'B':
		# Scrape experiment information from experiment file-name
		FM = nps[4] # Firn Model type
		FMQ = nps[3] # Firn Model Quantile perturbation used
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

	dmeta = {'Firn Model':FM,'Firn Model Quantile':FMQ,'Data Kind':KD,'Grid Resolution':RES,'Data Slice':DS,'Processing Step Number':SI,'Experiment':EX,'File':os.path.split(fname)[-1]}
	return df_mod, dmeta

def find_bestfit(df_mod,dmeta,method='res L2'):
	df_ = df_mod[df_mod[method]==df_mod[method].min()]
	dout = dmeta.copy()
	if df_.index == df_mod.index[0] or df_.index == df_mod.index[-1]:
		edge=True
	else:
		edge=False
	# TODO: have ZN Min/Max BOOL to say if the optimum is an edge effect or an actual cost minimum
	for k_ in df_.columns:
		dout.update({k_:df_[k_].values[0]})
	dout.update({'IsEdge':edge})
	return dout


# ROOT DIRECTORY
ROOT = os.path.join('..','..','..','..','..')
MROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# glob STRINGS
GSTR1 = os.path.join(MROOT,'velocity_models','structure_experiments','S5A*.csv')
GSTR2 = os.path.join(MROOT,'velocity_models','structure_experiments','S5B*K?.csv')
# Phase Travel-time v Offset data
DPHZ = os.path.join(MROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3.csv')
# Station locations
SITE = os.path.join(ROOT,'processed_data','Combined_SITE_Table_ELE_corr.csv')
# Handheld GPS Tracks
GPSc = os.path.join(ROOT,'processed_data','GPS','Prudhoe_Elevation_Corrected_GPS_Tracks.csv')

### DATA EXTRACTION ###

## Load SITE Locations ##
df_SITE = pd.read_csv(SITE,parse_dates=['Starttime','Endtime'])
df_SITE = df_SITE[df_SITE['Network']=='PL']
## Load GPSc Data ##
df_GPSc = pd.read_csv(GPSc,parse_dates=['time'],index_col=[0])


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
for i_ in tqdm(range(len(df_SUM))):
	mS_ = df_SUM.loc[i_,:]
	IND.append(mS_.name)
	# Subset Phase Data...
	# ...if using all data...
	if mS_['Data Slice']=='Average':
		pD_ = df_picks[df_picks['phz']=='S']

	# ...if using a whole-spread data slice...
	elif mS_['Data Slice'] in ['NS01','NS02','NS03','WE01','WE02','WE03']:	
		pD_ = df_picks[(df_picks['spread']==mS_['Data Slice'])&\
					   (df_picks['phz']=='S')&\
					   (df_picks['itype']=='GeoRod')]

	# ...if using a shot-specific data slice
	elif isinstance(mS_['Data Slice'],int):
		pD_ = df_picks[(df_picks['shot #']==mS_['Data Slice'])&\
					   (df_picks['phz']=='S')&\
					   (df_picks['itype']=='GeoRod')]

	# ...if there is pick-kind specific
	if mS_['Data Kind'] in [1,2]:
		pD_ = pD_[pD_['kind']==mS_['Data Kind']]

	# Calculate covariance matrix of included CMP coordinates
	CMPcov = pD_[['CMP mE','CMP mN','CMP mH']].cov().values
	# Compile output geometry line
	line = [pD_['CMP mE'].mean(),pD_['CMP mN'].mean(),pD_['CMP mH'].mean(),\
			pD_['CMP mE'].min(),pD_['CMP mN'].min(),pD_['CMP mH'].min(),\
			pD_['CMP mE'].max(),pD_['CMP mN'].max(),pD_['CMP mH'].max(),\
			pD_['CMP mE'].quantile(.5),pD_['CMP mN'].quantile(.5),pD_['CMP mH'].quantile(.5),\
			pD_['CMP mE'].quantile(.1),pD_['CMP mN'].quantile(.1),pD_['CMP mH'].quantile(.1),\
			pD_['CMP mE'].quantile(.9),pD_['CMP mN'].quantile(.9),pD_['CMP mH'].quantile(.9),\
			CMPcov[0,0],CMPcov[1,1],CMPcov[2,2],CMPcov[0,1],CMPcov[0,2],CMPcov[1,2]]

	CMP_stats.append(line)

df_CMP = pd.DataFrame(CMP_stats,columns=['mE mean','mN mean','mH mean',\
										 'mE max','mN max','mH max',\
										 'mE min','mN min','mH min',\
										 'mE med','mN med','mH med',\
										 'mE Q10','mN Q10','mH Q10',\
										 'mE Q90','mN Q90','mH Q90',\
										 'mE var','mN var','mH var',\
										 'mEmNcov','mEmHcov','mNmHcov'],\
					  index=IND)

df_M = pd.concat([df_SUM,df_CMP],axis=1,ignore_index=False)

plt.subplot(121)
plt.errorbar(df_M['mE mean'],df_M['mH mean'] - df_M['Z m'],\
			 xerr=df_M['mE var'].values**0.5,\
			 yerr=df_M['mH var'].values**0.5,\
			 fmt='.')
plt.scatter(df_M['mE mean'],df_M['mH mean'] - df_M['Z m'],c=df_M['mN mean'].values)

plt.plot(df_M['mE mean'],df_M['mH mean'],'ks')


plt.xlabel('Easting (Cross-Line) [m]')
plt.ylabel('Elevation [m ASL]')
# plt.ylim([df_M['Z m'].max() + 10,df_M['Z m'].min() - 10])
plt.legend()
plt.subplot(122)
plt.errorbar(df_M['mN mean'],df_M['mH mean'] - df_M['Z m'],
			 xerr=df_M['mN var'].values**0.5,\
			 yerr=df_M['mH var'].values**0.5,\
			 fmt='.')
plt.scatter(df_M['mN mean'],df_M['mH mean'] - df_M['Z m'],c=df_M['mE mean'].values)
plt.plot(df_M['mN mean'],df_M['mH mean'],'ks')

plt.xlabel('Northing (In-Line) [m]')
plt.ylabel('Elevation [m ASL]')
plt.legend()


for E_ in ['Ex1','Ex2']:
	plt.figure()
	for Q_ in ['mean','Q10','Q90']:
		df = df_M[(df_M['Grid Resolution']=='FINE')&\
				  (df_M['Firn Model Quantile']==Q_)&\
				  (df_M['Data Kind']==1)&\
				  (~df_M['IsEdge'].values)&\
				  (df_M['Experiment']==E_)]
		plt.subplot(121)
		plt.errorbar(df['mE '+Q_],df['mH '+Q_] - df['Z m'],\
					 xerr=df['mE var'].values**0.5,\
					 yerr=df['mH var'].values**0.5,\
					 fmt='.',label=Q_+" "+E_)
		plt.scatter(df['mE '+Q_],df['mH '+Q_] - df['Z m'],c=df['mN '+Q_].values)

		plt.plot(df['mE '+Q_],df['mH '+Q_],'ks')


		plt.xlabel('Easting (Cross-Line) [m]')
		plt.ylabel('Elevation [m ASL]')
		# plt.ylim([df['Z m'].max() + 10,df['Z m'].min() - 10])
		plt.legend()
		plt.subplot(122)
		plt.errorbar(df['mN '+Q_],df['mH '+Q_] - df['Z m'],
					 xerr=df['mN var'].values**0.5,\
					 yerr=df['mH var'].values**0.5,\
					 fmt='.',label=Q_+" "+E_)
		plt.scatter(df['mN '+Q_],df['mH '+Q_] - df['Z m'],c=df['mE '+Q_].values)
		plt.plot(df['mN '+Q_],df['mH '+Q_],'ks')

		plt.xlabel('Northing (In-Line) [m]')
		plt.ylabel('Elevation [m ASL]')
		plt.legend()
# plt.ylim([df['Z m'].max() + 10,df['Z m'].min() - 10])




plt.show()