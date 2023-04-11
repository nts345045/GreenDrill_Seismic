"""
Fig10_bcd_Drillsite_Vertical_Profiles.py
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from pyproj import Proj

sys.path.append('..')
import util.Firn_Density as fdu

# Primary Root Path
ROOT = os.path.join('..','..','..','..')
# GIS Data Root
GIROOT = os.path.join(ROOT,'gis','Contrib')
BDNS = os.path.join(GIROOT,'BedMachineExtracts_NS.csv')
BDWE = os.path.join(GIROOT,'BedMachineExtracts_WE.csv')
# Processed Data Root
DROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# Wiechert-Herglotz-Bateman Reference Model
UDAT = os.path.join(DROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_WHB_ODR_LHSn100.csv')
# Ice Thickness Models
MROOT = os.path.join(DROOT,'velocity_models','structure_summary')
MSUM = os.path.join(MROOT,'Ice_Thickness_Models.csv')
# Subsetting to Ex1 (uniform firn, shot-wise thickness models, type-1 results)
GMST = os.path.join(MROOT,'Ex1_Average_[34]??_1_summary.csv')
flist = glob(GMST)
# Finalized picks with polarity 
DPHZ = os.path.join(DROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_Amps_RPOL.csv')
# Get station table
SITE = os.path.join(ROOT,'processed_data','Combined_SITE_Table_ELE_corr.csv')
# Get shot locations
SHOT = os.path.join(ROOT,'processed_data','Active_Seismic','Master_Shot_Record_QCd_ELE_corr.csv')
# Get Proposed Drillsites
DSF = os.path.join(ROOT,'gis','Developed_Content','Prudhoe_Proposed_Drillsites_UTM19N.csv')

## DATA LOADING ##
# Stations
df_SITE = pd.read_csv(SITE)
# Shots
df_SHOT = pd.read_csv(SHOT)
# Phases and Amplitudes
df_PHZ = pd.read_csv(DPHZ,parse_dates=['time'])


# Drillsites
df_DRILL = pd.read_csv(DSF)


# Filter Site for only stations with data
stachan = list(df_SITE[['Station','Channel']].value_counts().index)
datstachan = list(df_PHZ[['sta','chan']].value_counts().index)
sta_filt = []
for i_ in range(len(df_SITE)):
	sta = df_SITE.loc[i_,'Station']
	chan = df_SITE.loc[i_,'Channel']
	if (sta,chan) in datstachan:
		sta_filt.append(True)
	else:
		sta_filt.append(False)

df_SITE = df_SITE[sta_filt]

# Depth Models
df_VZN = pd.read_csv(MSUM)
# Filter depth models
df_VZN_1 = df_VZN[(df_VZN['Experiment']=='Ex1')&(df_VZN['Data Kind']==1)&\
				(df_VZN['Grid Resolution']=='VFINE')]
df_VZN_2 = df_VZN[(df_VZN['Experiment']=='Ex2')&(df_VZN['Data Kind']==1)&\
				(df_VZN['Grid Resolution']=='VFINE')]
# Filter into firn model subsets
df_VZN_1_m = df_VZN_1[df_VZN_1['Firn Model Quantile']=='Q025'].sort_values('Data Slice')
df_VZN_1_M = df_VZN_1[df_VZN_1['Firn Model Quantile']=='Q975'].sort_values('Data Slice')
df_VZN_1_u = df_VZN_1[df_VZN_1['Firn Model Quantile']=='mean'].sort_values('Data Slice')
# Filter into firn model subsets
df_VZN_2_m = df_VZN_2[df_VZN_2['Firn Model Quantile']=='Q025'].sort_values('Data Slice')
df_VZN_2_M = df_VZN_2[df_VZN_2['Firn Model Quantile']=='Q975'].sort_values('Data Slice')
df_VZN_2_u = df_VZN_2[df_VZN_2['Firn Model Quantile']=='mean'].sort_values('Data Slice')


# Firn Model
df_FIR = pd.read_csv(UDAT)

# Calculate Firn Density Profiles
pzR = fdu.rho_robin(1e3/df_FIR['Q025 u(z)'].values)
zRtM = df_FIR.loc[np.argmin(np.abs(pzR-870)),'Q975 z']
zRtm = df_FIR.loc[np.argmin(np.abs(pzR-870)),'Q025 z']
pzK = fdu.rho_robin(1e3/df_FIR['Q025 u(z)'].values)
zKtM = df_FIR.loc[np.argmin(np.abs(pzK-870)),'Q975 z']
zKtm = df_FIR.loc[np.argmin(np.abs(pzK-870)),'Q025 z']


# BedMachine Surfaces Models
df_BM_NS = pd.read_csv(BDNS)
df_BM_WE = pd.read_csv(BDWE)

# Convert Coordinates as Needed
proj1 = Proj('epsg:3413')
proj2 = Proj('epsg:32619')
lon,lat = proj1(df_BM_NS['EPSG3413 East'].values,df_BM_NS['EPSG3413 North'].values,inverse=True)
mE,mN = proj2(lon,lat)
df_BM_NS = pd.concat([df_BM_NS,pd.DataFrame({'UTM19N mE':mE,'UTM19N mN':mN},index=df_BM_NS.index)],axis=1,ignore_index=False)
lon,lat = proj1(df_BM_WE['EPSG3413 East'].values,df_BM_WE['EPSG3413 North'].values,inverse=True)
mE,mN = proj2(lon,lat)
df_BM_WE = pd.concat([df_BM_WE,pd.DataFrame({'UTM19N mE':mE,'UTM19N mN':mN},index=df_BM_WE.index)],axis=1,ignore_index=False)

mEx,mNx = 458230,8670850
full_rng = [800,1400]
bed_rng = [820,910]

IND_WE = df_VZN_1_u['Data Slice'].astype(int).values > 350
IND_NS = df_VZN_1_u['Data Slice'].astype(int).values <= 350

Emin = df_VZN['mE min'].min() - 300
Emax = df_VZN['mE max'].max() + 100

Nmin = df_VZN['mN min'].min()
Nmax = df_VZN['mN max'].max() + 300

# Initialize Plot
plt.figure(figsize=(7.5,7.5))
ax1 = plt.subplot(222)
ax2 = plt.subplot(223)
ax3 = plt.subplot(224)


# Plot DrillSites
ax1.plot(df_DRILL['UTM19N mE'].values[0]*np.ones(2),[860,1400],color='red',linewidth=2)
ax2.plot(df_DRILL['UTM19N mE'].values[1]*np.ones(2),[854,1400],color='red',linewidth=2)
ax3.plot(df_DRILL['UTM19N mN'].values[2]*np.ones(2),[877,1400],color='red',linewidth=2)



# ax1.plot([mEx,mEx],full_rng,'r-',linewidth=2,alpha=0.5)
# ax1.text(mEx,min(full_rng) + 0.9*(max(full_rng) - min(full_rng)),'N-S',\
# 		 fontweight='extra bold',color='r',ha='center',va='center')
# ax2.plot([mNx,mNx],full_rng,'r-',linewidth=2,alpha=0.5)
# ax2.text(mNx,min(full_rng) + 0.9*(max(full_rng) - min(full_rng)),'W-E',\
# 		 fontweight='extra bold',color='r',ha='center',va='center')


# Plot BedMachine Data
ax1.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Surface Elevation'],'k-',\
		 linewidth=3,alpha=0.5,label='BedMachine $H_{surf}$')
ax1.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Surface Elevation']-df_BM_WE['Ice Thickness'],'-',\
		 linewidth=3,alpha=0.5,color='brown',label='BedMachine $H_{bed}$')
ax1.fill_between(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'].values - 30,df_BM_WE['Bed Elevation'].values + 30,color='brown',alpha=0.1)

ax2.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Surface Elevation'],'k-',\
		 linewidth=3,alpha=0.5,label='BedMachine $H_{surf}$')
ax2.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Surface Elevation']-df_BM_WE['Ice Thickness'],'-',\
		 linewidth=3,alpha=0.5,color='brown',label='BedMachine $H_{bed}$')
ax2.fill_between(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'].values - 30,df_BM_WE['Bed Elevation'].values + 30,color='brown',alpha=0.1)

ax3.plot(df_BM_NS['UTM19N mN'],df_BM_NS['Surface Elevation'],'k-',linewidth=3,alpha=0.5)
ax3.plot(df_BM_NS['UTM19N mN'],df_BM_NS['Surface Elevation']-df_BM_NS['Ice Thickness'],'-',linewidth=3,alpha=0.5,color='brown')
ax3.fill_between(df_BM_NS['UTM19N mN'],df_BM_NS['Bed Elevation'].values - 30,df_BM_NS['Bed Elevation'].values + 30,color='brown',alpha=0.1)



# Plot Bed Elevations from Seismics
Z_WE_1 = df_VZN_1_u[IND_WE]['mH mean']-df_VZN_1_u[IND_WE]['Z m']

Z_NS_1 = df_VZN_1_u[IND_NS]['mH mean']-df_VZN_1_u[IND_NS]['Z m']

patch_bounds_WE_1 = [df_VZN_1_u[IND_WE]['mE mean'].values - df_VZN_1_m[IND_WE]['mE min'].values,\
       			df_VZN_1_M[IND_WE]['mE max'].values - df_VZN_1_u[IND_WE]['mE mean'].values]

patch_bounds_NS_1 = [df_VZN_1_u[IND_NS]['mN mean'].values - df_VZN_1_m[IND_NS]['mN min'].values,\
       			df_VZN_1_M[IND_NS]['mN max'].values - df_VZN_1_u[IND_NS]['mN mean'].values]

CI95_bounds_WE_1 = [((df_VZN_1_u[IND_WE]['Z m'].values - df_VZN_1_m[IND_WE]['Z m'].values)**2 +\
				  df_VZN_1_u[IND_WE]['mH var'].values*1.96**2)**0.5,\
			 	 ((df_VZN_1_M[IND_WE]['Z m'].values - df_VZN_1_u[IND_WE]['Z m'].values)**2 +\
			 	  df_VZN_1_u[IND_WE]['mH var'].values*1.96**2)**0.5]

CI95_bounds_NS_1 = [((df_VZN_1_u[IND_NS]['Z m'].values - df_VZN_1_m[IND_NS]['Z m'].values)**2 +\
				  df_VZN_1_u[IND_NS]['mH var'].values*1.96**2)**0.5,\
			 	 ((df_VZN_1_M[IND_NS]['Z m'].values - df_VZN_1_u[IND_NS]['Z m'].values)**2 +\
			 	  df_VZN_1_u[IND_NS]['mH var'].values*1.96**2)**0.5]

ax1.errorbar(df_VZN_1_u[IND_WE]['mE mean'].values,Z_WE_1,xerr=patch_bounds_WE_1,yerr=CI95_bounds_WE_1,\
			 fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')

ax2.errorbar(df_VZN_1_u[IND_WE]['mE mean'].values,Z_WE_1,xerr=patch_bounds_WE_1,yerr=CI95_bounds_WE_1,\
			 fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')

ax3.errorbar(df_VZN_1_u[IND_NS]['mN mean'].values,Z_NS_1,xerr=patch_bounds_NS_1,yerr=CI95_bounds_NS_1,\
			 fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')

Z_WE_2 = df_VZN_2_u[IND_WE]['mH mean']-df_VZN_2_u[IND_WE]['Z m']

Z_NS_2 = df_VZN_2_u[IND_NS]['mH mean']-df_VZN_2_u[IND_NS]['Z m']

patch_bounds_WE_2 = [df_VZN_2_u[IND_WE]['mE mean'].values - df_VZN_2_m[IND_WE]['mE min'].values,\
       			df_VZN_2_M[IND_WE]['mE max'].values - df_VZN_2_u[IND_WE]['mE mean'].values]

patch_bounds_NS_2 = [df_VZN_2_u[IND_NS]['mN mean'].values - df_VZN_2_m[IND_NS]['mN min'].values,\
       			df_VZN_2_M[IND_NS]['mN max'].values - df_VZN_2_u[IND_NS]['mN mean'].values]

CI95_bounds_WE_2 = [((df_VZN_2_u[IND_WE]['Z m'].values - df_VZN_2_m[IND_WE]['Z m'].values)**2 +\
				  df_VZN_2_u[IND_WE]['mH var'].values*1.96**2)**0.5,\
			 	 ((df_VZN_2_M[IND_WE]['Z m'].values - df_VZN_2_u[IND_WE]['Z m'].values)**2 +\
			 	  df_VZN_2_u[IND_WE]['mH var'].values*1.96**2)**0.5]

CI95_bounds_NS_2 = [((df_VZN_2_u[IND_NS]['Z m'].values - df_VZN_2_m[IND_NS]['Z m'].values)**2 +\
				  df_VZN_2_u[IND_NS]['mH var'].values*1.96**2)**0.5,\
			 	 ((df_VZN_2_M[IND_NS]['Z m'].values - df_VZN_2_u[IND_NS]['Z m'].values)**2 +\
			 	  df_VZN_2_u[IND_NS]['mH var'].values*1.96**2)**0.5]

ax1.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,Z_WE_2,xerr=patch_bounds_WE_2,yerr=CI95_bounds_WE_2,\
			 fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')

ax2.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,Z_WE_2,xerr=patch_bounds_WE_2,yerr=CI95_bounds_WE_2,\
			 fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')

ax3.errorbar(df_VZN_2_u[IND_NS]['mN mean'].values,Z_NS_2,xerr=patch_bounds_NS_2,yerr=CI95_bounds_NS_2,\
			 fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')




# ax1.errorbar(df_VZN_1_u[IND_WE]['mE mean'].values,df_VZN_1_u[IND_WE]['mH mean']-df_VZN_1_u[IND_WE]['Z m'],\
# 			 xerr=[df_VZN_1_u[IND_WE]['mE mean'].values - df_VZN_1_m[IND_WE]['mE min'].values,\
# 			       df_VZN_1_M[IND_WE]['mE max'].values - df_VZN_1_u[IND_WE]['mE mean'].values],\
# 			 yerr=[df_VZN_1_u[IND_WE]['Z m'].values - df_VZN_1_m[IND_WE]['Z m'].values,\
# 			 	   df_VZN_1_M[IND_WE]['Z m'].values - df_VZN_1_u[IND_WE]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')
# ax1.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,df_VZN_2_u[IND_WE]['mH mean']-df_VZN_2_u[IND_WE]['Z m'],\
# 			 xerr=[df_VZN_2_u[IND_WE]['mE mean'].values - df_VZN_2_m[IND_WE]['mE min'].values,\
# 			       df_VZN_2_M[IND_WE]['mE max'].values - df_VZN_2_u[IND_WE]['mE mean'].values],\
# 			 yerr=[df_VZN_2_u[IND_WE]['Z m'].values - df_VZN_2_m[IND_WE]['Z m'].values,\
# 			 	   df_VZN_2_M[IND_WE]['Z m'].values - df_VZN_2_u[IND_WE]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')

# ax2.errorbar(df_VZN_1_u[IND_WE]['mE mean'].values,df_VZN_1_u[IND_WE]['mH mean']-df_VZN_1_u[IND_WE]['Z m'],\
# 			 xerr=[df_VZN_1_u[IND_WE]['mE mean'].values - df_VZN_1_m[IND_WE]['mE min'].values,\
# 			       df_VZN_1_M[IND_WE]['mE max'].values - df_VZN_1_u[IND_WE]['mE mean'].values],\
# 			 yerr=[df_VZN_1_u[IND_WE]['Z m'].values - df_VZN_1_m[IND_WE]['Z m'].values,\
# 			 	   df_VZN_1_M[IND_WE]['Z m'].values - df_VZN_1_u[IND_WE]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')
# ax2.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,df_VZN_2_u[IND_WE]['mH mean']-df_VZN_2_u[IND_WE]['Z m'],\
# 			 xerr=[df_VZN_2_u[IND_WE]['mE mean'].values - df_VZN_2_m[IND_WE]['mE min'].values,\
# 			       df_VZN_2_M[IND_WE]['mE max'].values - df_VZN_2_u[IND_WE]['mE mean'].values],\
# 			 yerr=[df_VZN_2_u[IND_WE]['Z m'].values - df_VZN_2_m[IND_WE]['Z m'].values,\
# 			 	   df_VZN_2_M[IND_WE]['Z m'].values - df_VZN_2_u[IND_WE]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')

# ax3.errorbar(df_VZN_1_u[IND_NS]['mN mean'].values,df_VZN_1_u[IND_NS]['mH mean']-df_VZN_1_u[IND_NS]['Z m'],\
# 			 xerr=[df_VZN_1_u[IND_NS]['mN mean'].values - df_VZN_1_m[IND_NS]['mN min'].values,\
# 			       df_VZN_1_M[IND_NS]['mN max'].values - df_VZN_1_u[IND_NS]['mN mean'].values],\
# 			 yerr=[df_VZN_1_u[IND_NS]['Z m'].values - df_VZN_1_m[IND_NS]['Z m'].values,\
# 			 	   df_VZN_1_M[IND_NS]['Z m'].values - df_VZN_1_u[IND_NS]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')
# ax3.errorbar(df_VZN_2_u[IND_NS]['mN mean'].values,df_VZN_2_u[IND_NS]['mH mean']-df_VZN_2_u[IND_NS]['Z m'],\
# 			 xerr=[df_VZN_2_u[IND_NS]['mN mean'].values - df_VZN_2_m[IND_NS]['mN min'].values,\
# 			       df_VZN_2_M[IND_NS]['mN max'].values - df_VZN_2_u[IND_NS]['mN mean'].values],\
# 			 yerr=[df_VZN_2_u[IND_NS]['Z m'].values - df_VZN_2_m[IND_NS]['Z m'].values,\
# 			 	   df_VZN_2_M[IND_NS]['Z m'].values - df_VZN_2_u[IND_NS]['Z m'].values],\
# 			 	   fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='orange')






plt.show()