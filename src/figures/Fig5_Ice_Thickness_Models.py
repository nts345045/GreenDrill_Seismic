"""
Fig5_Ice_Thickness_Models.py

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


## DATA LOADING ##
# Phases and Amplitudes
df_PHZ = pd.read_csv(DPHZ,parse_dates=['time'])
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
WE_rng = [490,540]
NS_rng = [495,535]
# Initialize Plot
plt.figure(figsize=(8,6.3))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
# Plot Crossing Points
ax1.plot([mEx,mEx],WE_rng,'r-',linewidth=2,alpha=0.5)
ax1.text(mEx,min(WE_rng) + 0.05*(max(WE_rng) - min(WE_rng)),'N-S',\
		 fontweight='extra bold',color='r',ha='center',va='center')
ax2.plot([mNx,mNx],NS_rng,'r-',linewidth=2,alpha=0.5)
# ax2.plot([min(df_VZN_1_u)])
ax2.text(mNx,min(NS_rng) + 0.05*(max(NS_rng) - min(NS_rng)),'W-E',\
		 fontweight='extra bold',color='r',ha='center',va='center')
## ICE THICKNESS MODELS ##
IND_WE = df_VZN_1_u['Data Slice'].astype(int).values > 350
IND_NS = df_VZN_1_u['Data Slice'].astype(int).values <= 350
# Bed Machine Total Thickness
ax1.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Ice Thickness'],'k-',linewidth=3,alpha=0.5,label='BedMachine $z_{tot}$')
ax1.fill_between(df_BM_WE['UTM19N mE'],df_BM_WE['Ice Thickness'].values - 30,df_BM_WE['Ice Thickness'].values + 30,color='k',alpha=0.1)
ax2.plot(df_BM_NS['UTM19N mN'],df_BM_NS['Ice Thickness'],'k-',linewidth=3,alpha=0.5,label='BedMachine $z_{tot}$')
ax2.fill_between(df_BM_NS['UTM19N mN'],df_BM_NS['Ice Thickness'].values - 30,df_BM_NS['Ice Thickness'].values + 30,color='k',alpha=0.1)

# Uniform firn model inversion results
ax1.errorbar(df_VZN_1_u[IND_WE]['mE mean'].values,df_VZN_1_u[IND_WE]['Z m'],\
			 xerr=[df_VZN_1_u[IND_WE]['mE mean'].values - df_VZN_1_m[IND_WE]['mE min'].values,\
			       df_VZN_1_M[IND_WE]['mE max'].values - df_VZN_1_u[IND_WE]['mE mean'].values],\
			 yerr=[df_VZN_1_u[IND_WE]['Z m'].values - df_VZN_1_m[IND_WE]['Z m'].values,\
			 	   df_VZN_1_M[IND_WE]['Z m'].values - df_VZN_1_u[IND_WE]['Z m'].values],\
			 	   fmt='.',capsize=5,label='$z_{tot}$(Uniform Firn)',color='dodgerblue')

ax2.errorbar(df_VZN_1_u[IND_NS]['mN mean'].values,df_VZN_1_u[IND_NS]['Z m'],\
			 xerr=[df_VZN_1_u[IND_NS]['mN mean'].values - df_VZN_1_m[IND_NS]['mN min'].values,\
			       df_VZN_1_M[IND_NS]['mN max'].values - df_VZN_1_u[IND_NS]['mN mean'].values],\
			 yerr=[df_VZN_1_u[IND_NS]['Z m'].values - df_VZN_1_m[IND_NS]['Z m'].values,\
			 	   df_VZN_1_M[IND_NS]['Z m'].values - df_VZN_1_u[IND_NS]['Z m'].values],\
			 	   fmt='.',capsize=5,label='$z_{tot}$(Uniform Firn)',color='dodgerblue')

## ICE THICKNESS MODELS ##
IND_WE = df_VZN_2_u['Data Slice'].astype(int).values > 350
IND_NS = df_VZN_2_u['Data Slice'].astype(int).values <= 350


# Laterially varying firn model inversion results
ax1.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,df_VZN_2_u[IND_WE]['Z m'],\
			 xerr=[df_VZN_2_u[IND_WE]['mE mean'].values - df_VZN_2_m[IND_WE]['mE min'].values,\
			       df_VZN_2_M[IND_WE]['mE max'].values - df_VZN_2_u[IND_WE]['mE mean'].values],\
			 yerr=[df_VZN_2_u[IND_WE]['Z m'].values - df_VZN_2_m[IND_WE]['Z m'].values,\
			 	   df_VZN_2_M[IND_WE]['Z m'].values - df_VZN_2_u[IND_WE]['Z m'].values],\
			 	   fmt='.',capsize=5,label='$z_{tot}$(Laterally Varying Firn)',color='orange')

ax2.errorbar(df_VZN_2_u[IND_NS]['mN mean'].values,df_VZN_2_u[IND_NS]['Z m'],\
			 xerr=[df_VZN_2_u[IND_NS]['mN mean'].values - df_VZN_2_m[IND_NS]['mN min'].values,\
			       df_VZN_2_M[IND_NS]['mN max'].values - df_VZN_2_u[IND_NS]['mN mean'].values],\
			 yerr=[df_VZN_2_u[IND_NS]['Z m'].values - df_VZN_2_m[IND_NS]['Z m'].values,\
			 	   df_VZN_2_M[IND_NS]['Z m'].values - df_VZN_2_u[IND_NS]['Z m'].values],\
			 	   fmt='.',capsize=5,label='$z_{tot}$(Laterally Varying Firn)',color='orange')


ax1.set_xlabel('Easting (UTM 19N) [m]')
ax1.set_ylabel('Ice Thickness [m]')
ax2.set_xlabel('Northing (UTM 19N) [m]')
ax2.set_ylabel('Ice Thickness [m]')

ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
ax1.text(df_BM_WE['UTM19N mE'].min()-20,WE_rng[1]-4,'a',\
	fontweight='extra bold',fontstyle='italic',fontsize=16)
ax2.text(df_BM_NS['UTM19N mN'].min()-20,NS_rng[1]-4,'b',\
	fontweight='extra bold',fontstyle='italic',fontsize=16)

ax1.text(df_BM_WE['UTM19N mE'].min(),WE_rng[0]+20,'W',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax1.text(df_BM_WE['UTM19N mE'].max(),WE_rng[0]+20,'E',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')

ax2.text(df_BM_NS['UTM19N mN'].min(),NS_rng[0]+20,'S',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax2.text(df_BM_NS['UTM19N mN'].max(),NS_rng[0]+20,'N',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')


plt.show()


# ax3.plot(df_BM_WE['UTM19N mE'].values,df_BM_WE['Surface Elevation'].values - df_FIR['Q975 z'].max(),'c:',linewidth=2,label='Shallow Profile Bottom')
# ax3.fill_between(df_BM_WE['UTM19N mE'].values,df_BM_WE['Surface Elevation'].values - zRtm,\
# 				 df_BM_WE['Surface Elevation'].values - zRtM,color='c',alpha=0.25,label='Robin Close-Off')
# ax3.fill_between(df_BM_WE['UTM19N mE'].values,df_BM_WE['Surface Elevation'].values - zKtm,\
# 				 df_BM_WE['Surface Elevation'].values - zKtM,alpha=0.25,label='Kohnen Close-Off',color='dodgerblue')

# ax4.plot(df_BM_NS['UTM19N mN'].values,df_BM_NS['Surface Elevation'].values - df_FIR['Q975 z'].max(),'c:',linewidth=2,label='Shallow Profile Bottom')
# ax4.fill_between(df_BM_NS['UTM19N mN'].values,df_BM_NS['Surface Elevation'].values - zRtm,\
# 				 df_BM_NS['Surface Elevation'].values - zRtM,color='c',alpha=0.25,label='Robin Close-Off')
# ax4.fill_between(df_BM_NS['UTM19N mN'].values,df_BM_NS['Surface Elevation'].values - zKtm,\
# 				 df_BM_NS['Surface Elevation'].values - zKtM,alpha=0.25,label='Kohnen Close-Off',color='dodgerblue')







# # Ice Thickness Models
# marks = {'Ex0':'.','Ex1':'o','Ex2':'s'}
# # colors = {1:'k',2:'r'}
# colors = {'Ex0':'k','Ex1':'b','Ex2':'r'}
# lines = {'COARSE':':','FINE':'--','VFINE':'-'}
# for DK_ in [1]:
# 	for EX_,GR_ in df_VZN[['Experiment','Grid Resolution']].value_counts().index:

# 		for S_,D_ in ODICT.items():	
# 			if D_['IsEdge'].sum(axis=0) == 0 and S_[-1]==DK_ and S_[0]==EX_:
# 				ax1.plot(D_.loc['mean','mE']*np.ones(3),D_['mH'] - D_['Z m'],colors[EX_]+marks[EX_]+lines[GR_])
# 				ax1.plot(D_['mE'],(D_.loc['mean','mH'] - D_.loc['mean','Z m'])*np.ones(3),colors[EX_]+marks[EX_]+lines[GR_])
# 				ax2.plot(D_.loc['mean','mN']*np.ones(3),D_['mH'] - D_['Z m'],colors[EX_]+marks[EX_]+lines[GR_])
# 				ax2.plot(D_['mN'],(D_.loc['mean','mH'] - D_.loc['mean','Z m'])*np.ones(3),colors[EX_]+marks[EX_]+lines[GR_])
# 				ax3.plot(D_.loc['mean','mE']*np.ones(3),D_['Z m'],colors[EX_]+marks[EX_]+lines[GR_])
# 				ax3.plot(D_['mE'],D_.loc['mean','Z m']*np.ones(3),colors[EX_]+marks[EX_]+lines[GR_])
# 				ax4.plot(D_.loc['mean','mN']*np.ones(3),D_['Z m'],colors[EX_]+marks[EX_]+lines[GR_])
# 				ax4.plot(D_['mN'],D_.loc['mean','Z m']*np.ones(3),colors[EX_]+marks[EX_]+lines[GR_])
