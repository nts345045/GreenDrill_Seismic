"""
Fig7_Bed_Elevation_Profiles.py
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
ROOT = os.path.join('..','..','..','..','..')
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

Ex1_INTERP = os.path.join(MROOT,'Uniform_Firn_Bed_Elevation_Model_QCd.csv')
Ex2_INTERP = os.path.join(MROOT,'Varying_Firn_Bed_Elevation_Model_QCd.csv')


df_Ex1 = pd.read_csv(Ex1_INTERP)
df_Ex2 = pd.read_csv(Ex2_INTERP)

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
full_rng = [800,1400]
bed_rng = [820,910]

IND_WE = df_VZN_1_u['Data Slice'].astype(int).values > 350
IND_NS = df_VZN_1_u['Data Slice'].astype(int).values <= 350

# Emin = df_VZN['mE min'].min()
# Emax = df_VZN['mE max'].max()

Emin,Emax = df_Ex1[df_Ex1['Transect']=='WE']['mE min'].min(),\
			df_Ex1[df_Ex1['Transect']=='WE']['mE max'].max()

# Nmin = df_VZN['mN min'].min()
# Nmax = df_VZN['mN max'].max()

Nmin,Nmax = df_Ex1[df_Ex1['Transect']=='NS']['mN min'].min(),\
			df_Ex1[df_Ex1['Transect']=='NS']['mN max'].max()

Z1min,Z1max = df_Ex1['Q025 H bed (mASL)'].min(),\
			  df_Ex1['Q975 H bed (mASL)'].max()

Z2min,Z2max = df_Ex2['Q025 H bed (mASL)'].min(),\
			  df_Ex2['Q975 H bed (mASL)'].max()

Zmin = np.min([Z1min,Z2min])
Zmax = np.max([Z1max,Z2max])


plt.figure(figsize=(7.5,7.5))


ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)


ax1.plot([mEx,mEx],bed_rng,'r-',linewidth=2,alpha=0.5)
ax1.text(mEx,Zmax,'N-S',\
		 fontweight='extra bold',color='r',ha='center',va='center')

ax2.plot([mEx,mEx],bed_rng,'r-',linewidth=2,alpha=0.5)
ax2.text(mEx,Zmax,'N-S',\
		 fontweight='extra bold',color='r',ha='center',va='center')

ax3.plot([mNx,mNx],bed_rng,'r-',linewidth=2,alpha=0.5)
ax3.text(mNx,Zmax,'W-E',\
		 fontweight='extra bold',color='r',ha='center',va='center')

ax4.plot([mNx,mNx],bed_rng,'r-',linewidth=2,alpha=0.5)
ax4.text(mNx,Zmax,'W-E',\
		 fontweight='extra bold',color='r',ha='center',va='center')
# Plot BedMachine Data
ax1.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'],'-',linewidth=3,alpha=0.5,color='brown',label='BedMachine $H_{bed}$')
ax1.fill_between(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'].values - 30,df_BM_WE['Bed Elevation'].values + 30,color='brown',alpha=0.1)
ax2.plot(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'],'-',linewidth=3,alpha=0.5,color='brown',label='BedMachine $H_{bed}$')
ax2.fill_between(df_BM_WE['UTM19N mE'],df_BM_WE['Bed Elevation'].values - 30,df_BM_WE['Bed Elevation'].values + 30,color='brown',alpha=0.1)
ax3.plot(df_BM_NS['UTM19N mN'],df_BM_NS['Bed Elevation'],'-',linewidth=3,alpha=0.5,color='brown')
ax3.fill_between(df_BM_NS['UTM19N mN'],df_BM_NS['Bed Elevation'].values - 30,df_BM_NS['Bed Elevation'].values + 30,color='brown',alpha=0.1)
ax4.plot(df_BM_NS['UTM19N mN'],df_BM_NS['Bed Elevation'],'-',linewidth=3,alpha=0.5,color='brown')
ax4.fill_between(df_BM_NS['UTM19N mN'],df_BM_NS['Bed Elevation'].values - 30,df_BM_NS['Bed Elevation'].values + 30,color='brown',alpha=0.1)



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

ax3.errorbar(df_VZN_1_u[IND_NS]['mN mean'].values,Z_NS_1,xerr=patch_bounds_NS_1,yerr=CI95_bounds_NS_1,\
			 fmt='.',capsize=5,label='$H_{bed}$(Uniform Firn)',color='dodgerblue')

# for i_,Z_ in enumerate(Z_WE_1):
# 	ax1.text(df_VZN_1_u[IND_WE]['mE mean'].values[i_],Z_,df_VZN_1_u[IND_WE]['Data Slice'].values[i_])
# for i_,Z_ in enumerate(Z_NS_1):
# 	ax3.text(df_VZN_1_u[IND_NS]['mN mean'].values[i_],Z_,df_VZN_1_u[IND_NS]['Data Slice'].values[i_])


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

ax2.errorbar(df_VZN_2_u[IND_WE]['mE mean'].values,Z_WE_2,xerr=patch_bounds_WE_2,yerr=CI95_bounds_WE_2,\
			 fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')

ax4.errorbar(df_VZN_2_u[IND_NS]['mN mean'].values,Z_NS_2,xerr=patch_bounds_NS_2,yerr=CI95_bounds_NS_2,\
			 fmt='.',capsize=5,label='$H_{bed}$(Laterally Varying Firn)',color='orange')


# Plot interpreted profiles
fmt_order = ['r--','k-','r--']

f_ = 0
for fld_,Q_ in [('Upper Bound','Q975 H bed (mASL)'),('Interpreted Profile','Bed Elevation (mASL)'),('Lower Bound','Q025 H bed (mASL)')]:
	df_Ex1_flt = df_Ex1[(df_Ex1[fld_].values)&(df_Ex1['Transect']=='WE')].sort_values('mE mean')
	XX = df_Ex1_flt['mE mean']
	YY = df_Ex1_flt[Q_]
	ax1.plot(XX,YY,fmt_order[f_],label=fld_)
	f_ += 1

ax1.set_xlim([df_Ex1[df_Ex1['Transect']=='WE']['mE min'].min() - 50,\
			  df_Ex1[df_Ex1['Transect']=='WE']['mE max'].max() + 50])

f_ = 0
for fld_,Q_ in [('Upper Bound','Q975 H bed (mASL)'),('Interpreted Profile','Bed Elevation (mASL)'),('Lower Bound','Q025 H bed (mASL)')]:
	df_Ex1_flt = df_Ex1[(df_Ex1[fld_].values)&(df_Ex1['Transect']=='NS')].sort_values('mN mean')
	XX = df_Ex1_flt['mN mean']
	YY = df_Ex1_flt[Q_]
	ax3.plot(XX,YY,fmt_order[f_],label=fld_)
	f_ += 1

f_ = 0
for fld_,Q_ in [('Upper Bound','Q975 H bed (mASL)'),('Interpreted Profile','Bed Elevation (mASL)'),('Lower Bound','Q025 H bed (mASL)')]:
	df_Ex2_flt = df_Ex2[(df_Ex2[fld_].values)&(df_Ex2['Transect']=='WE')].sort_values('mE mean')
	XX = df_Ex2_flt['mE mean']
	YY = df_Ex2_flt[Q_]
	ax2.plot(XX,YY,fmt_order[f_],label=fld_)
	f_ += 1

f_ = 0
for fld_,Q_ in [('Upper Bound','Q975 H bed (mASL)'),('Interpreted Profile','Bed Elevation (mASL)'),('Lower Bound','Q025 H bed (mASL)')]:
	df_Ex2_flt = df_Ex2[(df_Ex2[fld_].values)&(df_Ex2['Transect']=='NS')].sort_values('mN mean')
	XX = df_Ex2_flt['mN mean']
	YY = df_Ex2_flt[Q_]
	ax4.plot(XX,YY,fmt_order[f_],label=fld_)
	f_ += 1

# for i_,Z_ in enumerate(Z_WE_2):
# 	ax2.text(df_VZN_2_u[IND_WE]['mE mean'].values[i_],Z_,df_VZN_2_u[IND_WE]['Data Slice'].values[i_])
# for i_,Z_ in enumerate(Z_NS_2):
# 	ax4.text(df_VZN_2_u[IND_NS]['mN mean'].values[i_],Z_,df_VZN_2_u[IND_NS]['Data Slice'].values[i_])



ax1.set_xlabel('UTM 19N Easting [m]')
ax2.set_xlabel('UTM 19N Easting [m]')
ax1.set_ylabel('Elevation [m ASL]')
# ax2.set_ylabel('Elevation [m ASL]')

ax3.set_xlabel('UTM 19N Northing [m]')
ax4.set_xlabel('UTM 19N Northing [m]')
ax3.set_ylabel('Elevation [m ASL]')
# ax4.set_ylabel('Elevation [m ASL]')


ax1.set_xlim([Emin - 300,Emax + 200])
ax3.set_xlim([Nmin - 200,Nmax + 200])
ax2.set_xlim([Emin - 300,Emax + 200])
ax4.set_xlim([Nmin - 200,Nmax + 200])

ax1.set_ylim([Zmin - 2.5, Zmax + 2.5])
ax2.set_ylim([Zmin - 2.5, Zmax + 2.5])
ax3.set_ylim([Zmin - 2.5, Zmax + 2.5])
ax4.set_ylim([Zmin - 2.5, Zmax + 2.5])
# ax1.legend()

ax1.text(Emin-200,Zmax-0.75,'a',fontweight='extra bold',fontstyle='italic',fontsize=16)
ax3.text(Nmin-100,Zmax-0.75,'c',fontweight='extra bold',fontstyle='italic',fontsize=16)
ax2.text(Emin-200,Zmax-0.75,'b',fontweight='extra bold',fontstyle='italic',fontsize=16)
ax4.text(Nmin-100,Zmax-0.75,'d',fontweight='extra bold',fontstyle='italic',fontsize=16)

ax1.text(Emin-200,Zmax-10,'W',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax1.text(Emax+100,Zmax-10,'E',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')

ax2.text(Emin-200,Zmax-10,'W',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax2.text(Emax+100,Zmax-10,'E',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')

ax3.text(Nmin-100,Zmax-10,'S',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax3.text(Nmax+100,Zmax-10,'N',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')

ax4.text(Nmin-100,Zmax-10,'S',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')
ax4.text(Nmax+100,Zmax-10,'N',\
	fontweight='extra bold',fontstyle='italic',fontsize=16,color='r',\
	ha='center',va='center')



plt.show()