"""
GD_JGLAC_Fig2_waveforms_and_tx_data.py

:AUTH: Nathan T. Stevens
:EMAIL: nts5045@psu.edu
:PURPOSE: Display waveform examples and travel time data from
		  Prudhoe Dome (a-b) and Inglefield Margin (c-d) sites.

"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, Stream
from glob import glob

### USER CONTROL SECTION ###
filt_then_tar = False


PD_shot_ID = 314
PD_spread = 'NS01'
PD_CdBs = 40
PD_ftype = 'bandpass'
PD_fkw = {'freqmin':1,'freqmax':500}
# PD_fkw = {''}
IM_shot_ID = 423
IM_spread = 'NS01'
IM_ftype = 'bandpass'#'lowpass'
IM_fkw = {'freqmin':120,'freqmax':2000}#{'freq':800}
IM_CdBs = 50


FMT = 'PNG'
DPI = 200
issave = True

### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..')
PD_WFD = os.path.join(ROOT,'processed_data','Hybrid_Seismic','Corrected_t0','Prudhoe',\
					 PD_spread,'shot',str(PD_shot_ID))
IM_WFD = os.path.join(ROOT,'processed_data','Hybrid_Seismic','Corrected_t0','Hiawatha',\
					 IM_spread,'shot',str(IM_shot_ID))

# Waveform filenames for GeoRods
PD_WF = glob(os.path.join(PD_WFD,'PL.%s.*.mseed'%(PD_spread)))
PD_WF.sort()
IM_WF = glob(os.path.join(IM_WFD,'HI.%s.*.mseed'%(IM_spread)))
IM_WF.sort()

# Corrected t(x) data
PD_DTX = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0',\
					 'Prudhoe_Dome','Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL_RT.csv')
IM_DTX = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0',\
					 'Inglefield_Land','Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL.csv')


### LOAD DATA ###
st_PD = Stream([read(f_)[0] for f_ in PD_WF])
st_IM = Stream([read(f_)[0] for f_ in IM_WF])
df_PD = pd.read_csv(PD_DTX)
df_IM = pd.read_csv(IM_DTX)

# Subset Phase Pick Data
df_PD = df_PD[(df_PD['shot #'] == PD_shot_ID)&\
			  (df_PD['itype']=='GeoRod')]
# Sort Phase Pick Data
df_PD = df_PD.sort_values(['SRoff m','phz','kind'])
# Subset to channels that have diving wave picks
df_PDf = df_PD[(df_PD['kind']==2)&\
			   (df_PD['phz']=='P')]


df_IM = df_IM[(df_IM['shot #'] == IM_shot_ID)&\
			  (df_IM['itype']=='GeoRod')]
# Sort Phase Pick Data
df_IM = df_IM.sort_values(['SRoff m','phz','kind'])
# Subset to channels that have diving wave picks
df_IMf = df_IM[(df_IM['kind']==2)&\
			   (df_IM['phz']=='P')]


### MERGE DATA ###
st_PD2 = Stream()
# Iterate across picks
for i_ in range(len(df_PDf)):
	# plt.figure()
	S_i = df_PDf.iloc[i_,:].T
	# Fetch relevant trace
	tr = st_PD.select(station=S_i['sta'],channel=S_i['chan'])[0]

	# Calculate time-shift
	tr_dt = S_i['tt sec'] - S_i['epoch old'] + S_i['t0 ref']
	# Update tr header with distance
	tr.stats.update({'distance':S_i['SRoff m']})
	# Apply time-shift
	tr.stats.starttime += tr_dt

	# Apply filtering
	if PD_ftype is not None and filt_then_tar:
		tr = tr.filter(PD_ftype,**PD_fkw)
	# Normalize trace
	tr.normalize()
	# Apply WB TAR
	tr = WB_TAR_tr(tr,tWB=UTCDateTime(S_i['time']),C=PD_CdBs)

	# Apply filtering
	if PD_ftype is not None and not filt_then_tar:
		tr = tr.filter(PD_ftype,**PD_fkw)


	# Append to trace
	st_PD2 += tr
	# print('%d %d'%(i_,len(st_PD2)))

# Trim shifted data
st_PD2 = st_PD2.trim(starttime=UTCDateTime(S_i['t0 ref']),endtime=UTCDateTime(S_i['t0 ref']) + 4.)


st_IM2 = Stream()
# Iterate across picks
for i_ in range(len(df_IMf)):
	# plt.figure()
	S_i = df_IMf.iloc[i_,:].T
	# Fetch relevant trace
	tr = st_IM.copy().select(station=S_i['sta'],channel=S_i['chan'])[0]
	# Apply filtering
	if IM_ftype is not None:
		tr = tr.filter(IM_ftype,**IM_fkw)
	# plt.plot(tr.data,alpha=0.5,label='raw')
	# Normalize trace
	tr.normalize()
	# Apply WB TAR
	tr = WB_TAR_tr(tr,tWB=UTCDateTime(S_i['time']),C=IM_CdBs)
	# Calculate time-shift
	tr_dt = S_i['tt sec'] - S_i['epoch old'] + S_i['t0 ref']
	# Update tr header with distance
	tr.stats.update({'distance':S_i['SRoff m']})
	# Apply time-shift
	tr.stats.starttime += tr_dt
	# Append trace
	st_IM2 += tr
	# print('%d %d'%(i_,len(st_IM2)))

# Trim shifted data
st_IM2 = st_IM2.trim(starttime=UTCDateTime(S_i['t0 ref']),endtime=UTCDateTime(S_i['t0 ref']) + 4.)



### PLOTTING SECTION ###

fig = plt.figure()
GS = fig.add_gridspec(ncols=2,nrows=5)
axs = [fig.add_subplot(GS[:4,0]),fig.add_subplot(GS[4:,0]),\
	   fig.add_subplot(GS[:4,1]),fig.add_subplot(GS[4:,1])]



lbl = ['Diving','Reflection','Multiple']
### (a) Prudhoe Dome Example Section 
for tr_ in st_PD2:
	dat = tr_.data*10 + tr_.stats.distance
	dt = np.linspace(0,tr_.stats.endtime - tr_.stats.starttime,tr_.stats.npts)*1e3
	axs[0].fill_betweenx(dt,dat,np.ones(tr_.stats.npts)*tr_.stats.distance,\
						 where=tr_.data > 0,linewidth=0.25,\
						 color='red',alpha=0.75)
	axs[0].plot(dat,dt,'k-',lw=0.5)

for i_,p_ in enumerate(df_PD['phz'].unique()):
	idf = df_PD[(df_PD['kind']==2)&(df_PD['phz']==p_)].sort_values(['SRoff m'])
	axs[0].plot(idf['SRoff m'],idf['tt sec'].values*1e3,'.-',lw=0.5,ms=1,label=lbl[i_])
axs[0].set_ylim([800,0])
axs[0].set_xlim([df_PD['SRoff m'].min() - 10,df_PD['SRoff m'].max() + 10])

plt.legend()


### (b) Inglefield Margin Example Section
for tr_ in st_IM2:
	dat = tr_.data*10 + tr_.stats.distance
	dt = np.linspace(0,tr_.stats.endtime - tr_.stats.starttime,tr_.stats.npts)*1e3
	axs[2].plot(dat,dt,'k-',lw=0.5)
	axs[2].fill_betweenx(dt,dat,np.ones(tr_.stats.npts)*tr_.stats.distance,\
					 where=tr_.data > 0,linewidth=0.25,\
					 color='red',alpha=0.75)

for i_,p_ in enumerate(df_IM['phz'].unique()):
	idf = df_IM[(df_IM['kind']==2)&(df_IM['phz']==p_)].sort_values(['SRoff m'])

	axs[2].plot(idf['SRoff m'],idf['tt sec'].values*1e3,'.-',lw=0.5,ms=4,label=lbl[i_])
axs[2].set_ylim([800,0])
axs[2].set_xlim([df_IM['SRoff m'].min() - 10,df_IM['SRoff m'].max() + 10])

plt.show()

# st_PD2.plot(method='full',type='section',time_down=True,interactive=True,equal_scale=False)







# fig = plt.figure()
# GS = fig.add_gridspec(ncol=2,nrow=5)
# # Generate Subplots


# ### Plot Waveforms ###
