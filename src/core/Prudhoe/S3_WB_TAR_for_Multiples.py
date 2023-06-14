"""
S3_WB_TAR_for_Multiples.py

Use time-corrected diving wave picks to fix timing for traces apply a Water Bottom True Amplitude Recovery 
to traces

:AUTH: Nathan T. Stevens
:EMAIL: nts5045@psu.edu


"""
import os
import pandas as pd
import numpy as np
from obspy import read, UTCDateTime, Stream
import pyrocko.gui.marker as pm
from pyrocko import obspy_compat
from tqdm import tqdm
from glob import glob

def WB_TAR_tr(tr,tWB=None,C=6):
	"""
	Apply a Water Bottom True Amplitude Recovery
	to an obspy.core.Trace object
	
	:: INPUTS ::
	:param tr: [obspy.core.Trace] Trace to copy and apply TAR to
	:param tWB: [obspy.core.UTCDateTime or None] water bottom time
	:param C: [float] gain correction factor (dB/sec)

	:: OUTPUT ::
	:return trTAR: [obspy.core.Trace] Trace with applied gain correction
					and processing annotation in the trTAR.stats.processing
					header information
	"""
	# Copy data vector
	dat = tr.copy().data
	# Get time deltas
	dt = np.linspace(0,tr.stats.endtime - tr.stats.starttime,tr.stats.npts)
	# If a user-defined WB time is provided, apply to dt
	if isinstance(tWB,UTCDateTime):
		dt -= (tWB - tr.stats.starttime)
	# Apply gain to data following the WB time
	# breakpoint()
	dat = np.r_[dat[dt < 0.],dat[dt >= 0.] *10.**((C*dt[dt >= 0.])/20.)]
	# Create a copy of the trace for output
	trTAR = tr.copy()
	trTAR.data = dat
	try:
		trTAR.stats.processing.append('Custom: WB_TAR_tr(tWB=%s,C=%.2f)'%(str(tWB),C))
	except:
		trTAR.stats.update({'processing':['Custom: WB_TAR_tr(tWB=%s,C=%.2f)'%(str(tWB),C)]})
	return trTAR



### MAP DATA ###

ROOT = os.path.join('..','..','..','..','..')
# Glob path for waveforms
PD_GST = os.path.join(ROOT,'processed_data','Hybrid_Seismic','Corrected_t0','Prudhoe','{SPREAD}','shot','{SHOT}','{SITE}.{SPREAD}.*.mseed')
# File path/name for timing corrected pick data
PD_DTX = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0',\
					 'Prudhoe_Dome','Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')
# Initial data analysis processing records
SHFD = os.path.join(ROOT,'processed_data','Hybrid_Seismic','Corrected_t0','Amplitude_Pick_File_Metadata_v5.csv')
# Output Directory
ODIR = os.path.join(ROOT,'processed_data','Hybrid_Seismic','AC_PROCESSING','{SITE}','{SPREAD}','shot','{SHOT}')


### USER CONTROL SECTION ###
CdBs = 50  	# [dB/sec] Gain correction coefficient
RL = 2.5  	# [sec] output record length

### LOAD METADATA ###
# Load shot metadata
df_META = pd.read_csv(SHFD)
df_META = df_META[df_META['Site']=='Prudhoe']
# Load picks
df_PICK = pd.read_csv(PD_DTX)

# Iterate across spread-shot combinations
for SP_,SH_ in df_META[['Spread','Shot #']].value_counts().index:
	print('Processing {} {}'.format(SP_,SH_))
	FLIST  = glob(PD_GST.format(SPREAD=SP_,SHOT=SH_,SITE='PL'))
	# Load waveforms
	stream = Stream([read(f_)[0] for f_ in FLIST])
	# Subset specific picks
	df_PG = df_PICK[(df_PICK['shot #'] == SH_) & (df_PICK['itype'] == 'GeoRod')]
	df_PG = df_PG.sort_values(['SRoff m','phz','kind'])
	# Subset by diving wave observations
	df_PGF = df_PG[(df_PG['kind']==2) & (df_PG['phz']=='P')]

	stream2 = Stream()
	markers = []
	# Define output path from template	
	OPATH = ODIR.format(SITE='PL',SPREAD=SP_,SHOT=SH_)
	try:
		os.makedirs(OPATH)
	except:
		pass
	# Iterate across diving-wave-picked traces
	for i_ in tqdm(range(len(df_PGF))):
		### ASSOCIATION BLOCK ###
		# Get diving wave pick information
		S_i = df_PGF.iloc[i_,:].T
		# Get matching trace
		trace = stream.select(station=S_i['sta'],channel=S_i['chan'])[0]

		### TIMING CORRECTION BLOCK ###
		# Back-calculate timing correction
		dt_corr = S_i['tt sec'] - S_i['epoch old'] + S_i['t0 ref']
		# Apply timing correction to trace timing
		trace.stats.starttime += dt_corr
		# trace.stats.starttime = UTCDateTime(0) - dt_corr
		# Get absolute time for diving wave reference pick
		tp_new = S_i['t0 ref'] + S_i['tt sec']
		# Update absolute pick time
		nslc = (trace.stats.network,trace.stats.station,\
				trace.stats.location,trace.stats.channel)
		# Trim trace
		trace = trace.trim(starttime=UTCDateTime(S_i['t0 ref']),\
						   endtime=UTCDateTime(S_i['t0 ref'] + RL))
		# trace = trace.trim(endtime=UTCDateTime(0) + RL)
		df_J = df_PG[(df_PG['sta']==S_i['sta']) & (df_PG['chan'] == S_i['chan'])]
		for j_ in range(len(df_J)):
			# Subset to specific pick
			S_j = df_J.iloc[j_,:].T
			# Update pick absolute timing
			# tpj_new = S_j['t0 ref'] + S_j['tt sec']
			tpj_new = UTCDateTime(0).timestamp + S_j['tt sec']
			# Make pyrocko pick marker for pick
			mark = pm.PhaseMarker((nslc,),tmin=tpj_new,tmax=tpj_new,phasename=S_j['phz'],kind=S_j['kind'])
			# Append to marker list
			markers.append(mark)
			
			# Model Multiple
			if S_j['phz'] == 'S' and S_j['kind']==2:
				# tpk_new = S_j['t0 ref'] + 2*S_j['tt sec']
				tpk_new = UTCDateTime(0).timestamp + 2.*S_j['tt sec']
				mark = pm.PhaseMarker((nslc,),tmin=tpk_new,tmax=tpk_new,phasename='R',kind=5)
				markers.append(mark)

		### WB TAR BLOCK ###
		# Apply water bottom TAR to trace
		trace = WB_TAR_tr(trace,tWB=UTCDateTime(tp_new),C=CdBs)
		# Shift to Epoch reference
		trace.stats.starttime = UTCDateTime(0)
		# Add trace to output stream
		stream2 += trace
		# Write trace to output directory
		OFILE = '%s.%s.%s.%s_t0_corr_TAR_C'%nslc + str(int(CdBs)) + '.mseed'
		# breakpoint()
		trace.write(os.path.join(OPATH,OFILE),fmt='MSEED')



	# Write markers to output directory
	OFILE = 'timing_corrected_picks.dat'
	OUT_TR_FMT = pm.save_markers(markers,os.path.join(OPATH,OFILE),fdigits=4)











# ### MERGE DATA ###
# st_PD2 = Stream()
# # Iterate across picks
# for i_ in range(len(df_PDf)):
# 	# plt.figure()
# 	S_i = df_PDf.iloc[i_,:].T
# 	# Fetch relevant trace
# 	tr = st_PD.select(station=S_i['sta'],channel=S_i['chan'])[0]

# 	# Calculate time-shift
# 	tr_dt = S_i['tt sec'] - S_i['epoch old'] + S_i['t0 ref']
# 	# Update tr header with distance
# 	tr.stats.update({'distance':S_i['SRoff m']})
# 	# Apply time-shift
# 	tr.stats.starttime += tr_dt

# 	# Apply filtering
# 	if PD_ftype is not None and filt_then_tar:
# 		tr = tr.filter(PD_ftype,**PD_fkw)
# 	# Normalize trace
# 	tr.normalize()
# 	# Apply WB TAR
# 	tr = WB_TAR_tr(tr,tWB=UTCDateTime(S_i['time']),C=PD_CdBs)

# 	# Apply filtering
# 	if PD_ftype is not None and not filt_then_tar:
# 		tr = tr.filter(PD_ftype,**PD_fkw)


# 	# Append to trace
# 	st_PD2 += tr
# 	# print('%d %d'%(i_,len(st_PD2)))

# # Trim shifted data
# st_PD2 = st_PD2.trim(starttime=UTCDateTime(S_i['t0 ref']),endtime=UTCDateTime(S_i['t0 ref']) + 4.)