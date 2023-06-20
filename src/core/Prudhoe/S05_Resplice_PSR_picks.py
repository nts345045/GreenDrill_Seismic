"""
:module: S05_Resplice_PSR_picks.py - splice picks back in from multiple picking on TAR analysis
									 into the primary Phase data

:AUTH: Nathan T. Stevens
:EMAIL: nts5045@psu.edu
"""
import os
import pandas as pd
import numpy as np
import pyrocko.gui.marker as pm
from glob import glob
import matplotlib.pyplot as plt


### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..')
PROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome') #Inglefield_Land
DPHZ = os.path.join(PROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured.csv')
OFILE = os.path.join(PROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR.csv')
ACROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','AC_PROCESSING','PL','{SPREAD}','shot','*') #HI

### LOAD PRIOR PHASE DATA ###
df_PH = pd.read_csv(DPHZ,parse_dates=['time'])

# Copy column headers for post-iteration 
cols = list(df_PH.columns)

# Create holder for updated lines
holder = []
# Iterate across spreads
for SP_ in ['NS01','NS02','NS03','WE01','WE02','WE03']:
	# Fetch Directory Names 
	DGLOB = glob(ACROOT.format(SPREAD=SP_))
	DGLOB.sort()
	# Iterate Across Shot Directories
	for D_ in DGLOB:
		# Split off Shot Code [str]
		PA_,SH_ = os.path.split(D_)
		# Pull PSR file
		PSRF = glob(os.path.join(D_,'PSR.*.*.picks.dat'))
		# Get the PSR file name
		_name_ = os.path.split(PSRF[0])[-1]
		# Get waveform file names
		wf_names = glob(os.path.join(D_,'*RAW.mseed'))
		# Pull filtering information
		_,fmin,fmax,_,_ = _name_.split('.')
		# Load markers
		marks = pm.load_markers(PSRF[0])
		for M_ in marks:
			# Split out NSLC information
			n_,s_,l_,c_, = M_.get_nslc_ids()[0]
			# Get associated, time-corrected, RAW waveform file
			wffile = os.path.join(D_,'%s.%s.%s.%s._t0_corr_RAW.mseed'%(n_,s_,l_,c_))
			# Get marker kind
			k_ = M_.kind
			# Get marker phase
			ph_ = M_.get_label()
			# Set type
			typ = 'Phase'
			# Get marker time
			ttsec = M_.get_tmin()
			# Subset old phases for metadata
			df_i = df_PH[(df_PH['spread']==SP_) &\
						 (df_PH['shot #']==int(SH_)) &\
						 (df_PH['sta']==s_) &\
						 (df_PH['chan']==c_)]
			OC_ = df_i['offset code'].values[0]
			# Get full-timestamps
			eold = ttsec
			epoch = ttsec #df_i['t0 ref'].values[0] + ttsec 
			time = pd.Timestamp(epoch,unit='s')
			# breakpoint()
			# Get association hash
			ihash = df_i['hash'].values[0]
			# Set static status (i.e., is roving shot timer?)
			static = True
			freq1 = float(fmin)
			freq2 = float(fmax)
			if freq1 == 0 and freq2 == 2000:
				filt = np.nan
				freq1 = np.nan
				freq2 = np.nan
			elif freq1 > 0 and freq2 == 2000:
				filt = 'highpass'
				freq2 = freq1
				freq1 = np.nan
			elif freq1 == 0 and freq2 < 2000:
				filt = 'lowpass'
				freq1 = freq2
				freq2 = np.nan
			else:
				filt = 'bandpass'


			itype = 'GeoRod'
			# Scrape Source-Receiver Geometry Information
			SRoff = df_i['SRoff m'].values[0]
			SRmE = df_i['SR mE'].values[0]
			SRmN = df_i['SR mN'].values[0]
			SRmH = df_i['SR mH'].values[0]
			az = df_i['az rad'].values[0]
			ang = df_i['ang rad'].values[0]
			CMPmE = df_i['CMP mE'].values[0]
			CMPmN = df_i['CMP mN'].values[0]
			CMPmH = df_i['CMP mH'].values[0]
			t0REF = df_i['t0 ref'].values[0]

			line = [SP_,int(SH_),OC_,n_,s_,l_,c_,k_,ph_,typ,\
				    time,eold,ihash,static,filt,freq1,freq2,\
				    itype,wffile,SRoff,SRmE,SRmN,SRmH,az,ang,\
				    CMPmE,CMPmN,CMPmH,t0REF,epoch,ttsec]
			holder.append(line)

			# cols = ['spread','shot #','offset code','net','sta','loc','chan','phz','type',\
			# 		'time','epoch old','hash','static','filt','freq1','freq2',\
			# 		'itype','wf file','SRoff m','SR mE','SR mN','SR mH','az rad','ang rad',\
			# 		'CMP mE','CMP mN',CMP mH','t0 ref','epoch','tt sec']
df_G_new = pd.DataFrame(holder,columns=cols)

df_N = df_PH[df_PH['itype']=='Node']
df_OUT = pd.concat([df_G_new,df_N],axis=0,ignore_index=True)

df_OUT.to_csv(OFILE,header=True,index=False)


