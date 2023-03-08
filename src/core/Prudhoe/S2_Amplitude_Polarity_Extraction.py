"""
:module: S2_Amplitude_Polarity_Extraction.py
:purpose: STEP 2 - Extract amplitude measures and polarity values for all picks from waveform data and write
				to file for quick access in subsequent amplitude analyses
:auth: Nathan T. Stevens
:email: nts5045@psu.ed | ntstevens@wisc.edu
:Synopsis: 
	Inputs: Compiled Picks, waveform file catalog (WFDISC)
	Tasks: Extract amplitude estimates from waveform data at pick times 
			and append to compiled pick entries
	Outputs: Compiled Picks += Amplitude and polarity data on all picks, 
			polarity agreement on all traces with direct & reflected/multiple arrivals

:: TODO ::
"""
import pandas as pd
import numpy as np
from obspy import UTCDateTime, read
# Add repository root to path & get repo modules of use
import sys
import os
sys.path.append(os.path.join('..','..'))
import util.TimeseriesTools as TsT


##### CORE PROCESSES #####

def fetch_amplitudes(df_picks,wf_,t0pad=0.005,tfpad=0.055,fftpad=0.01):
	"""
	Fetch waveform data for a given shot-station combination and use analyst picks for 'kind'==1 and 'kind'==2
	picks to extract the following wavelet amplitude estimates:
	1) Pick Polarity: -1.0 for negative, 1.0 for positive
	2) Closest extreme: for all individual picks
	3) Cycle RMS amplitude (if kind 1 and kind 2 picks are both present) 
	4) Wavelet RMS amplitude for tpick - t0pad to tpick + tfpad

	Amplitude estimates are made after applying the analyst-specified pre-filter to waveform data, this information
	is containded in df_picks

	Also calculate the FFT for each wavelet for a window specified as t \\in [tpick - t0pad -fftpad, tpick + tfpad + fftpad]

	:: INPUTS ::
	:param df_picks: Primary output from S1 (Compiled Picks with analyst metadata)
	:param wf_: name of waveform file (with necessary relative/absolute path) from which to extract amplitudes and spectra
	:param t0pad: time in seconds to front-pad picks for RMS amplitude and spectrum extraction
	:param tfpad: time in seconds to back-pad picks for RMS amplitude and spectrum extraction
	:param fftpad: additional time in seconds to front-/back-pad picks for spectrum extraction

	:: OUTPUTS ::
	:return df_out: Updated copy of df_picks with extracted amplitudes appended
	:return df_fft: DataFrame containing FFT amplitudes (real-valued components) and Fourier frequencies for each 
					spectrum extracted from the input waveform

	"""

	df_out = pd.DataFrame()
	df_fft = pd.DataFrame()
	dfi = df_picks[df_picks['wf file'] == wf_]
	# Iterate over waveforms
	if wf_ is not None:
		try:
			print('Processing %s (Site: %s, Spread: %s, Shot: %s)'%(wf_,dfi['net'].values[0],dfi['spread'].values[0],dfi['shot #'].values[0]))
		except IndexError:
			breakpoint()
		tr = read(wf_,fmt='MSEED')[0]

		# Identify filters used by picking analyst
		if dfi['filt'].values[0] == 'highpass':
			trf = tr.filter('highpass',freq=dfi['freq2'].values[0])
		elif dfi['filt'].values[0] == 'bandpass':
			trf = tr.filter('bandpass',freqmin=dfi['freq1'].values[0],\
									  freqmax=dfi['freq2'].values[0])
		else:
			trf = tr.copy().detrend()


		### EXTRACT INDIVIDUAL AMPLITUDE ESTIMATES DATA ###
		for phz_ in ['P','S','R']:
			S_i = dfi[(dfi['phz']==phz_) & (dfi['kind']==1)].squeeze()
			S_j = dfi[(dfi['phz']==phz_) & (dfi['kind']==2)].squeeze()
			# If there is a phase pick of kind 1
			if len(S_i) >= 1:
				# NOTE: 'time' does not have velocity time adjusts, so it matches V1 timing for georods
				ti = UTCDateTime(S_i['time'])
				# Ai,txi = find_local_extremum_tr(trf,ti - ext_bound,ti + ext_bound)
				Ai,txi = TsT.extract_pick_value_tr(trf,ti)# - ext_bound,ti + ext_bound)
				# breakpoint()
				AiRMS = TsT.apply_to_windowed_trace(trf,ti - t0pad,ti + tfpad)
				# Calculate FFT for each pick
				S_fft = TsT.spect_tr(trf,ti - t0pad - fftpad,ti + tfpad + fftpad,name=S_i.name)
				ipol = np.sign(Ai)
			else:
				ti = np.nan; Ai = np.nan; AiRMS = np.nan; ipol = np.nan

			# If there is a phase pick of kind 2
			if len(S_j) >= 1:
				try:
					tj = UTCDateTime(S_j['time'])
				except TypeError:
					breakpoint()
				# Aj,txj = find_local_extremum_tr(trf,tj - ext_bound,tj + ext_bound)
				Aj,txj = TsT.extract_pick_value_tr(trf,tj)# - ext_bound,tj + ext_bound)
				AjRMS = TsT.apply_to_windowed_trace(trf,tj - t0pad,tj + tfpad)
				if not len(S_i) >= 1:
					S_fft = TsT.spect_tr(trf,tj - t0pad - fftpad,tj + tfpad + fftpad,name=S_j.name)
				jpol = np.sign(Aj)
			else:
				tj = np.nan; Aj = np.nan; AjRMS = np.nan; jpol = np.nan

			# If there are a pair of phase picks for a given wavelet
			# Get additional metrics based on the defined "first cycle"
			if len(S_i) >= 1 and len(S_j) >= 1:
				dtij = tj - ti
				Aij = TsT.apply_to_windowed_trace(trf,ti - 0.5*dtij,tj + 0.5*dtij)
			else:
				dtij = np.nan; Aij = np.nan

			if len(S_i) >= 1 or len(S_j) >= 1:
				idict = {'Polarity':ipol,'Peak Amp':Ai,'RMS Amp':AiRMS,'Cycle RMS Amp':Aij,'dt12':dtij}
				jdict = {'Polarity':jpol,'Peak Amp':Aj,'RMS Amp':AjRMS,'Cycle RMS Amp':Aij,'dt12':dtij}
				# breakpoint()
				if len(S_i) >= 1:
					S_io = S_i.copy().append(pd.Series(idict,name=S_i.name))
					df_out = pd.concat([df_out,S_io],axis=1,ignore_index=False)
					df_fft = pd.concat([df_fft,S_fft],axis=1,ignore_index=False)
					# breakpoint()
				if len(S_j) >= 1:
					S_jo = S_j.copy().append(pd.Series(jdict,name=S_j.name))
					df_out = pd.concat([df_out,S_jo],axis=1,ignore_index=False)
					if not len(S_i) >= 1:
						df_fft = pd.concat([df_fft,S_fft],axis=1,ignore_index=False)
			# breakpoint()
			# df_out = pd.concat([df_out,S_io,S_jo],axis=1,ignore_index=False)
	else:
		dfe = dfi.copy()
		# breakpoint()
		df_out = pd.concat([df_out,dfe],axis=0,ignore_index=False)

	return df_out.T, df_fft.T

##### DRIVER #####
def fetch_amplitudes_driver(df_picks,t0pad=0.005,tfpad=0.055,fftpad=0.01):
	"""
	Wrapper for fetch_amplitudes() that iterates over shots and unique waveform files 
	listed in df_picks to feed into fetch_amplitudes()

	Driver saves spectrum and amplitudes on a shot-gather basis into the waveform root
	directory.

	:: INPUTS :: - see fetch_amplitudes()
	:: OUTPUTS ::
	:return df_report: Summary index of extracted amplitude and spectrum files saved to 
					source data directories.
	:saves processed_amplitudes_*.csv: Files containing amplitude data saved to disk
	:saves processed_spectra_*.csv: Files containing spectra data saved to disk
	"""
	S_meta= df_picks[['spread','shot #']].value_counts()
	amp_save_files = []; save_spreads = []; save_shots = []; fft_save_files = []
	for I_ in range(len(S_meta)):
		I_spread,I_shot = S_meta.index[I_]
		save_spreads.append(I_spread)
		save_shots.append(I_shot)
		idf_picks = df_picks[(df_picks['spread']==I_spread) & (df_picks['shot #']==I_shot)]
		print('Processing Spread %s, Shot %d'%(I_spread, I_shot))
		WF_ = idf_picks['wf file'].unique()
		DF_OUT = pd.DataFrame()
		DF_FFT = pd.DataFrame()
		for i_,wf_ in enumerate(WF_):
			print('%03d/%03d'%(i_+1,len(WF_)))
			idf_out,idf_fft = fetch_amplitudes(idf_picks,wf_,t0pad=t0pad,tfpad=tfpad,fftpad=fftpad)
			DF_OUT = pd.concat([DF_OUT,idf_out],axis=0,ignore_index=False)
			DF_FFT = pd.concat([DF_FFT,idf_fft],axis=0,ignore_index=False)
		I_savefile = os.path.join(os.path.split(wf_)[0],'processed_amplitudes_v5.csv')
		F_savefile = os.path.join(os.path.split(wf_)[0],'processed_spectra_v5.csv')
		amp_save_files.append(I_savefile)
		fft_save_files.append(F_savefile)
		DF_OUT.to_csv(I_savefile,header=True,index=True)
		DF_FFT.to_csv(F_savefile,header=True,index=True)

	df_report = pd.DataFrame({'Spread':save_spreads,'Shot':save_shots,'Amp File':amp_save_files,'FFT File':fft_save_files})
	return df_report

##### ACTUAL PROCESSING #####
ROOT = os.path.join('..','..','..','..','..')
DROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
DPHZ = os.path.join(DROOT,'VelCorrected_Phase_Picks_O2_idsw_v5.csv')
### LOAD PHASE DATA ###
df_ = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
### RUN CORE PROCESSING ###
df_R_PD = fetch_amplitudes_driver(df_,fftpad=0.02)
### WRITE DIRECTORIES TO DISK ###
df_R_PD.to_csv(os.path.join(DROOT,'AmpSpect_Extraction_Index_v5.csv'),header=True,index=False)




##### END #####