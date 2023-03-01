import numpy as np
import pandas as pd
from scipy.fft import fft,fftfreq

def finite_rms(x):
	"""
	Calculate Root Mean Squared from finite-valued
	elements of input array x
	"""
	IND = np.isfinite(x)
	n_finite = len(x[IND])
	x_finite = x[IND]
	RMS = np.sqrt((1./n_finite) * np.sum(x_finite**2))
	return RMS

def find_local_extremum(dat):
	"""
	Find largest extremum and return its signed amplitude and index
	:: INPUTS ::
	:param dat: data vector

	:: OUTPUTS ::
	:return amp: signed amplitude
	:return ind: index of largest extremum
	"""

	amin = np.nanmin(dat)
	imin = np.nanargmin(dat)
	amax = np.nanmax(dat)
	imax = np.nanargmax(dat)
	if np.abs(amin) > np.abs(amax):
		pol = -1
		amp = np.abs(amin)
		ind = imin
	elif np.abs(amax) > np.abs(amin):
		pol = 1
		amp = np.abs(amax)
		ind = imax
	else:
		pol = np.sign(amax)
		amp = amax
		ind = imax
	# breakpoint()
	return pol*amp, ind

def extract_pick_value(dat):
	abar = np.nanmean(dat)
	ibar = (len(dat)-1)/2
	return abar,ibar

def spect(data,dt):
	npts = len(data)
	x = np.linspace(0.,npts*dt,npts,endpoint=False)
	fft_c = np.abs(fft(data))[:npts//2]
	ffreq = fftfreq(npts,dt)[:npts//2]
	return fft_c,ffreq


### WRAPPERS ###

def find_local_extremum_tr(tr,to,tf):
	"""
	Wrapper for find_local_extremum() for an Obspy.Trace object
	and specified bounding UTCDateTimes

	:: INPUTS ::
	:param tr: obspy.Trace with data
	:param to: starttime for tr.copy().trim()
	:param tf: endtime for tr.copy().trim()

	:: OUTPUTS ::
	:return amp: signed extremum amplitude value
	:return tx: timestamp of extremum value

	"""
	tr_ = tr.copy().trim(starttime=to,endtime=tf)
	dat = tr_.data
	amp,ind = find_local_extremum(dat)
	tx = to + ind*tr_.stats.delta
	return amp, tx

def extract_pick_value_tr(tr,ti):
	ii = int((ti-tr.stats.starttime)/tr.stats.delta)
	# breakpoint()
	amp = tr.data[ii]
	tx = tr.stats.starttime + tr.stats.delta*ii
	return amp, tx


def find_local_extremum_st(st,to,tf,method='rms'):
	"""
	Wrapper for find_local_extremum() for Obspy.Stream objects

	:: INPUTS ::
	:param st: obspy.Stream with data-containing Traces
	:param to: starttime for tr.copy().trim()
	:param tf: endtime for tr.copy().trim()
	:param method: method for calculating composite values. Options are
				'rms': sqrt((1/#traces)*sum(trace.data**2))
				'euclidian': sqrt(sum(trace.data**2))
				with summation across the traces in st

	:: OUTPUTS ::
	:return amp: unsigned amplitude of extremum as calculated based on 'method'
	:return tx: timestamp of extremum amplitude
	:return amp_i: dictionary of signed amplitudes at time tx with station NSLC as
					dictionary keys 
	"""
	st_ = st.copy().trim(starttime=to,endtime=tf)
	if method in ['euclidian','rms']:
		for i_,tr_ in enumerate(st_):
			if i_ == 0:
				dat = tr_.data**2
			else:
				dat += tr_.data**2
		if method == 'euclidian':
			dat = dat**0.5
		elif method == 'rms':
			dat = ((1./len(st_))*dat)**0.5

	amp,ind = find_local_extremum(dat)
	tx = to + ind*tr_.stats.delta
	amp_i = {}
	for tr_ in st_:
		nslc = (tr_.stats.network,tr_.stats.station,tr_.stats.location,tr_.stats.channel)
		iamp = tr_.data[ind]
		amp_i.update({nslc:iamp})
	return amp,tx,amp_i


def apply_to_windowed_trace(tr,to,tf,method=finite_rms):
	"""
	WRAPPER: Apply specified method to amplitude data from
	a time-bounded waveform object and return method output

	:: INPUTS ::
	:type tr: obspy.core.trace.Trace
	:param tr: Trace object with valid data vector
	:type to: obspy.core.UTCDateTime
	:param to: Starting Time
	:type tf: obspy.core.UTCDateTime
	:param tf: Ending Time
	:type method: method
	:param method: method to apply to tr.data
	
	:: OUTPUT ::
	:rtype out: float
	:return out: output from method(tr.copy().trim(<to,tf>).data)

	"""
	tr_ = tr.copy().trim(starttime=to,endtime=tf)
	out = method(tr_.data)
	return out


def spect_tr(tr,ts,te,name):
	tr_ = tr.copy().trim(starttime=ts,endtime=te)
	fft_c,ffreq = spect(tr_.data,tr.stats.delta)
	S_ = pd.Series(fft_c,index=ffreq,name=name)
	return S_


def read_fft_file(fft_file,ffv=np.arange(0,1000,12.5)):
	"""
	Read fft spectrum collection from S4_amplitude_extraction.py

	:: INPUTS ::
	:param fft_file: complete (or valid relative path) to fft spectrum collection
	:param ffv: fourier frequencies for interpolating spectra

	:: OUTPUTS ::
	:return df_F: raw formatted DataFrame of spectrum collection
	:return df_FH: resampled spectra defined by 
	"""
	df_F = pd.read_csv(fft_file,index_col=[0])
	ffc_bar = []; IND = []
	for i_ in range(len(df_F)):
		idf = df_F.iloc[i_,:]
		ff_bar = np.interp(ffv,idf[idf.notna()].index.astype(float),idf[idf.notna()].values)
		ffc_bar.append(ff_bar)
		IND.append(idf.name)
	df_FH = pd.DataFrame(np.array(ffc_bar),columns=ffv,index=IND)
	return df_F,df_FH