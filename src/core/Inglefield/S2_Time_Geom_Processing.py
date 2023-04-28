"""
:module: S1_Time_Geom_Processing.py
:purpose: STEP 1: Conduct shot-receiver geometry calculations for subsequent calculations and conduct spread-wise
		time corrections for GeoRod picks.
:auth: Nathan T. Stevens
:Synopsis:
    Inputs: Pick Times, Shot Locations, Receiver Locations
    Tasks: Compile individual shot-gather picks, calculate geometries for CMP gathers, 
    		use KB79 modeling to correct non-GPS timed data (GeoRods)
    Outputs: Compiled picks with t(x) correction factors and Source-Receiver Geometry 
    		data (Compiled Picks hereafter), initial KB79 models

:: TODO ::
Add elevation estimation 

"""
import pandas as pd
import numpy as np
import pyrocko.gui.marker as pm
from glob import glob
import sys
import os
import matplotlib.pyplot as plt
# Add repository root to path & get repo modules of use
sys.path.append(os.path.join('..','..'))
import util.GeometryTools as gt
import util.InvTools as inv


##### SUPPORTING PROCESSES #####

def calc_geom(src_LLH,rcv_LLH,proj_epsg='epsg:32619',name=None):
	"""
	Calculate a set of geometric parameters from source-receiver 
	locations.

	:: INPUTS ::
	:param src_LLH: array-like Source coordinates in order: Latitude, Longitude, Height
	:param rcv_LLH: array-like Receiver coordinates in order: Latitude, Longitude, Height
	:param proj_epsg: string for pyproj.Proj for an appropriate EPSG
	:param Name: Name for output Series (use phase index)

	:: OUTPUTS ::
	:return geom: pandas.Series containing the following fields:
					SRoff m - source-receiver offset in meterse
					SRoff mE - SRoff easting element
					SRoff mN - SRoff northing element
					SRoff mH - SRoff elevation element (positive up)
					SRaz - Source-to-receiver azimuth (in radians), clockwise rotation from North
					SRang - Source-to-receiver angle (in radians), counterclockwise rotation from East
					src mE - Source easting 
					src mN - Source northing
					src mH - Source elevation
					rcv mE - receiver easting
					rcv mN - receiver northing
					rcv mH - receiver elevation
					CMP mE - midpoint easting element
					CMP mN - midpoint northing element

	"""
	# Convert into UTM
	Sx,Sy = gt.LL2epsg(src_LLH[1],src_LLH[0],epsg=proj_epsg)
	Sh = src_LLH[2]
	Rx,Ry = gt.LL2epsg(rcv_LLH[1],rcv_LLH[0],epsg=proj_epsg)
	Rh = rcv_LLH[2]
	# Get SR offset elements
	dx = Rx - Sx
	dy = Ry - Sy
	dh = Rh - Sh
	dD = (dx**2 + dy**2 + dh**2)**0.5
	# Get angles
	az,ang = gt.cartesian_azimuth(dx,dy)
	# Calculate midpoints
	mpE = Sx + 0.5*dx
	mpN = Sy + 0.5*dy

	S_out = pd.Series([dD,dx,dy,dz,az,ang,Sx,Sy,Sh,Rx,Ry,Rh,mpE,mpN],\
					  ['SRoff m','SRoff mE','SRoff mN','SRoff mH','SRaz','SRang','s mE','s mN','s mH',\
					   'r mE','r mN','r mH','CMP mE','CMP mN'],name=Name)
	return S_out


def ODR_poly_dt_est(df_g,err_g,df_n,err_n,order=1,dfilt={'kind':1,'phz':'P'}):
	"""
	Estimate the bulk time correction for GeoRod data from some collection
	of Node data - note that the nodes should encompass the given spread of GeoRods
	to minimize the influence of lateral variations.

	1) Model fit to the GeoRod data to estimate 1+order terms
	2) Find the intercept value that minimizes data-model residuals with the Node data, including uncertainties
	3) Apply the intercept shift to a copy of GeoRod data and output the result as a new series

	"""
	# Pull desired polynomial function for curve_fit_2Derr()
	funs = {1:(inv.lin_fun,np.ones(2)),2:(inv.quad_fun,np.ones(3)),3:(inv.cube_fun,np.ones(4))}
	fun = funs[order][0]
	beta0 = funs[order][1]
	# Conduct additional filtering on data for DT calculation
	idf_g = df_g.copy()
	idf_n = df_n.copy()
	for k_ in dfilt.keys():
		idf_g = idf_g[idf_g[k_]==dfilt[k_]]
		idf_n = idf_n[idf_n[k_]==dfilt[k_]]
	# # Update beta0 intercep to rough epoch-time
	# beta0[-1] += idf_n['epoch'].values[0]
	# Update beta0 slope to ice slowness
	beta0[-2] += 1./3850.

	# Reduce data to first geo-rod as local origin
	X0g = idf_g['SRoff m'].values[0]; T0g = idf_g['epoch'].values[0]
	Xg = idf_g['SRoff m'].values - X0g
	Xn = idf_n['SRoff m'].values - X0g
	Tg = idf_g['epoch'].values - T0g
	Tn = idf_n['epoch'].values - T0g

	# Conduct fitting to GeoRod data to estimate model fit (provides curvature)
	out_g = inv.curve_fit_2Derr(fun,Xg,Tg,err_g['xsig'],err_g['tsig'],beta0=beta0)
	beta_g = out_g.beta
	# Pass to second fitting to Node data
	# And do ODR with tightly constrained beta if there are enough data to invert
	# if len(df_n) > 5:
	# 	# Lock all parameters except intercept
	# 	ifix_g = np.zeros(beta_g.shape,dtype=int) 
	# 	ifix_g[-1] += 1
	# 	out_n = inv.curve_fit_2Derr(fun,idf_n['SRoff m'].values,idf_n['epoch'].values,\
	# 								err_n['xsig'],err_n['tsig'],beta0=beta_g,ifixb=ifix_g)
	# 	DT = out_n.beta[-1] - beta_g[-1]
	# # Otherwise, take the median data-model residual for node locations
	# else:
	DT = np.nanmean(Tn - fun(out_g.beta,Xn))
	DT = np.average(Tn - fun(out_g.beta,Xn),weights=Xn**-2)
	# breakpoint()
	# Add time corrections to ALL GeoRod data
	df_g_out = pd.DataFrame({'epoch corr':df_g['epoch'].values + DT},index=df_g.index)
	return df_g_out


def smfconcat(MROOT,S_meta):
	"""
	Concatenate snuffler marker files and output a pandas.DataFrame containing key information
	from picks and events

	:: INPUTS ::
	:param MROOT: Root directory containing all pick files
	:param S_meta: Metadata Series for specific shot

	:: OUTPUT ::
	:return df_picks: Concatenated picks with metadata
	"""
	PATH =os.path.join(MROOT,S_meta['Site'],S_meta['Spread'],'shot',str(S_meta['Shot #'])) 
	SMF = os.path.join(PATH,S_meta['Pick File'])
	markers = pm.load_markers(SMF)
	# Find event markers and phase markers
	plist = []; elist = []; mtype = []; time = []; timestamp = []; t0_ref = []; phz = []; itype = [];
	kind = []; net = []; sta = []; loc = []; chan = []; ehash = []; shotno = [];
	static = []; filt_type = []; freq1 = []; freq2 = []; wffil = []; spread = [];
	# Iterate across markers
	for m_ in markers:
		# print(m_)
		if isinstance(m_,pm.PhaseMarker) or isinstance(m_,pm.EventMarker):
			time.append(m_.get_tmin())
			timestamp.append(pd.Timestamp(m_.get_tmin()*1e9))
			kind.append(m_.kind)
			ehash.append(m_.get_event_hash())
			shotno.append(S_meta['Shot #'])			
			spread.append(S_meta['Spread'])
			if S_meta['Lowpass'] > 0 and S_meta['Highpass'] > 0:
				filt_type.append('bandpass')
				freq1.append(S_meta['Lowpass'])
				freq2.append(S_meta['Highpass'])
			elif S_meta['Lowpass'] > 0 and S_meta['Highpass'] < 0:
				filt_type.append('lowpass')
				freq1.append(S_meta['Lowpass'])
				freq2.append(np.nan)
			elif S_meta['Lowpass'] < 0 and S_meta['Highpass'] > 0:
				filt_type.append('highpass')
				freq1.append(np.nan)
				freq2.append(S_meta['Highpass'])
			else:
				filt_type.append(None)
				freq1.append(None)
				freq2.append(None)
			if isinstance(m_,pm.EventMarker):
				elist.append(m_)
				mtype.append('Event')
				phz.append(None)
				net.append(None)
				sta.append(None)
				loc.append(None)
				chan.append(None)
				static.append(False)
				wffil.append(None)
				itype.append(None)
			elif isinstance(m_,pm.PhaseMarker):
				# breakpoint()
				plist.append(m_)
				mtype.append('Phase')
				phz.append(m_.get_phasename())
				inet = m_.get_nslc_ids()[0][0]
				net.append(inet)
				ista = m_.get_nslc_ids()[0][1]
				sta.append(ista)
				iloc = m_.get_nslc_ids()[0][2]
				loc.append(iloc)
				ichan = m_.get_nslc_ids()[0][3]
				chan.append(ichan)
				iwf = glob(os.path.join(PATH,'%s.%s.*.%s*.mseed'%(inet,ista,ichan)))[0]
				wffil.append(iwf)
				if ichan in ['GN3','GNZ']:
					itype.append('Node')
				else:
					itype.append('GeoRod')
				if m_.get_nslc_ids()[0][1] in list(df_SITE['Station'].unique()):
					static.append(True)
				else:
					static.append(False)
	df_picks = pd.DataFrame({'spread':spread,'shot #':shotno,'net':net,'sta':sta,'loc':loc,'chan':chan,\
							 'kind':kind,'phz':phz,'type':mtype,'time':timestamp,'epoch':time,\
							 'hash':ehash,'static':static,'filt':filt_type,'freq1':freq1,'freq2':freq2,\
							 'itype':itype,'wf file':wffil}) 	
	return df_picks


##### CORE PROCESS #####
def smf2df(MROOT,S_meta,df_SITE,df_SHOT,df_GPSc,proj_epsg='epsg:32619'):
	"""
	Concatenate snuffler marker files (smf) and shot geometry/metadata
	into a single DataFrame (df) summarizing shot-reciever timings and
	distances and associate trace file-path

	:: INPUT ::
	:type SMF: str
	:param SMF: Snuffler Marker File name
	:type shotdir: str
	:param shot: shot file identifier
	:type df_SITE: pandas.DataFrame
	:param df_SITE: dataframe containing receiver locaiton information
	:type df_SHOT: pandas.DataFrame
	:param df_SHOT: dataframe containing shot location information
	:type df_GPSc: pandas.DataFrame
	:param df_GPSc: dataframe containing GPS elevation estimates (with corrections from SmartSolos)
	:type proj_epsg: str
	:param proj_epsg: target referenceframe 

	:: OUTPUT ::
	:rtype df_out: pandas.DataFrame
	:return df_out: output dataframe with lines summarizing phase travel time and
					geometry data
	"""
	PATH =os.path.join(MROOT,S_meta['Site'],S_meta['Spread'],'shot',str(S_meta['Shot #'])) 
	SMF = os.path.join(PATH,S_meta['Pick File'])
	df_SN = df_SITE[df_SITE['Channel']=='GNZ']
	markers = pm.load_markers(SMF)
	# wflist = glob(os.path.join(PATH,'*.mseed'))
	# Find event markers and phase markers
	plist = []; elist = []; mtype = []; time = []; timestamp = []; t0_ref = []; phz = []; itype = [];
	kind = []; net = []; sta = []; loc = []; chan = []; ehash = []; shotno = []; offset_code = [];
	static = []; filt_type = []; freq1 = []; freq2 = []; wffil = []; spread = [];
	# Iterate across markers
	for m_ in markers:
		# print(m_)
		if isinstance(m_,pm.PhaseMarker) or isinstance(m_,pm.EventMarker):
			time.append(m_.get_tmin())
			timestamp.append(pd.Timestamp(m_.get_tmin()*1e9))
			kind.append(m_.kind)
			ehash.append(m_.get_event_hash())
			shotno.append(S_meta['Shot #'])
			off_code = df_SHOT[df_SHOT['Data_File']==str(S_meta['Shot #'])+'.dat']['OffsetCode'].values[0]
			offset_code.append(int(off_code/250))	
			spread.append(S_meta['Spread'])
			if S_meta['Lowpass'] > 0 and S_meta['Highpass'] > 0:
				filt_type.append('bandpass')
				freq1.append(S_meta['Lowpass'])
				freq2.append(S_meta['Highpass'])
			elif S_meta['Lowpass'] > 0 and S_meta['Highpass'] < 0:
				filt_type.append('lowpass')
				freq1.append(S_meta['Lowpass'])
				freq2.append(np.nan)
			elif S_meta['Lowpass'] < 0 and S_meta['Highpass'] > 0:
				filt_type.append('highpass')
				freq1.append(np.nan)
				freq2.append(S_meta['Highpass'])
			else:
				filt_type.append(None)
				freq1.append(None)
				freq2.append(None)
			if isinstance(m_,pm.EventMarker):
				elist.append(m_)
				mtype.append('Event')
				phz.append(None)
				net.append(None)
				sta.append(None)
				loc.append(None)
				chan.append(None)
				static.append(False)
				wffil.append(None)
				itype.append(None)
			elif isinstance(m_,pm.PhaseMarker):
				# breakpoint()
				plist.append(m_)
				mtype.append('Phase')
				phz.append(m_.get_phasename())
				inet = m_.get_nslc_ids()[0][0]
				net.append(inet)
				ista = m_.get_nslc_ids()[0][1]
				sta.append(ista)
				iloc = m_.get_nslc_ids()[0][2]
				loc.append(iloc)
				ichan = m_.get_nslc_ids()[0][3]
				chan.append(ichan)
				iwf = glob(os.path.join(PATH,'%s.%s.*.%s*.mseed'%(inet,ista,ichan)))[0]
				wffil.append(iwf)
				if ichan in ['GN3','GNZ']:
					itype.append('Node')
				else:
					itype.append('GeoRod')
				if m_.get_nslc_ids()[0][1] in list(df_SITE['Station'].unique()):
					static.append(True)
				else:
					static.append(False)
	try:
		df_picks = pd.DataFrame({'spread':spread,'shot #':shotno,'offset code':offset_code,\
								 'net':net,'sta':sta,'loc':loc,'chan':chan,\
								 'kind':kind,'phz':phz,'type':mtype,'time':timestamp,'epoch':time,\
								 'hash':ehash,'static':static,'filt':filt_type,'freq1':freq1,'freq2':freq2,\
								 'itype':itype,'wf file':wffil}) 
	except:
		print('timestamp %d'%(len(timestamp)))
		print('chan %d'%(len(chan)))
		# breakpoint()
	# Get Origin Time from Event pick
	# breakpoint()
	t0 = elist[0].get_tmin()
	# tts = np.array(time) - t0
	t0_ref = [t0]*len(df_picks)

	# Get Source Location
	NET = plist[0].get_nslc_ids()[0][0]
	SFID = str(S_meta['Shot #']) + '.dat'
	Sxyhz = df_SHOT[df_SHOT['Data_File']==SFID][['SHOT_Lon','SHOT_Lat','SHOT_elev','SHOT_dep_m']].values[0]
	Sx,Sy = gt.LL2epsg(lon=Sxyhz[0],lat=Sxyhz[1],epsg=proj_epsg)
	Sh = Sxyhz[2]
	# Get Shot Depth
	Sz = Sxyhz[3]
	# For NaN valued entries, assume average depth for whole site as shot depth
	if ~np.isfinite(Sz):
		Sz = np.nanmean(df_SHOT['SHOT_dep_m'].values)
	# Place shot at depth
	Sh -= Sz
	# Get Source-Receiver Offsets
	SRX = []; SRY = []; SRH = []; SRoff = []; SRaz = []; SRang = []; 
	CMP_mE = []; CMP_mN = []; CMP_mH = [];

	for i_ in range(len(df_picks)):
		if df_picks.iloc[i_]['type']=='Phase' and df_picks.iloc[i_]['static']:
			# Get specific pick entry
			iS_PICK = df_picks.iloc[i_]
			# Filter for relevant station
			iS_SITE = df_SITE[(df_SITE['Network']==iS_PICK['net']) &\
							   (df_SITE['Station'] == iS_PICK['sta']) &\
							   (df_SITE['Channel'] == iS_PICK['chan'])]
			# Get receiver location in UTM
			iRX,iRY = gt.LL2epsg(lon=iS_SITE['Longitude'].values[0],lat=iS_SITE['Latitude'].values[0],epsg=proj_epsg)
			# Get source-receiver offset vector elements
			iDX = (Sx - iRX)
			iDY = (Sy - iRY)
			iDH = (Sh - iS_SITE['Elevation'].values[0])
			SRX.append(iDX)
			SRY.append(iDY)
			SRH.append(iDH)
			SRoff.append(np.sqrt(iDX**2 + iDY**2 + iDH**2))
			# Get Common Mid Point Coordinates
			iCMPX = np.mean([Sx,iRX])
			iCMPY = np.mean([Sy,iRY])
			iCMPH = np.mean([Sh,Sh - iDH])
			CMP_mE.append(iCMPX)
			CMP_mN.append(iCMPY)
			CMP_mH.append(iCMPH)
			# Get Source-Receiver Azimuth (for human-readable) and rotation angle (for calculation)
			iSRaz,iSRang = gt.cartesian_azimuth(iDX,iDY)
			SRaz.append(iSRaz)
			SRang.append(iSRang)

		# If pick is on moving shot-co-located receivers
		elif df_picks.iloc[i_]['sta'] in ['SR4K','SR2K']:
			# Set offsets at a standard stand-off of 3 m laterally, and shot depth vertically
			SRX.append(3/np.sqrt(2))
			SRY.append(3/np.sqrt(2))
			SRH.append(Sh)
			SRoff.append(3)
			iSRaz,iSRang = gt.cartesian_azimuth(1,1)
			SRaz.append(iSRaz)
			SRang.append(iSRang)
			CMP_mE.append(Sx + 1.5/np.sqrt(2))
			CMP_mN.append(Sy + 1.5/np.sqrt(2))
			# In this one case, use IDW to fill in 
			CMP_mH.append(gt.simple_idw(df_SN['UTM19N Easting'].values,df_SN['UTM19N Northing'].values,\
					  					df_SN['Elevation'].values,Sx + 1.5/np.sqrt(2),\
										Sy + 1.5/np.sqrt(2),power=2)[0])

		# If there is a Geode Colocated Recorder (2kHz) pick
		elif df_picks.iloc[i_]['sta'] == 'GCR2K':
			Rxyz = df_SHOT[df_SHOT['Data_File']==SFID][['REC_lon','REC_lat','REC_ele']].values[0]
			iRX,iRY = gt.LL2epsg(lon=Rxyz[0],lat=Rxyz[1],epsg=proj_epsg)
			iDX = (Sx - iRX)
			iDY = (Sy - iRY)
			iDH = (Sh - Rxyz[2])
			iCMPX = np.mean([Sx,iRX])
			iCMPY = np.mean([Sy,iRY])
			iCMPH = np.mean([Sh,Rxyz[2]])
			iSRaz,iSRang = gt.cartesian_azimuth(iDX,iDY)
			SRX.append(iDX)
			SRY.append(iDY)
			SRH.append(iDH)
			SRoff.append(np.sqrt(iDX**2 + iDY**2 + iDH**2))
			SRaz.append(iSRaz)
			SRang.append(iSRang)
			CMP_mE.append(iCMPX)
			CMP_mN.append(iCMPY)
			CMP_mH.append(iCMPH)

		else:
			SRX.append(np.nan)
			SRY.append(np.nan)
			SRH.append(np.nan)
			SRoff.append(np.nan)
			SRaz.append(np.nan)
			SRang.append(np.nan)
			CMP_mE.append(np.nan)
			CMP_mN.append(np.nan)
			CMP_mH.append(np.nan)
	# Append to distances to picks
	df_picks = pd.concat([df_picks,pd.DataFrame({"SRoff m":SRoff,"SR mE":SRX,"SR mN":SRY,'SR mH':SRH,\
												 'az rad':SRaz,'ang rad':SRang,'CMP mE':CMP_mE,'CMP mN':CMP_mN,\
												 'CMP mH':CMP_mH,"t0 ref":t0_ref},index=df_picks.index)],\
						 axis=1,ignore_index=False)
	# # Append to distances to picks
	# df_picks = pd.concat([df_picks,pd.DataFrame({"SRoff m":SRoff,"SR mE":SRX,"SR mN":SRY,'SR mELE':SRH,\
	# 											 'az rad':SRaz,'ang rad':SRang,'CMP mE':CMP_mE,'CMP mN':CMP_mN,\
	# 											 "tt sec":tts,"t0 ref":t0_ref},index=df_picks.index)],\
	# 					 axis=1,ignore_index=False)

	return df_picks






##### DRIVERS #####
def run_pick_concat():
	"""
	Wrapper method to run smf2db()
	"""
	# def run_smf2df(ROOT,df_META,df_SITE,df_SHOT,proj_epsg='epsg:32619'):
	df_out = pd.DataFrame()
	# Iterate across shot entries in MetaData
	for i_ in range(len(df_META)):
		# Get subset metadata series
		S_i = df_META.iloc[i_,:]
		if S_i['Include']:
			# Conduct 
			df_iout = smf2df(MROOT,S_i,df_SITE,df_SHOT,df_GPSc)
			df_out = pd.concat([df_out,df_iout],axis=0,ignore_index=True)
	return df_out

##### ACTUAL PROCESSING #####
ROOT = os.path.join('..','..','..','..','..')
MROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','Corrected_t0')
OROOT = os.path.join(ROOT,'processed_data','Hybrid_Seismic','VelCorrected_t0')
ICSV = os.path.join(MROOT,'Amplitude_Pick_File_Metadata_v5.csv')

# SHOT = os.path.join(ROOT,'processed_data','Active_Seismic','Master_Shot_Record_QCd.csv')
SHOT = os.path.join(ROOT,'processed_data','Active_Seismic','IL_Shot_Record_QCd_ELE_corr.csv')
# SITE = os.path.join(ROOT,'data','Combined_SITE_Table.csv')
SITE = os.path.join(ROOT,'processed_data','IL_SITE_Table_ELE_corr.csv')
GPSc = os.path.join(ROOT,'processed_data','GPS','IL_Elevation_Corrected_GPS_Tracks.csv')

## Load METADATA ##
df_META = pd.read_csv(ICSV)
df_META = df_META[df_META['Site']=='Hiawatha']
## Load SHOT Locations ##
df_SHOT = pd.read_csv(SHOT)
df_SHOT = df_SHOT[df_SHOT['Site']=='Hiawatha']
## Load SITE Locations ##
df_SITE = pd.read_csv(SITE,parse_dates=['Starttime','Endtime'])
df_SITE = df_SITE[df_SITE['Network']=='HI']
## Load GPSc Data ##
df_GPSc = pd.read_csv(GPSc,parse_dates=['time'],index_col=[0])

print('==== Running Phase Pick Concatenation ====')
# 
df = run_pick_concat()
pick_err = 2e-3
gx_err = 1.
nx_err = 2.
fit_poly = 3
isplot = True

### SURVEY SPECIFIC PROCESSING ARGUMENTS ###
# Define Reference Stations for Each Shot
REF_NODES = {'NS01':['PR10','PR11','PR12'],'NS02':['PR11','PR12','PR13'],\
			 'NS03':['PR12','PR13','PR07'],'WE01':['PR08','PR07','PR06'],\
			 'WE02':['PR07','PR06','PR05'],'WE03':['PR06','PR05','PR04']}

# Zero-offset shots to process only using channels 25Z-48Z 
ZO_TROUBLE_LIST = [307,308,319,352,396]

df_tc = pd.DataFrame()
# Need to correct the below to filter for kind = 1 and phz = P
for SP_,SH_ in df[['spread','shot #']].value_counts().index:
	print('Processing %s %s'%(SP_,SH_))
	# Subset phase DataFrame
	df_i = df[(df['spread']==SP_)&(df['shot #']==SH_)]
	# Get Shot Index in Sequence
	offset_code = df_i['offset code'].unique()[0]
	# Subset GeoRod Entries
	df_ig = df_i[df_i['itype']=='GeoRod']
	# Subset Node Entries
	df_in = df_i[(df_i['itype']=='Node')&(df_i['sta'].isin(REF_NODES[SP_]))]
	# Create dataframe with Node errors
	err_n = pd.DataFrame({'xsig':np.ones(len(df_in),)*nx_err,'tsig':np.ones(len(df_in),)*pick_err})

	# if this shot is in the middle of the spread, do separate corrections for each side of the array
	# This counteracts blow-up of small distance uncertainties in sorting out timing corrections
	if offset_code == 0 and SH_ not in ZO_TROUBLE_LIST:
		for P_ in range(2):
			# Create station sublists
			if P_ == 0:
				sta_list = [x for x in df_ig['chan'] if int(x[:2]) <= 24]
			else:
				sta_list = [x for x in df_ig['chan'] if int(x[:2]) > 24]
			# Apply filter
			df_igP = df_ig[df_ig['chan'].isin(sta_list)]
			# Proceed with processing
			# Create dataframe with GeoRod errors
			err_gP = pd.DataFrame({'xsig':np.ones(len(df_igP),)*gx_err,'tsig':np.ones(len(df_igP),)*pick_err})
			# Conduct time corrections
			df_tcorr = ODR_poly_dt_est(df_igP,err_gP,df_in,err_n,order=fit_poly)
			# Vertically concat
			df_tc = pd.concat([df_tc,df_tcorr],axis=0,ignore_index=False)
			if isplot:
				if P_ == 0:
					plt.figure()
				# Plot time-corrected epochs
				plt.plot(df_i[df_i.index.isin(df_tcorr.index)]['SRoff m'].values,df_tcorr['epoch corr'].values,'s',\
						 alpha=0.2,label='Corrected Data %d'%(P_))
		if isplot:
			# Plot original epochs
			plt.plot(df_i['SRoff m'].values,df_i['epoch'].values,'o',label='Uncorrected Data')
			plt.plot(df_in['SRoff m'].values,df_in['epoch'].values,'.',label='Reference Picks')
			plt.title('%s %s'%(SP_,SH_))
			plt.legend()
	# Just work with the channels 25Z-48Z for correction
	elif offset_code == 0 and SH_ in ZO_TROUBLE_LIST:
		sta_list = [x for x in df_ig['chan'] if int(x[:2]) > 24]
		# Apply filter
		df_igP = df_ig[df_ig['chan'].isin(sta_list)]
		# Get inverse filtered
		df_igR = df_ig[~df_ig['chan'].isin(sta_list)]
		# Proceed with processing
		# Create dataframe with GeoRod errors
		err_gP = pd.DataFrame({'xsig':np.ones(len(df_igP),)*gx_err,'tsig':np.ones(len(df_igP),)*pick_err})
		# Conduct time corrections
		df_tcorr = ODR_poly_dt_est(df_igP,err_gP,df_in,err_n,order=fit_poly)
		dt_corr = np.mean(df_igP['epoch'].values - df_tcorr.values[:,0])
		# Generate corrections for un-included data in ODR procesing
		df_tcorr_sup = pd.DataFrame({'epoch corr':df_igR['epoch'].values - dt_corr},index=df_igR.index)
		df_tc = pd.concat([df_tc,df_tcorr,df_tcorr_sup],axis=0,ignore_index=False)
		if isplot:
			plt.figure()
			# Plot time-corrected epochs
			plt.plot(df_i[df_i.index.isin(df_tcorr.index)]['SRoff m'].values,df_tcorr['epoch corr'].values,'s',label='Corrected Data')
			# Plot original epochs
			plt.plot(df_i['SRoff m'].values,df_i['epoch'].values,'o',label='Uncorrected Data')
			plt.plot(df_in['SRoff m'].values,df_in['epoch'].values,'.',label='Reference Picks')
			plt.title('%s %s'%(SP_,SH_))
			plt.legend()
	# Otherwise, process all georods - location uncertainties tend to blow-up less
	else:	
		# Create dataframe with GeoRod errors
		err_g = pd.DataFrame({'xsig':np.ones(len(df_ig),)*gx_err,'tsig':np.ones(len(df_ig),)*pick_err})
		# Conduct time corrections
		df_tcorr = ODR_poly_dt_est(df_ig,err_g,df_in,err_n,order=fit_poly)
		# Vertically concat
		df_tc = pd.concat([df_tc,df_tcorr],axis=0,ignore_index=False)

	# breakpoint()
		if isplot:
			plt.figure()
			# Plot time-corrected epochs
			plt.plot(df_i[df_i.index.isin(df_tcorr.index)]['SRoff m'].values,df_tcorr['epoch corr'].values,'s',label='Corrected Data')
			# Plot original epochs
			plt.plot(df_i['SRoff m'].values,df_i['epoch'].values,'o',label='Uncorrected Data')
			plt.plot(df_in['SRoff m'].values,df_in['epoch'].values,'.',label='Reference Picks')
			plt.title('%s %s'%(SP_,SH_))
			plt.legend()
# Fill out Node entries for epoch corr
df_tc = pd.concat([df_tc,pd.DataFrame({'epoch corr':df[df['itype']=='Node']['epoch'].values},\
									   index=df[df['itype']=='Node'].index)])
# Update Column Naming
df_tc = df_tc.rename(columns={'epoch corr':'epoch'})
df = df.rename(columns={'epoch':'epoch old'})
# Append corrected times to DataFrame
df = pd.concat([df,df_tc],axis=1,ignore_index=False)
# Calculate travel time
df_tt = pd.DataFrame({'tt sec':df['epoch'].values - df['t0 ref'].values},index=df.index)

df = pd.concat([df,df_tt],axis=1,ignore_index=False)
# Write DataFrame
if isplot: 	
	plt.show()

# Clear out spurious Event line entries, if any
df = df[df['type']=='Phase']

df.to_csv(os.path.join(OROOT,'Inglefield_Land','Corrected_Phase_Picks_v5_ele_MK2_pfO%d.csv'%(fit_poly)),header=True,index=False)

# ### PRUDHOE DOME PROCESSING ###

# # Create holder for updated travel times
# df_up = df.copy()

# shot = []; itype = []; ttype = []; Shat = []; t0hat = []; oSS = []; oSt0 = []; ot0t0 = [];
# rmin = []; rmax = []; rmean = []; gdt = []; resL2 = []; kind = [];
# plt.figure()
# for S_ in df['shot #'].unique():
# 	for K_ in [1]:
# 		IGP1 = (df['shot #']==S_) & (df['itype']=='GeoRod') &\
# 			   (df['phz']=='P') & (df['kind']==K_)
# 		r_min = df[IGP1]['SRoff m'].min()
# 		r_max = df[IGP1]['SRoff m'].max()
# 		# Subset nodal data for selected offsets, first break picks on direct arrivals
# 		INP1R = (df['SRoff m'] >= r_min - 150) & \
# 				(df['SRoff m'] <= r_max + 150) & \
# 				(df['itype']=='Node') & (df['phz']=='P') &\
# 				(df['phz']=='P') & (df['kind']==K_)

# 		# Model S & t0 from Node data subset for all data
# 		imod_n,icov_n = np.polyfit(df[INP1R]['SRoff m'].values,df[INP1R]['tt sec'].values,\
# 								   1,w=np.ones(np.sum(INP1R))*(1./3e-3),cov='unscaled')
# 		# Calculate residuals
# 		ires_n = calc_resid(df[INP1R]['SRoff m'].values,df[INP1R]['tt sec'].values,imod_n)

# 		resL2.append(np.linalg.norm(ires_n,2))
# 		shot.append(S_)
# 		itype.append('Node')
# 		ttype.append('V0 all')
# 		Shat.append(imod_n[0])
# 		t0hat.append(imod_n[1])
# 		oSS.append(icov_n[0,0])
# 		oSt0.append(icov_n[0,1])
# 		ot0t0.append(icov_n[1,1])
# 		rmin.append(r_min)
# 		rmax.append(r_max)
# 		rmean.append(np.nanmean(df[IGP1]['SRoff m'].values))
# 		gdt.append(0)
# 		kind.append(K_)


# 		# Filter out large residuals
# 		INP1RR = np.abs(ires_n) < 5.*np.std(ires_n)
# 		# Run modeling on cleaned dataset
# 		jmod_n,jcov_n = np.polyfit(df[INP1R][INP1RR]['SRoff m'].values,\
# 								   df[INP1R][INP1RR]['tt sec'].values,\
# 								   1,w=np.ones(np.sum(INP1RR))*(1./3e-3),cov='unscaled')
# 		jres_n = calc_resid(df[INP1R][INP1RR]['SRoff m'].values,\
# 							df[INP1R][INP1RR]['tt sec'].values,imod_n)
# 		plt.plot(df[INP1R][INP1RR]['SRoff m'].values,\
# 				 df[INP1R][INP1RR]['tt sec'].values,'.',label='Shot %d'%(S_),\
# 				 alpha=0.25)
# 		print(df[INP1R][~INP1RR])
# 		plt.plot(df[INP1R][~INP1RR]['SRoff m'].values,\
# 				 df[INP1R][~INP1RR]['tt sec'].values,'x',label='Shot %d'%(S_),\
# 				 alpha=0.25)
		
# 		resL2.append(np.linalg.norm(jres_n,2))
# 		shot.append(S_)
# 		itype.append('Node')
# 		ttype.append('V0 res filt')
# 		Shat.append(jmod_n[0])
# 		t0hat.append(jmod_n[1])
# 		oSS.append(jcov_n[0,0])
# 		oSt0.append(jcov_n[0,1])
# 		ot0t0.append(jcov_n[1,1])
# 		rmin.append(r_min)
# 		rmax.append(r_max)
# 		rmean.append(np.nanmean(df[IGP1]['SRoff m'].values))
# 		gdt.append(0)
# 		kind.append(K_)

# 		# Model linear fits to georod data subset
# 		imod_g,icov_g = np.polyfit(df[IGP1]['SRoff m'].values,df[IGP1]['tt sec'].values,\
# 								   1,w=np.ones(np.sum(IGP1))*(1./3e-3),cov='unscaled')
# 		resL2.append(np.linalg.norm(calc_resid(df[IGP1]['SRoff m'].values,\
# 										   df[IGP1]['tt sec'].values,imod_g),2))

# 		### CALCULATE TIME CORRECTIONS FOR GEOROD DATA
# 		# Use the GeoRod model mean values as adjustment point
# 		dt_g3 = np.poly1d(jmod_n)(np.nanmean(df[IGP1]['SRoff m'].values)) - \
# 				np.poly1d(imod_g)(np.nanmean(df[IGP1]['SRoff m'].values))
# 		# Apply corrections to all georod data
# 		IND = (df_up['shot #']==S_) & (df_up['itype']=='GeoRod')
# 		# Update traveltime values
# 		df_up.loc[IND,'tt sec'] += dt_g3
# 		# Update phase arrivial epoch times
# 		df_up.loc[IND,'epoch'] -= dt_g3

# 		# Compile into
# 		shot.append(S_)
# 		itype.append('GeoRod')
# 		ttype.append('V0 all')
# 		Shat.append(imod_g[0])
# 		t0hat.append(imod_g[1])
# 		oSS.append(icov_g[0,0])
# 		oSt0.append(icov_g[0,1])
# 		ot0t0.append(icov_g[1,1])
# 		rmin.append(r_min)
# 		rmax.append(r_max)
# 		rmean.append(np.nanmean(df[IGP1]['SRoff m'].values))
# 		gdt.append(dt_g3)
# 		kind.append(K_)

# ### THIS DATAFRAME HOLDS SHOT-GATHER VELOCITY MODEL DATA ###
# df_MOD = pd.DataFrame({'shot #':shot,'itype':itype,'kind':kind,'ttype':ttype,'rmin':rmin,'rmean':rmean,'rmax':rmax,\
# 					   'S sec/m':Shat,'t0 sec':t0hat,'oSS':oSS,'oSt0':oSt0,'ot0t0':ot0t0,'dt g sec':gdt,\
# 					   'res L2':resL2})

# ### WRITE UPDATED
# try:
# 	os.makedirs(os.path.join(OROOT,'Prudhoe_Dome','Shot'))
# except:
# 	pass

# ### WRITE MODELS TO FILE ###
# df_MOD.to_csv(os.path.join(OROOT,'Prudhoe_Dome','Shot','Direct_Arrival_Velocity_Models.csv'),header=True,index=False)
# ### WRITE UPDATED PHASES TO FILE ###
# df_up.to_csv(os.path.join(OROOT,'Prudhoe_Dome','VelCorrected_Phase_Picks.csv'),header=True,index=False)



##### END #####