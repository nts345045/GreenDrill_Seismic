"""
:module: S1_GPS_elevation_processing.py
:purpose: Combine GPS observations to create a basis for correcting GeoRod and shot
			elevation estimates via Inverse Distance Weighting. This provides the
			necessary data for static corrections and pseudo-section generation
			from ice-column structure in later steps.
			
:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
sys.path.append(os.path.join('..','..'))
import util.GeometryTools as gt

## DIRECTORY/FILE MAPPING SECTION ##
ROOT = os.path.join('..','..','..','..','..')
SITE = os.path.join(ROOT,'data','Combined_SITE_Table.csv')
TRACK = os.path.join(ROOT,'data','GPS','Prudhoe_Tracks_Raw.csv')
SHOT = os.path.join(ROOT,'processed_data','Active_Seismic','Master_Shot_Record_QCd.csv')
ODIR = os.path.join(ROOT,'processed_data','GPS')
OSITE = os.path.join(ROOT,'processed_data','Combined_SITE_Table_ELE_corr.csv')
OSHOT = os.path.join(ROOT,'processed_data','Active_Seismic','Master_Shot_Record_QCd_ELE_corr.csv')

## DATA LOADING SECTION ##
df_SITE = pd.read_csv(SITE,parse_dates=['Starttime','Endtime'])
# Fiter down to just SmartSolos
df_SITE = df_SITE[(df_SITE['Network']=='PL')]
# Load GPS Tracks
df_GPS = pd.read_csv(TRACK,parse_dates=['time'],index_col=[0])
# Trim off extraneous (empty) fields
df_GPS = df_GPS[['lat','lon','ele']]
# Import Shot Data
df_SHOT = pd.read_csv(SHOT)
# Filter for just Prudhoe Dome
df_SHOT = df_SHOT[df_SHOT['Site']=='Prudhoe']

### START OF PROCESSING ###
# Get UTM19N equivalents for GPS tracks
mE,mN = gt.LL2epsg(df_GPS['lon'].values,df_GPS['lat'].values)
df_GPS = pd.concat([df_GPS,pd.DataFrame({'UTM19N mE':mE,'UTM19N mN':mN},index=df_GPS.index)],\
				   axis=1,ignore_index=False)

# Define sampling window length for doing corrections
DT_corr = pd.Timedelta(1,unit='hour')
# Define periods to process
track_pds=[(pd.Timestamp("2022-04-22T10:20+0000"),pd.Timestamp("2022-04-22T17:35+000")),\
		   (pd.Timestamp("2022-04-22T18:00+0000"),pd.Timestamp("2022-04-23T12:35+0000")),\
		   (pd.Timestamp("2022-04-26T19:23+0000"),pd.Timestamp("2022-04-26T19:57+0000")),\
		   (pd.Timestamp("2022-04-27T02:17+0000"),pd.Timestamp("2022-04-27T04:10+0000")),\
		   (pd.Timestamp("2022-04-27T12:15+0000"),pd.Timestamp("2022-04-27T19:50+0000"))]

## RUN INITIAL PROCESSING (SUBSETTING) ##
# Trim down GPS track data to specified periods
df_GPSf = pd.DataFrame()
for t1_,t2_ in track_pds:
	idf = df_GPS[(df_GPS.index >= t1_)&(df_GPS.index <= t2_)]
	df_GPSf = pd.concat([df_GPSf,idf],axis=0,ignore_index=False)

 # Initial Plots
plt.figure()
plt.subplot(211)
plt.plot(df_GPS['ele'],label='RAW')
plt.plot(df_GPSf['ele'],label='Trimmed')
plt.ylabel('Elevation (m)')
plt.xlabel('UTC Date Time')
plt.title('Handheld GPS Data')
plt.legend()

plt.subplot(224)
plt.scatter(df_GPSf['lon'].values,df_GPSf['ele'].values,c=df_GPSf['lat'])
plt.colorbar()
plt.plot(df_SITE['Longitude'],df_SITE['Elevation'],'rv')
plt.plot(df_SHOT['SHOT_Lon'],df_SHOT['SHOT_elev'],'m*',label='Shots (raw)')
plt.xlabel('Longitude [$^oE$]')
plt.ylabel('Elevation [m ASL]')

plt.subplot(223)
plt.scatter(df_GPSf['lat'].values,df_GPSf['ele'].values,c=df_GPSf['lon'])
plt.colorbar()
plt.plot(df_SITE['Latitude'],df_SITE['Elevation'],'rv',label='Receivers (raw)')
plt.plot(df_SHOT['SHOT_Lat'],df_SHOT['SHOT_elev'],'m*',label='Shots (raw)')
plt.xlabel('Latitude [$^oN$]')
plt.ylabel('Elevation [m ASL]')

plt.figure()
## PROCESS DATA IN DT_corr length windows ##
corrections = {}
df_ele_corr = pd.DataFrame()
for t1_,t2_ in track_pds:
	for i_ in range(int(np.floor((t2_ - t1_)/DT_corr))):
		it1_ = t1_ + i_*DT_corr
		it2_ = t1_ + (i_ + 1)*DT_corr
		# Filter GPS data by time
		idf_G = df_GPSf[(df_GPSf.index >= it1_)&(df_GPSf.index < it2_)]
		# Slightly smooth/perturb data (prevents infinite weighting from same point)
		idf_Gs = idf_G.rolling(pd.Timedelta(1,unit='min')).mean()
		# Use IDW smoothing on perturbed data locations
		z_smooth = gt.llh_idw(idf_G['lon'].values,idf_G['lat'].values,idf_G['ele'].values,\
							  idf_Gs['lon'].values,idf_Gs['lat'].values,power=1)
		# Create iterative visualization of data subsetting and smoothing
		plt.subplot(211)
		plt.plot(idf_G['lon'].values,z_smooth,'.')
		plt.subplot(212)
		plt.plot(idf_G.index,z_smooth,'.')

		## Get subset handheld GPS data and cross-reference 
		# Get bounding box from handheld GPS data
		S_ll = idf_G[['lon','lat']].min()
		S_ur = idf_G[['lon','lat']].max()
		# Get reference data (from Nodes) that's in box
		idf_S = df_SITE[(df_SITE['Longitude'] >= S_ll['lon']) & \
						(df_SITE['Longitude'] <= S_ur['lon']) & \
						(df_SITE['Latitude'] >= S_ll['lat']) & \
						(df_SITE['Latitude'] <= S_ur['lat']) &\
						(df_SITE['Channel']=='GNZ')]
		# If there's at least one node in the sampling area
		if len(idf_S) > 0:
			# Get "Model Values"
			idw_ele = gt.llh_idw(idf_G['lon'].values,idf_G['lat'].values,z_smooth,\
								 idf_S['Longitude'].values,idf_S['Latitude'].values,\
								 power=2)
			# Get difference in elevations
			idZv = idf_S['Elevation'].values - idw_ele
			# Summarize corrections
			line = [np.mean(idZv),np.median(idZv),np.std(idZv),np.min(idZv),np.max(idZv),len(idZv)]
			corrections.update({it1_:line})
		# Otherwise, throw out the data - this gets rid of the big point cloud at basecamp
		else:
			idZv = np.nan

		# Apply corrections
		df_c = pd.DataFrame(z_smooth + np.mean(idZv),index=idf_G.index,columns=['mean(dZ)'])
		df_ele_corr = pd.concat([df_ele_corr,df_c],axis=0,ignore_index=False)
plt.subplot(212)
plt.xlabel('Longitude [$^oE$]')
plt.ylabel('Elevation [m ASL]')
plt.title('Smoothed, Segmented GPS Data (Pre Elevation Correction)')
plt.subplot(211)
plt.xlabel('Latitude [$^oN$]')
plt.ylabel('Elevation [m ASL]')

# Combine GPS observations, adding a new column
df_GPSc = df_GPSf.copy()
df_GPSc = pd.concat([df_GPSc,df_ele_corr],axis=1,ignore_index=False)
df_GPSc = df_GPSc[df_GPSc['mean(dZ)'].notna()]


### GEOROD AND STATION ELEVATION CORRECTION SECTION ###

# Write corrected data to file
df_GPSc.to_csv(os.path.join(ODIR,'Prudhoe_Elevation_Corrected_GPS_Tracks.csv'),\
			   header=True,index=True)

# Update SITE data
onlynodes = False
# Subset SITE by instrument type
df_GR = df_SITE[~df_SITE['Channel'].isin(['GNZ','GN1','GN2'])]
df_NO = df_SITE[df_SITE['Channel'].isin(['GNZ','GN1','GN2'])]

# Combine Elevation Corrected GPS data and Node data to further reinforce elevation model
lon = np.r_[df_NO['Longitude'].values,df_GPSc['lon'].values]
lat = np.r_[df_NO['Latitude'].values,df_GPSc['lat'].values]
ele = np.r_[df_NO['Elevation'].values,df_GPSc['mean(dZ)'].values]
# Calculate IDW interpolated elevations for GeoRod locations
g_ELE_corr = gt.llh_idw(lon,lat,ele,df_GR['Longitude'].values,df_GR['Latitude'].values,power=2)

# Overwrite Elevations
df_GR.loc[:,'Elevation'] = g_ELE_corr
## Create new SITE table ##
df_SITE2 = pd.concat([df_NO,df_GR],axis=0,ignore_index=False)


# Update SHOT data
s_ELE_corr = gt.llh_idw(lon,lat,ele,df_SHOT['SHOT_Lon'].values,df_SHOT['SHOT_Lat'].values,power=2)
# Update Geode Co-Located Node Locations
r_ELE_corr = gt.llh_idw(lon,lat,ele,df_SHOT['REC_lon'].values,df_SHOT['REC_lat'].values,power=2)

## Re-Insert
df_SHOT2 = df_SHOT.copy()
df_SHOT2.loc[:,'SHOT_elev'] = s_ELE_corr
df_SHOT2.loc[:,'REC_ele'] = r_ELE_corr

# Update 


vbnds = {'vmin':1260,'vmax':1320}
plt.figure()
plt.subplot(223)
plt.scatter(df_GPSc['lon'].values,df_GPSc['mean(dZ)'].values,c=df_GPSc['lat'])
plt.plot(df_SITE['Longitude'],df_SITE['Elevation'],'rv',label='Input GeoRod')
plt.plot(df_GR['Longitude'],df_GR['Elevation'],'cv',label='Corrected GeoRod')
plt.plot(df_SHOT['SHOT_Lon'],df_SHOT['SHOT_elev'],'r*',label='Input Shot')
plt.plot(df_SHOT2['SHOT_Lon'],df_SHOT2['SHOT_elev'],'b*',label='Corrected Shot')
plt.plot(df_SHOT['REC_lon'],df_SHOT['REC_ele'],'rs',label='Input GCR2K')
plt.plot(df_SHOT2['REC_lon'],df_SHOT2['REC_ele'],'bs',label='Corrected GCR2K')
plt.legend()
plt.xlabel('Longitude [$^oE$]')
plt.ylabel('Elevation [m ASL]')
plt.colorbar()

plt.subplot(221)
plt.scatter(df_GPSc['lat'].values,df_GPSc['mean(dZ)'].values,c=df_GPSc['lon'])
plt.plot(df_SITE['Latitude'],df_SITE['Elevation'],'rv',label='Input GeoRod')
plt.plot(df_GR['Latitude'],df_GR['Elevation'],'cv',label='Corrected GeoRod')
plt.plot(df_SHOT['SHOT_Lat'],df_SHOT['SHOT_elev'],'r*',label='Input Shot')
plt.plot(df_SHOT2['SHOT_Lat'],df_SHOT2['SHOT_elev'],'b*',label='Corrected Shot')
plt.plot(df_SHOT['REC_lat'],df_SHOT['REC_ele'],'rs',label='Input GCR2K')
plt.plot(df_SHOT2['REC_lat'],df_SHOT2['REC_ele'],'bs',label='Corrected GCR2K')
plt.xlabel('Latitude [$^oN$]')
plt.ylabel('Elevation [m ASL]')
plt.colorbar()


plt.subplot(122)
# Plot GPS Tracks
plt.scatter(df_GPSc['lon'].values,df_GPSc['lat'].values,s=1,c=df_GPSc['mean(dZ)'].values,**vbnds)
# Plot All Static Recorders (GeoRods and most Nodes)
plt.scatter(df_SITE2['Longitude'].values,df_SITE2['Latitude'].values,\
			c=df_SITE2['Elevation'].values,marker='v',s=36,ec='k',**vbnds)
# Plot All Shots
plt.scatter(df_SHOT2['SHOT_Lon'].values,df_SHOT2['SHOT_Lat'].values,\
			c=df_SHOT2['SHOT_elev'].values,marker='*',s=36,ec='k',**vbnds)
# Plot all Mobile Nodes
plt.scatter(df_SHOT2['REC_lon'].values,df_SHOT2['REC_lat'].values,\
			c=df_SHOT2['REC_ele'].values,marker='s',s=36,ec='k',**vbnds)
plt.colorbar()
plt.xlabel('Longitude [$^oE$]')
plt.ylabel('Latitude [$^oN$]')

# Write UPDATED SITE to disk
df_SITE2.to_csv(OSITE,header=True,index=False)
# Write UPDATED SHOT to disk
df_SHOT2.to_csv(OSHOT,header=True,index=False)


# NTS: Chose to abandon this last in favor of  using
# IDW methods developed here and elevation-corrected
# GPS data to do IDW elevations for GeoRods
# ## CONDUCT RBF Interpolation

# df_GPScf = df_GPSc[df_GPSc['mean(dZ)'].notna()]
# xq = df_GPScf['UTM19N mE'].values
# yq = df_GPScf['UTM19N mN'].values
# zq = df_GPScf['mean(dZ)'].values


# ZI,XI = gt.interp_3D_Rbf(yq,xq,zq,nx=200j,ny=200j)






plt.show()


# Initial Plots
# plt.figure()
# plt.subplot(211)
# plt.plot(df_GPS['ele'],label='RAW')
# plt.plot(df_GPSf['ele'],label='Trimmed')
# plt.ylabel('Elevation (m)')

# plt.subplot(223)
# plt.scatter(df_GPSf['lon'].values,df_GPSf['ele'].values,c=df_GPSf['lat'])
# plt.plot(df_SITE['Longitude'],df_SITE['Elevation'],'rv')
# plt.subplot(224)
# plt.scatter(df_GPSf['lat'].values,df_GPSf['ele'].values,c=df_GPSf['lon'])
# plt.plot(df_SITE['Latitude'],df_SITE['Elevation'],'rv')

# Plots after calculating corrections
# plt.figure()
# plt.subplot(221)
# plt.scatter(df_GPSc['lon'].values,df_GPSc['mean(dZ)'].values,c=df_GPSc['lat'])
# plt.plot(df_SITE['Longitude'],df_SITE['Elevation'],'rv')
# plt.subplot(223)
# plt.scatter(df_GPSc['lat'].values,df_GPSc['mean(dZ)'].values,c=df_GPSc['lon'])
# plt.plot(df_SITE['Latitude'],df_SITE['Elevation'],'rv')


# plt.subplot(122)
# plt.scatter(df_GPSc['lon'].values,df_GPSc['lat'].values,c=df_GPSc['mean(dZ)'].values)


