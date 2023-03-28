"""
:module: S1_GPS_DEM.py
:purpose: Create a Digitial Elevation Model of the glacier surface from GPS data
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


ROOT = os.path.join('..','..','..','..','..')
SITE = os.path.join(ROOT,'data','Combined_SITE_Table.csv')
TRACK = os.path.join(ROOT,'data','GPS','Prudhoe_Tracks_Raw.csv')
ODIR = os.path.join(ROOT,'processed_data','GPS')

df_SITE = pd.read_csv(SITE,parse_dates=['Starttime','Endtime'])
# Fiter down to just SmartSolos
df_SITE = df_SITE[(df_SITE['Network']=='PL')]
# Load GPS Tracks
df_GPS = pd.read_csv(TRACK,parse_dates=['time'],index_col=[0])
# Trim off extraneous (empty) fields
df_GPS = df_GPS[['lat','lon','ele']]

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

## RUN INITIAL VISUALIZATION ##
# Trim down GPS track data to specified periods
df_GPSf = pd.DataFrame()
for t1_,t2_ in track_pds:
	idf = df_GPS[(df_GPS.index >= t1_)&(df_GPS.index <= t2_)]
	df_GPSf = pd.concat([df_GPSf,idf],axis=0,ignore_index=False)



## Start To Run Processing ##
corrections = {}
df_ele_corr = pd.DataFrame()
for t1_,t2_ in track_pds:
	for i_ in range(int(np.floor((t2_ - t1_)/DT_corr))):
		it1_ = t1_ + i_*DT_corr
		it2_ = t1_ + (i_ + 1)*DT_corr
		# Filter GPS data by time
		idf_G = df_GPSf[(df_GPSf.index >= it1_)&(df_GPSf.index < it2_)]
		# Get bounding box
		S_ll = idf_G[['lon','lat']].min()
		S_ur = idf_G[['lon','lat']].max()
		# Get reference data that's in box
		idf_S = df_SITE[(df_SITE['Longitude'] >= S_ll['lon']) & \
						(df_SITE['Longitude'] <= S_ur['lon']) & \
						(df_SITE['Latitude'] >= S_ll['lat']) & \
						(df_SITE['Latitude'] <= S_ur['lat']) &\
						(df_SITE['Channel']=='GNZ')]
		if len(idf_S) > 0:
		# breakpoint()
			# Get "Model Values"
			idw_ele = gt.llh_idw(idf_G['lon'].values,idf_G['lat'].values,idf_G['ele'].values,\
								 idf_S['Longitude'].values,idf_S['Latitude'].values,\
								 power=2)
			# Get difference in elevations
			idZv = idf_S['Elevation'].values - idw_ele
			# Summarize corrections
			line = [np.mean(idZv),np.median(idZv),np.std(idZv),np.min(idZv),np.max(idZv),len(idZv)]
			corrections.update({it1_:line})
		else:
			idZv = np.nan

		# Apply corrections
		df_c = pd.DataFrame(idf_G['ele'].values + np.mean(idZv),index=idf_G.index,columns=['mean(dZ)'])
		df_ele_corr = pd.concat([df_ele_corr,df_c],axis=0,ignore_index=False)


df_GPSc = df_GPSf.copy()
df_GPSc = pd.concat([df_GPSc,df_ele_corr],axis=1,ignore_index=False)
df_GPSc = df_GPSc[df_GPSc['mean(dZ)'].notna()]

# Write corrected data to file
df_GPSc.to_csv(os.path.join(ODIR,'Prudhoe_Elevation_Corrected_GPS_Tracks.csv'),\
			   header=True,index=True)

# Update SITE data and save copy to file ::TODO::



# NTS: Chose to abandon this last in favor of  using
# IDW methods developed here and elevation-corrected
# GPS data to do IDW elevations for GeoRods
# ## CONDUCT RBF Interpolation

# df_GPScf = df_GPSc[df_GPSc['mean(dZ)'].notna()]
# xq = df_GPScf['UTM19N mE'].values
# yq = df_GPScf['UTM19N mN'].values
# zq = df_GPScf['mean(dZ)'].values


# ZI,XI = gt.interp_3D_Rbf(yq,xq,zq,nx=200j,ny=200j)






# plt.show()


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


