"""
:module: RayTracing_Modeling.py
:purpose: Conduct ray-tracing for each pick in a given spread using the best-fit velocity
			models from Step 5 (Sensitivity testing) to calculate the geometric spreading
			(gamma) and ray-path length for all phase-pick data.

:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu

:: TODO ::
 - Add in methods for handling multiples
 	- include modeled path for only full multiple? i.e., surface bounce.
 	- Consider firn-bottom bounce?
"""

import sys
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
sys.path.append(os.path.join('..','..'))
import util.Dix_1D_Raytrace_Analysis as d1d

### MAP DATA ###
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
OROOT = os.path.join(ROOT,'velocity_models')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL.csv')
DMOD = os.path.join(ROOT,'velocity_models','structure_summary','Uniform_Firn_Bed_Elevation_Model.csv')
# Wiechert-Herglotz-Bateman Reference Model(s)
DWHB = os.path.join(ROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_WHB_ODR_LHSn100.csv')
# Reference KB79 Model
DKB79 = os.path.join(ROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_KB79_ODR.csv')
# Output File
OUTFILE = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL_RT.csv')


# Load pick data
df_picks = pd.read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Filter down to phase picks of diving (P), primary reflections (S), and multiples (R)
df_picks = df_picks[df_picks['phz'].isin(['P','S','R'])]
# Filter down to geo-rods
df_GR = df_picks[df_picks['itype']=='GeoRod']
df_NO = df_picks[df_picks['itype']=='Node']

# Load velocity model
df_mod = pd.read_csv(DMOD)
# Filter down to VFINE (shot-specific) models
df_mod = df_mod[df_mod['Grid Resolution']=='VFINE']
# Load WHB model
df_whb = pd.read_csv(DWHB)
df_kb = pd.read_csv(DKB79,index_col=[0])
# Initialize Cake velocity model 
CakeMod_WHB = d1d.generate_layercake_slow(Uv = df_whb['mean u(z)'].values*1e-3,\
										  Zv = [0]+list(df_whb['median z'].values))

# Create travel-time/ray-path lengths for uniform firn model
ddp,ttp = d1d.diving_raypath_oval_est(df_whb['mean X'].values,df_whb['median z'].values,df_kb['mean'].values)
# Convert modeled diving wave travel-times into seconds
ttp /= 1e3


# Create holder for updated picks
df_picks_out = pd.DataFrame()
print('======= RUNNING GEORODS WITH SHOT-SPECIFIC DEPTH MODELS =======')

### CONDUCT PATH MODELING FOR GEORODS WITH SHOT-SPECIFIC MODELS
# Iterate across spread/shot combinations
for SP_ ,SH_ in df_GR[['spread','shot #']].value_counts().sort_index().index:
	print('------- processing {} {} -------'.format(SP_,SH_))
	### PROCESS DIVING WAVES
	idf_P = df_GR[(df_GR['spread']==SP_)&(df_GR['shot #']==SH_)&(df_GR['phz']=='P')]
	# Interpolate diving wave travel-times to get path lengths
	if len(idf_P) > 0:
		print('Interpolating for Diving Waves on GeoRods')
		# Do interpolation
		ddp_hat = np.interp(idf_P['tt sec'].values,ttp,ddp)	
		# Append result to subset pick data
		idf_P = pd.concat([idf_P,pd.DataFrame({'dd m':ddp_hat},index=idf_P.index)],\
						   axis=1,ignore_index=False)
		# append results to output
		df_picks_out = pd.concat([df_picks_out,idf_P],axis=0,ignore_index=False)


	### PROCESS REFLECTED PHASES (PRIMARY & MULTIPLE)
	# Subset depth model
	idf_m = df_mod[(df_mod['Data Slice']==SH_)]
	
	## Compose ray-tracing velocity model
	Hsurf_bar = idf_m['mH mean'].values[0] # Get average surface elevation
	# Get best-fit total ice-thickness
	Zice = idf_m['Ice thickness (m)'].values[0]
	# Get best-fit deep ice velocity
	Vice = idf_m['VN m/s'].values[0]
	# Append deep ice layer to firn velocity model
	CakeMod = d1d.add_halfspace(CakeMod_WHB,Vice,Hn=Zice+50)

	# Subset Picks for Primary Reflections
	idf_S = df_GR[(df_GR['spread']==SP_)&(df_GR['shot #']==SH_)&(df_GR['phz']=='S')]
	lines = []
	if len(idf_S) > 0:	
		print('Running Ray Tracing on Primary Reflections for GeoRods')
		# Iterate across each source-receiver combination
		for i_ in tqdm(range(len(idf_S))):
			S_pick = idf_S.iloc[i_,:].T
			# Calculate static-corrected reflection depth
			Z_src_stat = Zice - (Hsurf_bar - S_pick['CMP mH'])
			itts,idds,iOs = d1d.raytrace_single(CakeMod,S_pick['SRoff m'],Z_src_stat)
			line = [idds[0],iOs[0]]
			lines.append(line)
		# Append results to phase pick data
		idf_S = pd.concat([idf_S,pd.DataFrame(lines,columns=['dd m','theta rad'],index=idf_S.index)],\
						  axis=1,ignore_index=False)
		# Append results to output
		df_picks_out = pd.concat([df_picks_out,idf_S],axis=0,ignore_index=False)

	# Subset Picks for Multiple Reflections
	idf_R = df_GR[(df_GR['spread']==SP_)&(df_GR['shot #']==SH_)&(df_GR['phz']=='R')]
	lines = []
	if len(idf_R) > 0:
		print('Running Ray Tracing on Multiple Reflections for GeoRods')
		# Iterate across each source-receiver combination
		for i_ in tqdm(range(len(idf_R))):
			S_pick = idf_R.iloc[i_,:].T
			# Calculate static-corrected reflection depth
			Z_src_stat = Zice - (Hsurf_bar - S_pick['CMP mH'])
			# Calculate ray-path length using half source-receiver offset 
			itts,idds,iOs = d1d.raytrace_single(CakeMod,S_pick['SRoff m']/2,Z_src_stat)
			# Double path-length to estimate multiple. Incidence angle is correct as-is
			line = [2*idds[0],iOs[0]]
			# append result to holder list
			lines.append(line)
		# Append results to phase pick data
		idf_R = pd.concat([idf_R,pd.DataFrame(lines,columns=['dd m','theta rad'],index=idf_R.index)],\
						  axis=1,ignore_index=False)
		# Append results to output
		df_picks_out = pd.concat([df_picks_out,idf_R],axis=0,ignore_index=False)


print('======= RUNNING NODES WITH SITE AVERAGE DEPTH MODEL =======')
### CONDUCT PATH MODELING FOR NODES WITH AREA-AVERAGED MODEL

## Compose ray-tracing velocity model
Hsurf_bar = df_mod['mH mean'].mean() # Get average surface elevation
# Get best-fit total ice-thickness
Zice = df_mod['Ice thickness (m)'].mean()
# Get best-fit deep ice velocity
Vice = df_mod['VN m/s'].mean()
# Append deep ice layer to firn velocity model
CakeMod = d1d.add_halfspace(CakeMod_WHB,Vice,Hn=Zice+50)


# Iterate across spread/shot combinations
for SP_ ,SH_ in df_NO[['spread','shot #']].value_counts().sort_index().index:
	print('------- processing {} {} -------'.format(SP_,SH_))
	### PROCESS DIVING WAVES
	idf_P = df_NO[(df_NO['spread']==SP_)&(df_NO['shot #']==SH_)&(df_NO['phz']=='P')]
	# Interpolate diving wave travel-times to get path lengths
	if len(idf_P) > 0:
		print('Interpolating for Diving Waves on Nodes')
		# Do interpolation
		ddp_hat = np.interp(idf_P['tt sec'].values,ttp,ddp)	
		# Append result to subset pick data
		idf_P = pd.concat([idf_P,pd.DataFrame({'dd m':ddp_hat},index=idf_P.index)],\
						   axis=1,ignore_index=False)
		# append results to output
		df_picks_out = pd.concat([df_picks_out,idf_P],axis=0,ignore_index=False)


	### PROCESS REFLECTED PHASES (PRIMARY & MULTIPLE)


	# Subset Picks for Primary Reflections
	idf_S = df_NO[(df_NO['spread']==SP_)&(df_NO['shot #']==SH_)&(df_NO['phz']=='S')]
	lines = []
	if len(idf_S) > 0:	
		print('Running Ray Tracing on Primary Reflections for Nodes')
		# Iterate across each source-receiver combination
		for i_ in tqdm(range(len(idf_S))):
			S_pick = idf_S.iloc[i_,:].T
			# Calculate static-corrected reflection depth
			Z_src_stat = Zice - (Hsurf_bar - S_pick['CMP mH'])
			itts,idds,iOs = d1d.raytrace_single(CakeMod,S_pick['SRoff m'],Z_src_stat)
			line = [idds[0],iOs[0]]
			lines.append(line)
		# Append results to phase pick data
		idf_S = pd.concat([idf_S,pd.DataFrame(lines,columns=['dd m','theta rad'],index=idf_S.index)],\
						  axis=1,ignore_index=False)
		# Append results to output
		df_picks_out = pd.concat([df_picks_out,idf_S],axis=0,ignore_index=False)

	# Subset Picks for Multiple Reflections
	idf_R = df_NO[(df_NO['spread']==SP_)&(df_NO['shot #']==SH_)&(df_NO['phz']=='R')]
	lines = []
	if len(idf_R) > 0:
		print('Running Ray Tracing on Multiple Reflections for Nodes')
		# Iterate across each source-receiver combination
		for i_ in tqdm(range(len(idf_R))):
			S_pick = idf_R.iloc[i_,:].T
			# Calculate static-corrected reflection depth
			Z_src_stat = Zice - (Hsurf_bar - S_pick['CMP mH'])
			# Calculate ray-path length using half source-receiver offset 
			itts,idds,iOs = d1d.raytrace_single(CakeMod,S_pick['SRoff m']/2,Z_src_stat)
			# Double path-length to estimate multiple. Incidence angle is correct as-is
			line = [2*idds[0],iOs[0]]
			# append result to holder list
			lines.append(line)
		# Append results to phase pick data
		idf_R = pd.concat([idf_R,pd.DataFrame(lines,columns=['dd m','theta rad'],index=idf_R.index)],\
						  axis=1,ignore_index=False)
		# Append results to output
		df_picks_out = pd.concat([df_picks_out,idf_R],axis=0,ignore_index=False)




df_picks_out.to_csv(OUTFILE,header=True,index=False)































# 	## Conduct ray-tracing for
# 	rr_hat = np.arange(0,1500,50)
# 	# Model reflection travel times, path lengths, and incidence angles
# 	tts_hat,dds_hat,Os_hat = d1d.raytrace_summary(CakeMod,rr_hat,Zice)


# 	# Use ray-tracing to model
# 	if len(idf_S) > 0:

# 		# Interpolate for relevant data








# # Create dictionary for velocity model files
# mod_dict = {'Full':{'WHB':'Full_v8_WHB_ODR_LHSn100.csv',\
# 					'RT':'Ice_Thickness_1DRT_v8_K2_FINE_WIDE_Full_mean_GridSearch.csv'}}
# for sp_ in ['NS01','NS02','NS03','WE01','WE02','WE03']:
# 	mod_dict.update({sp_:{'WHB':'Spread_{SP}_v8_GeoRod_WHB_ODR_LHSn100.csv'.format(SP=sp_),\
# 						  'RT':'Ice_Thickness_1DRT_v8_D2_FINE_{SP}_mean_DepthSweep.csv'.format(SP=sp_)}})


# # Iterate across unique combinations
# df_p_update = pd.DataFrame()


# ## Process Node Raytracing ##
# idf_picks = df_picks[df_picks['itype']=='Node']
# # Load shallow velocity model
# df_WHB = pd.read_csv(os.path.join(OROOT,mod_dict['Full']['WHB']))
# df_RH = 


# for SP_ in df_picks['spread'].unique():
# 	df_WHB = pd.read_csv(mod_dict)


# for PH_,IT_,SP_ in df_picks[['phz','itype','spread']].unique():
# 	# Subset data
# 	idf_picks = df_picks[(df_picks['phz']==PH_)&\
# 						(df_picks['itype']==IT_)&\
# 						(df_picks['spread']==SP_)]
# 	# If pick is on a node, use the 'Full' ensemble models
# 	if IT_ == 'Node':
# 		P_mod = 
# 		SR_mod = 

# 	# If pick is on a GeoRod, use the spread-specific models
# 	elif IT_ == 'GeoRod':



	


# 	# ...if diving wave...
# 	if PH_ == 'P':
# 		# ...picked on a node...
# 		if IT_ == 'Node':
# 			# ...use ensemble WHB model
# 		else:
# 			# ...use spread specific WHB model
# 	# ...if primary reflection...	
# 	elif PH_ == 'S':
# 		# ...picked on a node...
# 		if IT_ == 'Node':
# 			#...use ensemble mean ice-thickness model
# 		else:
# 			#...use spread-specific ice-thickness model
# 	#...if reflection multiple...
# 	elif PH_ == 'R':
# 		if IT_ == 'Node':
# 			#...









