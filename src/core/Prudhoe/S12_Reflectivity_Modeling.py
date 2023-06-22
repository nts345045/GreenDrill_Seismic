"""
:module: S9_Reflectivity_Modeling.py
:purpose: Conduct calculation of bed reflectivity from A0 and alpha estimates
		  from the previous step

:: TODO ::
Bring in information about data filtering and check if there is a 
systematic bias introduced by filtering to R(\\Theta) calculations.

"""
import os
import sys
from pandas import DataFrame, read_csv, concat
import matplotlib.pyplot as plt
from numpy import pi
sys.path.append(os.path.join('..','..'))
import util.Reflectivity as ref


ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL_RT.csv')
DA0a = os.path.join(ROOT,'A0_alpha_estimates.csv')

df_picks = read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
df_picks = df_picks[df_picks['itype']=='GeoRod']
df_mods = read_csv(DA0a)

# Preferred method
PA_ = 'RMS Amp'
# Filter by preferred amplitude estimate
df_mods = df_mods[(df_mods['A type']==PA_)&(df_mods['itype']=='GeoRod')]

# Get average estimated attenuation
alpha_u = df_mods['alpha'].median()
alpha_s2 = df_mods['var alpha'].median()



# Iterate over shots

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for I_ in [2]:
	for J_ in [2]:
		df_out = DataFrame()
		for SH_ in df_mods['shot #'].unique():
			# Subset picks to reflections for the designated shot
			idf_p = df_picks[(df_picks['phz']=='S')&\
							 (df_picks['shot #']==SH_)]
			# Get source amplitude - overkill, but call median anyway
			A0_u = df_mods[df_mods['shot #']==SH_]['A0'].values[0]
			A0_s2 = df_mods[df_mods['shot #']==SH_]['var A0'].values[0]
			# Compose signed amplitudes of arriving phases
			A1 = idf_p['PS Relative Polarity'].values * idf_p[PA_].values
			# Calculate reflectivity
			iR = ref.calculate_R_direct(A0_u + I_*A0_s2**0.5,A1,\
										idf_p['dd m'].values,\
										alpha_u - J_*alpha_s2**0.5)
			odf = concat([idf_p,DataFrame({'R(A_{})'.format(PA_):iR},index=idf_p.index)],\
							axis = 1,ignore_index=False)
			df_out = concat([df_out,odf],axis=0,ignore_index=False)



		ax1.plot((pi/2 - df_out['theta rad'].values)*(180./pi),df_out['R(A_{})'.format(PA_)],'.',alpha=0.5)
		ax2.plot(df_out['SRoff m'],df_out['R(A_{})'.format(PA_)],'.',alpha=0.5,\
				 label='$A_0 {0:+}\\sigma$ $\\alpha {1:+}\\sigma$'.format(I_,J_))

ax2.legend()
ax1.set_xlabel('Incidence Angle [$\\Theta$] ($^o$)')
ax2.set_xlabel('Source-Receiver Offset [$r$] (m)')
ax1.set_ylabel('Reflectivity [$\\mathcal{R}(\\Theta)$] ( - )')
ax2.set_ylabel('Reflectivity [$\\mathcal{R}(\\Theta)$] ( - )')
ax1.set_ylim([-1,1])
ax2.set_ylim([-1,1])
plt.show()