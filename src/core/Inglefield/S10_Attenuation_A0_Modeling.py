"""
:module: S8_Attenuation_A0_Modeling.py
:purpose: Estimate source-strengths and attenuation from GeoRod amplitudes defined by the equation
			ln(Ai/d) = 
:
"""
import os
import sys
from pandas import DataFrame,read_csv
from tqdm import tqdm
from numpy import log, exp, polyfit, nan
sys.path.append(os.path.join('..','..'))
import util.Reflectivity as ref


isplot = True
if isplot:
	import matplotlib.pyplot as plt

## MAP DATA ##
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Inglefield_Land')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_Amps_RPOL_RT.csv')
OFILE = os.path.join(ROOT,'A0_alpha_estimates.csv') 

# Load picks
df_picks = read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Subset to kind=2 (amplitude defining pick)
df_picks = df_picks[df_picks['kind']==2]
# Minimum offset to isolate linear portion of A*d(d)
SRmin = 250
# Minimum number of data for linear regression
min_data = 5

# Subset diving waves
df_P = df_picks[df_picks['phz']=='P']

# Subset multiples
df_R = df_picks[df_picks['phz']=='R']

if isplot:
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

lines = []
# Iterate across shot/phase-type/instrument-type combinations
for SH_,IT_ in df_P[['shot #','itype']].value_counts().sort_index().index:
	idf = df_P[(df_P['shot #']==SH_)&(df_P['itype']==IT_)&(df_P['SRoff m']>=SRmin)]
	if len(idf) > min_data:
		# If considering diving waves, use the A0 / alpha simultaneous estimate approach
		for AT_ in ['Peak Amp','RMS Amp','Cycle RMS Amp']:
			jdf = idf[idf[AT_].notna()]
			A_D = abs(jdf[AT_])
			d_D = jdf['dd m']
			if isplot:
				if IT_ == 'Node':
					ax1.plot(d_D,log(A_D*d_D),'.-',label=str(SH_))
				else:
					ax2.plot(d_D,log(A_D*d_D),'.-',label=str(SH_))

			imod,icov = polyfit(d_D.values,log(A_D.values*d_D.values),1,cov=True)
			iA0 = exp(imod[1])
			ia = -1.*imod[0]
			ivarA0 = icov[1,1]*iA0**2
			ivara = icov[0,0]
			icovalogAd = icov[0,1]
			line = [SH_,IT_,'semilog',AT_,iA0,ia,ivarA0,ivara,icovalogAd]

			lines.append(line)
	else:
		line = [SH_,IT_,'semilog',nan,nan,nan,nan,nan,nan]
		lines.append(line)
# Compose dataframe here to allow use of average alpha estimate in subsequent steps
DF = DataFrame(lines,columns=['shot #','itype','method','A type','A0','alpha','var A0','var alpha','cov log(A0)-alpha'])


#### Conduct estimation of A0 using multiples where able ####

# Iterate across each multiple entry
for i_ in range(len(df_R)):
	S_Mi = df_R.iloc[i_,:].T
	# pull shot and receiver information 
	SH_,RE_,IT_ = S_Mi[['shot #','chan','itype']]
	# Match to a primary reflection pick
	S_Si = df_picks[(df_picks['phz']=='S')&\
					 (df_picks['shot #']==SH_)&\
					 (df_picks['chan']==RE_)].iloc[0,:].T

	for AT_ in ['Peak Amp','RMS Amp','Cycle RMS Amp']:
		iA0 = ref.estimate_A0_multiple(S_Si,S_Mi,DF['alpha'].median(),AT_)
		line = [SH_,IT_,'multiple',AT_,iA0,nan,nan,nan,nan]
		lines.append(line)

DF = DataFrame(lines,columns=['shot #','itype','method','A type','A0','alpha','var A0','var alpha','cov log(A0)-alpha'])


#### PLOTTING COMPARISON OF METHODS ####

fig = plt.figure()
axs = fig.subplots(nrows=3,ncols=2).ravel()

ia_ = 0

df = DF[DF['A type'].notna()]
for A_ in df['A type'].unique():
	for I_ in df['itype'].unique():
		for M_ in df['method'].unique():
			flbl = str(M_)
					# if M_ == 'multiple':
			if M_ == 'semilog':
				fmt = 'd'
			else:
				fmt = '.'
			idf = df[(df['method']==M_)&(df['A type']==A_)&(df['itype']==I_)]
			axs[ia_].plot(idf['shot #'],idf['A0'],fmt,alpha=0.5,label=flbl)
		axs[ia_].legend()
		axs[ia_].set_title('{} {}'.format(A_,I_))
		ia_ += 1

		# ax2.plot(IDF['shot #'].values,IDF['A0'],fmt,label=flbl,alpha=0.5)
# ax1.legend()
# ax2.legend()

DF.to_csv(OFILE,header=True,index=False)

plt.show()

