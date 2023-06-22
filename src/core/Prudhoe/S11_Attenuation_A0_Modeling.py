"""
:module: S11_Attenuation_A0_Modeling.py
:purpose: Estimate source-strengths and attenuation from GeoRod amplitudes defined by the equation
			ln(Ai/d) = 
:auth: Nathan T. Stevens
:email: nts5045@psu.edu
:TODO:
	- update A0 estimation from multiples with new nomenclature.
"""
import os
import sys
from pandas import DataFrame,read_csv
from tqdm import tqdm
from numpy import nan, std, mean, log, exp, polyfit, poly1d
sys.path.append(os.path.join('..','..'))
import util.Reflectivity as ref


isplot = True
if isplot:
	import matplotlib.pyplot as plt

## MAP DATA ##
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
DPHZ = os.path.join(ROOT,'Corrected_Phase_Picks_v5_ele_MK2_pfO3_sutured_PSR_Amps_RPOL_RT.csv')
OFILE = os.path.join(ROOT,'A0_alpha_estimates.csv') 

# Load picks
df_picks = read_csv(DPHZ,parse_dates=['time']).sort_values('SRoff m')
# Subset to kind=2 (amplitude defining pick)
df_picks = df_picks[df_picks['kind']==2]
# Minimum offset to isolate linear portion of A*d(d)
SRmin = 250
# Minimum number of data for linear regression
min_data = 8
# Coefficient for outlier detection (C_out*std(res))
C_out = 5
WL_ = 1

# Subset diving waves
df_P = df_picks[df_picks['phz']=='P']

# Subset multiples
df_R = df_picks[(df_picks['phz']=='R') &\
				(df_picks['kind'].isin([1,2,3])) &\
				(df_picks['SR Relative Polarity'].notna()) &\
				(df_picks['SRoff m'] < 800)]

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
			# Conduct initial linear fitting
			imod,icov = polyfit(d_D.values,log(A_D.values*d_D.values),1,cov=True)
			# Get residuals
			res = log(A_D.values*d_D.values) - poly1d(imod)(d_D.values)
			# Filter for outliers
			IND = (abs(res) <= C_out*std(res)) & (abs(res) < WL_)
			if isplot:
				if IT_ == 'Node':
					ax1.plot(d_D,log(A_D*d_D),'.-',label=str(SH_))
					ax1.plot(d_D[~IND],log(A_D[~IND]*d_D[~IND]),'ks')
				else:
					ax2.plot(d_D,log(A_D*d_D),'.-',label=str(SH_))
					ax2.plot(d_D[~IND],log(A_D[~IND]*d_D[~IND]),'ks')
			# Conduct second linear fitting with outliers rejected
			imod,icov = polyfit(d_D.values[IND],log(A_D.values[IND]*d_D.values[IND]),1,cov=True)
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
					 (df_picks['chan']==RE_)]\
					.iloc[0,:].T
	# TODO: Need to match
	for AT_ in ['Peak Amp','RMS Amp','Cycle RMS Amp']:
		iA0 = ref.estimate_A0_multiple(S_Si,S_Mi,DF['alpha'].median(),AT_)
		line = [SH_,IT_,'multiple%d'%(S_Mi['kind']),AT_,iA0,nan,nan,nan,nan]
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
			elif M_ == 'multiple1':
				fmt = '^'
			elif M_ == 'multiple2':
				fmt = 'v'
			elif M_ == 'multiple3':
				fmt = '.'
			idf = df[(df['method']==M_)&(df['A type']==A_)&(df['itype']==I_)]
			if A_ == 'semilog':
				MS = 20
			else:
				MS = 5
			axs[ia_].semilogy(idf['shot #'],idf['A0'],fmt,ms=MS,alpha=0.5,label=flbl)
		axs[ia_].legend()
		axs[ia_].set_title('{} {}'.format(A_,I_))
		ia_ += 1

		# ax2.plot(IDF['shot #'].values,IDF['A0'],fmt,label=flbl,alpha=0.5)
# ax1.legend()
# ax2.legend()

DF.to_csv(OFILE,header=True,index=False)

plt.show()

