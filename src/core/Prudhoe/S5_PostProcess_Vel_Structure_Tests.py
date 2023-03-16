"""
:module: Postprocess_Vel_Structure_Tests.py
:prupose: Create a summary table of all vertical velocity structure experiments and incorporate 

:TODO:
Bring in Phase picks and create a geometric representation of sampled parts of the bed from 
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


### SUBROUTINES ###
def read_VN_experiment(fname):
	df_mod = pd.read_csv(fname)
	nps = os.path.split(fname)[-1].split('_')
	if nps[0] == 'S4A':
		# Scrape experiment information from experiment file-name
		FM = nps[3] # Firn Model type
		FMQ = nps[2] # Firn Model Quantile perturbation used
		KD = 'all' # Data KinD used
		RES = nps[1] # Grid/Parameter Search RESolution
		DS = nps[6] # Data Slice
		SI = nps[0] # Step Number
		EX = 'Ex0' # Experiment Number
	elif nps[0] == 'S4B':
		# Scrape experiment information from experiment file-name
		FM = nps[5] # Firn Model type
		FMQ = nps[4] # Firn Model Quantile perturbation used
		KD = nps[-1][1] # Data KinD used
		RES = nps[2] # Grid/Parameter Search RESolution
		DS = nps[7] # Data Slice
		SI = nps[0] # Step Number
		EX = nps[1] # Experiment Number

	dmeta = {'FM':FM,'FMQ':FMQ,'KD':KD,'RES':RES,'DS':DS,'SI':SI,'EX':EX,'fname':os.path.split(fname)[-1]}
	return df_mod, dmeta

def find_bestfit(df_mod,dmeta,method='res L2'):
	df_ = df_mod[df_mod[method]==df_mod[method].min()]
	dout = dmeta.copy()
	# TODO: have ZN Min/Max BOOL to say if the optimum is an edge effect or an actual cost minimum
	for k_ in df_.columns:
		dout.update({k_:df_[k_].values[0]})
	return dout


# ROOT DIRECTORY
ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# glob STRINGS
GSTR1 = os.path.join(ROOT,'velocity_models','structure_experiments','S4A*.csv')
GSTR2 = os.path.join(ROOT,'velocity_models','structure_experiments','S4B*K?.csv')


### DATA EXTRACTION ###
flist1 = glob(GSTR1)

table1 = []
for f_ in flist1:
	df_mod, dmeta = read_VN_experiment(f_)
	dout = find_bestfit(df_mod,dmeta,method='res L2')
	line = list(dout.values())
	table1.append(line)

df_SUM1 = pd.DataFrame(table1,columns=dout.keys())

flist2 = glob(GSTR2)
table2 = []
for f_ in flist2:
	df_mod, dmeta = read_VN_experiment(f_)
	dout = find_bestfit(df_mod,dmeta,method='res L2')
	line = list(dout.values())
	table2.append(line)

df_SUM2 = pd.DataFrame(table2,columns=dout.keys())




