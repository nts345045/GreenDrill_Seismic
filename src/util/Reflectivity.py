"""
:module: Reflectivity.py
:purpose: Contains methods for estimating source amplitudes ($A_0$) and ice-bed reflectivity (R)
:author: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu


:: TODO ::
Migrate source amplitude estimation calculations from private into public, general use methods
"""
import numpy as np
import os
import sys
sys.path.append('..')
sys.path.append(os.path.join('..','..'))
import util.InvTools as inv

def estimate_A0_diving(S_dir1,S_dir2,ATYPE='RMS Amp'):
	"""
	Estimate the source-amplitude ($A_0$) using the diving wave method
	from Holland and Anandakrishnan (2009) [their equation 9] that uses
	amplitudes A1 at ray-distance = s1 and A2 at ray distance = s2 = 2*s1

	This assumes rays experience idential path-averaged attenuation

	:: INPUTS ::
	:param S_dir1: pandas.Series with measured amplitudes and modeled
					ray-path length for the diving wave with ray-length s1
	:param S_dir2: pandas.Series with measured amplitudes and modeled
					ray-path length for the diving wave with ray-length s2
	:param alpha: attenuation coefficient
	:param ATYPE: amplitude measure to use

	:: OUTPUT ::
	:return A0: Estimate of soruce-amplitude
	"""
	B1 = np.abs(S_dir1[ATYPE].values[0])
	B2 = np.abs(S_dir2[ATYPE].values[0])
	d1 = np.abs(S_dir1['dd m'].values[0])
	d2 = np.abs(S_dir2['dd m'].values[0])
	y1 = d1**-0.5
	y2 = d2**-0.5
	A0 = (B1**2/B2)*(y2/y1**2)
	return A0


def estimate_A0_multiple(S_prime,S_mult,alpha,ATYPE='RMS Amp'):
	"""
	Estimate the source amplitude from a seismic reflection and
	it's first multiple recorded at a single station using equation
	(7) from Holland & Anandakrishnan (2009). Amplitudes are used
	as absolute values for calculation of A0

	:: INPUTS ::
	:param S_prime: pandas.Series with measured amplitudes and modeled
					ray-path length for the primary reflection
	:param S_mult: pandas.Series with measured amplitudes and modeled
					ray-path length for the first multiple
	:param alpha: attenuation coefficient
	:param ATYPE: amplitude measure to use

	:: OUTPUT ::
	:return A0: Estimate of soruce-amplitude
	"""
	A1 = np.abs(S_prime[ATYPE])
	A2 = np.abs(S_mult[ATYPE])
	d1 = S_prime['dd m']
	d2 = S_mult['dd m']
	y1 = d1**-1
	y2 = d2**-1
	A0 = (A1**2/A2)*(y2/y1**2)*np.exp(alpha*(2*d1 - d2))
	return A0



def calculate_R_direct(A0,A1,dd,alpha):
	"""
	Calculate the reflection coefficient using equation 10 in 
	Holland & Anandakrishnan (2009).

	Assumes spherical spreading S.T. \\gamma = 1/dd

	:: INPUTS ::
	:param A0: source amplitude
	:param A1: reflected wave amplitude recorded at receiver
	:param dd: ray path length
	:param alpha: attenuation coefficient

	:: OUTPUT ::
	:return R_theta: reflection coefficient 
	"""
	gamma = dd**-1
	num = A1*np.exp(alpha*dd)
	den = A0*gamma
	R_theta = num/den
	return R_theta



def estimate_A0_alpha_diving(A_D,d_D,A_sig=None,d_sig=None):
	"""
	Calculate the source amplitude and path-averaged attenuation
	using semi-log regression of diving-wave amplitudes and modeled
	ray-paths
	
	:: INPUTS ::
	:param A_D: Diving wave amplitudes [array-like]
	:param d_D: Diving wave ray-paths [array-like]

	:: OUTPUT ::
	:return A0: Source strength
	:return alpha: attenuation coefficient
	:return var_A0: Source strength variance
	:return var_alpha: attenuation coefficient variance
	:return beta: ODR fitting results (see scipy.odr.ODR)
	"""
	# Compose linearized terms
	yobs = np.log(A_D*d_D)
	xobs = d_D
	# Do orthogonal distance regression
	beta = inv.curve_fit_2Derr(inv.lin_fun,xobs,yobs,A_sig,d_sig)
	# Extract values and covariance matrix
	alpha = -1.*beta.beta[0]
	A0 = np.exp(beta.beta[1])
	var_alpha = beta.sd_beta[0]**2
	var_A0 = (A0*beta.sd_beta[1])**2

	return A0,alpha,var_A0,var_alpha,beta

