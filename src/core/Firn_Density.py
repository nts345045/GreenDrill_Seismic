"""
:module: Firn_Density.py
:auth: Nathan T. Stevens
:purpose: Contains methods to estimate vertical density profiles from vertical velocity profiles
 using methods summarized in:
	Diez and others (2013)
 	Riverman and others (2019)
 	Schlegel and others (2019)
"""
import numpy as np

def rho_robin(vp):
	"""
	Implement the emprical relationship from Robin (1958) based on
	laboratory tests and observations in Antarctica and Jungfraujoch
	(see Diez and others, 2013)
	
	rho(z) = 0.221*vp(z) + 59

	:: INPUT ::
	:param vp: [np.ndarray] P-wave velocities in m/sec

	:: OUTPUT ::
	:return rhoz: [np.ndarray] density estimates in kg/m**3
	"""
	rhoz = 0.221*vp + 59.
	return rhoz

def rho_kohnen(vp,vpice=3850.,rhoice=915.):
	"""
	Implement the empirical relationship from Kohnen (1972) for firn
	density from P-wave velocities

	rho(z) = rhoice/(1 + (vpice - vp(z))/2250)**1.22)

	:: INPUTS ::
	:param vp: [np.ndarray] P-wave velocities in m/sec
	:param vpice: [float]  Reference vp for fully compacted firn/glacial ice in m/sec
	:param rhoice: [float] Reference density for fully compacted firn/glacial ice in kg/m**3

	:: OUTPUT ::
	:return rhoz: [np.ndarray] density estimates in kg/m**3
	"""
	den = 1. + ((vpice - vp)/2250)**1.22
	rhoz = rhoice/den
	return rhoz


