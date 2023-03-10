"""
:module: GeometryTools.py
:auth: Nathan T. Stevens
:purpose: Contains subroutines used for handling shot-receiver geometry extraction
geometries

"""
import numpy as np
from pyproj import Proj


def LL2epsg(lon,lat,epsg='epsg:32619'):
	"""
	Convert lat/lon in WGS84 into another coordinate reference system
	Default is converting to UTM19N for Prudhoe Dome & Inglefield Land sites

	:: INPUTS ::
	:param lat: [array-like or float] latitudes (y-coordinates)
	:param lon: [array-like or float] longitudes (x-coordinates)
	:param epsg: [string] definition to feed to pyproj.Proj

	:: OUTPUTS ::
	:return X: [array-like or float] X-coordinates in target EPSG
	:return Y: [array-like or float] Y-coordinates in target EPSG
	"""
	myproj = Proj('epsg:32619')
	X,Y = myproj(lon,lat)
	return X,Y


def cartesian_azimuth(dmE,dmN):
	"""
	Calculate azimuth in cartesian coordinates
	(also return counterclockwise rotation angle)

	:: INPUTS ::
	:param dmE: distance Easting
	:param dmN: distance Northing

	:: OUTPUTS ::
	:param az: azimuth (clockwise angle from positiven North)
	:param theta: angle (counterclockwise from positive East)
	"""
	theta = np.arctan2(dmN,dmE)
	az = np.arctan2(dmE,dmN)
	az = -1.*az + np.pi/2.
	if isinstance(az,np.ndarray):
		az[az<0] += 2.*np.pi
	elif isinstance(az,float) and az < 0:
		az += 2.*np.pi
	return az, theta


def find_nearest_leash(x0,xv,leash=10):
	"""
	Find the index of the nearest value to x0
	in xv that falls within the leash distance
	or return False

	:: INPUTS ::
	:param x0: [float] reference location
	:param xv: [array-like] test locations
	:param leash: [float] maximum acceptable residual magnitude for |xv - x0|

	:: OUTPUT ::
	:param ind: [float or BOOL] closest value in xv if there is a valid xv[i_] 
				S.T. min|xv - x0| <= leash, else return False
	"""
	dxv = xv - x0
	ind = np.argmin(np.abs(dxv))
	x1i = xv[ind]
	if np.abs(dxv[ind]) <= leash:
		return ind
	else:
		return False



# def query_gridded_clusters(X,Y,max_offset=100,grid_step=50):
# 	Xv = np.arange(np.nanmin(X),np.nanmax(X)+grid_step,grid_step)
# 	Yv = np.arange(np.nanmin(Y),np.nanmax(Y)+grid_step,grid_step)
# 	Cn = 0:
# 	for 
		