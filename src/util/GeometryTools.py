"""
:module: GeometryTools.py
:auth: Nathan T. Stevens
:purpose: Contains subroutines used for handling shot-receiver geometry extraction
geometries

:attribution: Inverse Distance Weighting code by Majramos (https://gist.github.com/Majramos/5e8985adc467b80cccb0cc22d140634e)

"""
import numpy as np
from pyproj import Proj
from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator


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



def interp_3D_Rbf(xq,yq,zq,nx=50j,ny=50j,kwargs={}):
	"""
	Conveineice method for running scipy.interpolate.RBFInterpolator()
	Conduct a Radial Basis Function interpolation for a point-cloud
	of data with 2 independent variables.

	:: INPUTS ::
	:param xq: X-coordinate data vector
	:param yq: Y-coordinate data vector
	:param zq: Dependent variable data vector
	:param nx: X-grid discretization [number of equidistant M-nodes]
	:param ny: Y-grid discretization [number of equidistant N-nodes]

	:: OUTPUT ::
	:return YI: Interpolated dependent data [M,N]
	:return XI: Interpolation grid [2,M,N]

	"""
	# Create independent variable array
	xobs = np.array([np.array(yq),np.array(xq)]).T
	# Rename/reformat dependent variable data
	yobs = np.array(zq)
	# Create interpolation coordinate grid
	XI = np.mgrid[np.nanmin(yq):np.nanmax(yq):ny,np.nanmin(xq):np.nanmax(xq):nx]
	# Reshape coordinates for input to RBFInterpolator()
	xflat = XI.reshape(2,-1).T
	# INTERPOLATE #
	yfld = RBFInterpolator(xobs,yobs,**kwargs)(xflat)
	# Reshape output to match grids
	YI = yfld.reshape(XI[0,:,:].shape)

	return YI,XI


def interp_CT2D(xq,yq,zq,nx=50j,ny=50j,kwargs={}):
	"""
	Conveineice method for running scipy.interpolate.CloughTocher2DInterpolator()
	Conduct an interpolation for a point-cloud
	of data with 2 independent variables.

	:: INPUTS ::
	:param xq: X-coordinate data vector
	:param yq: Y-coordinate data vector
	:param zq: Dependent variable data vector
	:param nx: X-grid discretization [number of equidistant M-nodes]
	:param ny: Y-grid discretization [number of equidistant N-nodes]

	:: OUTPUT ::
	:return YI: Interpolated dependent data [M,N]
	:return XI: Interpolation grid [2,M,N]

	"""
	# Create independent variable array
	xobs = np.array([np.array(yq),np.array(xq)]).T
	# Rename/reformat dependent variable data
	yobs = np.array(zq)
	# Create interpolation coordinate grid
	XI = np.mgrid[np.nanmin(yq):np.nanmax(yq):ny,np.nanmin(xq):np.nanmax(xq):nx]
	interp=CloughTocher2DInterpolator(xobs,yobs)
	YI = interp(XI[0],XI[1])
	return YI,XI


### IDW Implementations From https://gist.github.com/Majramos/5e8985adc467b80cccb0cc22d140634e

def distance_matrix(x0, y0, x1, y1):
	""" Make a distance matrix between pairwise observations.
	Note: from <http://stackoverflow.com/questions/1871536> 
	"""
	
	obs = np.vstack((x0, y0)).T
	interp = np.vstack((x1, y1)).T

	d0 = np.subtract.outer(obs[:,0], interp[:,0])
	d1 = np.subtract.outer(obs[:,1], interp[:,1])
	
	# calculate hypotenuse
	return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi, power=1):
	""" 
	Simple inverse distance weighted (IDW) interpolation 
	Weights are proportional to the inverse of the distance, so as the distance
	increases, the weights decrease rapidly.
	The rate at which the weights decrease is dependent on the value of power.
	As power increases, the weights for distant points decrease rapidly.

	# NTS Added the dmax argument to limit admitted data
	"""
	
	dist = distance_matrix(x,y, xi,yi)

	# IND = dist < dmax
	# In IDW, weights are 1 / distance
	weights = 1.0/(dist+1e-12)**power

	# Make weights sum to one
	weights /= weights.sum(axis=0)
	# breakpoint()
	# Multiply the weights for each interpolated point by all observed Z-values
	return np.dot(weights.T, z)



def llh_idw(lons,lats,ele,rlons,rlats,power=1,epsg='epsg:32619'):
	"""
	Wrapper for simple_idw() to accept latitudes and longitudes that will
	be converted into meter-scaled measures using a given epsg

	:: INPUTS ::
	:param lons: longitudes that are being sampled
	:param lats: latitudes that are being sampled
	:param ele: elevations that are being sampled
	:param rlons: longitudes where interpolation occur
	:param rlats: latitudes where interpolation occur
	:param power: weighting power
	:param epsg: string defining EPSG to use to convert from lat/lon to meter-scaled

	:: OUTPUTS ::
	:return zi: interpolated values
	"""
	x,y = LL2epsg(lons,lats,epsg=epsg)
	x0,y0 = LL2epsg(rlons,rlats,epsg=epsg)
	zi = simple_idw(x,y,ele,x0,y0,power=power)
	return zi
