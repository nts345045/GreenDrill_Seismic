"""
:module: Dix_Conversion.py
:purpose: Contains methods supporting Dix Conversion for estimating
interval velocities and ray-tracing methods for forward modeling
travel times in 1-D layered media

:: TODO ::


"""
import numpy as np
import pandas as pd
import sys
import os
from pyrocko import cake
from copy import deepcopy
from scipy.optimize import curve_fit
from tqdm import tqdm
sys.path.append(os.path.join('..','..'))
import util.InvTools as inv


### OVAL / ELLIPSE RAYPATH APPROXIMATION METHODS ###

def diving_raypath_oval_est(X,Z,aKB):
	"""
	Approximate the path-length of a diving wave as half an oval using the Ramanujan
	approximation with turning depth and maximum offsets from WHB inversions as the 
	semi-minor and semi-major axes of the oval.

	Calculate travel-times using the Kirchner & Bentley (1979) model used as an input
	for WHB analyses

	:: INPUTS ::
	:param X: float or array-like] Maximum source-receiver offset for a given WHB integration step
	:param Z: [float or array-like] WHB inversion estimate of maximum turning depth
	:param aKB: coefficients a1 -- a5 from a fitting of the KB79 model associated with the WHB analysis
				time units dictate units of tt (generally milliseconds)
	:: OUTPUT ::
	:return dd: ray path-length 
	:return tt: ray travel-time (scaled as aKB, generally milliseconds)

	"""
	# Define semi-major axis length
	aa = X/2.
	# Define semi-minor axis length
	bb = Z
	# Calculate H parameter in Ramanujan approximation
	hh = (aa - bb)**2 / (aa + bb)**2
	# Calculate whole perimeter approximatino
	pp = np.pi*(aa + bb)*(1. + (3.*hh)/(10 + np.sqrt(4. - 3.*hh)))
	# Calculate the ray-path length as the half perimeter
	dd = pp/2
	# Calculate travel time with the Kirchner & Bentley (1979) model
	tt = aKB[0]*(1. - np.exp(-aKB[1]*X)) + aKB[2]*(1. - np.exp(-aKB[3]*X)) + aKB[4]*X
	return dd,tt


### RMS VELOCITY METHODS ###

def hyperbolic_tt(xx,Htot,Vrms):
	"""
	Model the hyperbolic ttvo (t(x)) of a flat reflector
	as a function of depth to layer and RMS velocity
	:: INPUTS ::
	:param xx: source-receiver offsets
	:param Htot: total thickness to reflector
	:param Vrms: RMS velocity of medium above reflector

	:: OUTPUT ::
	:return tt: travel time estimates
	"""
	dd = np.sqrt(xx**2 + 4*Htot**2)
	tt = dd/Vrms
	return tt

def hyperbolic_tt_ODR(beta,xx):
	dd = np.sqrt(xx**2 + 4*beta[0]**2)
	tt = dd/beta[1]
	return tt

def calc_Vrms_2L(V1,V2,H1,H2):
	t1 = H1/V1
	t2 = H2/V2
	num = V2**2 * t2 + V1**2 * t1
	den = t2 + t1
	Vrms = (num/den)**0.5
	return Vrms

def calc_Vrms_cont(zz,VV):
	"""
	Calculate the RMS velocity of an n-layered
	vertical velocity profile assuming a linear
	velocity gradient between specified velocity points
	and that the last velocity point is the lower bound
	of the last layer, while all others represent top depths
	of velocity layers

	:: INPUTS ::
	:param zz: depth points
	:param VV: Velocity values at specified points

	:: OUTPUT ::
	:return Vrms: RMS velocity of the layered medium 
	"""
	# Get layer thicknesses
	dz = zz[1:] - zz[:-1]
	# Calculate mean velocity in each layer
	Vbar = 0.5*(VV[1:] + VV[:-1])
	# Calculate travel-time through the layer
	dt = 2*dz/Vbar
	# Calculate the RMS velocity
	num = np.sum(dt*Vbar**2)
	den = np.sum(dt)
	Vrms = (num/den)**0.5
	return Vrms


def dix_VN(Vrms,Hrms,Vi,Zi):
	"""
	Calculate the velocity of the Nth layer of an N-layered layer-cake velocity model
	using estimates of the RMS velocity of the entire stack and the overlying velocity
	structure (Dix equation)

	:: INPUTS ::
	:param Vrms: [float] RMS velocity for the reflector at the bottom of layer N
	:param Hrms: [float] RMS estimate of total column thickness from hyperbolic fitting to a reflector at the base of layer N
	:param Vi: [array-like] shallow velocity structure velocities assumed to be interval-averaged values
	:param Zi: [array-like] shallow velocity structure depths assumed to be bottom-interval depths

	:: OUTPUT ::
	:return V_N: [float] interval velocity for the Nth layer


	"""
	# Estimate Nth layer thickness
	H_N_hat = Hrms - Zi[-1]
	# Calculate shallow Vrms
	Vrms1 = calc_Vrms_cont(Zi,Vi)
	# Calculate shallow travel-time
	t1 = 2*Zi[-1]/Vrms1
	# Get calculated total travel-time for near-vertical incidence
	t2 = 2*Hrms/Vrms
	# Calculate lower interval velocity
	V_N = np.sqrt(((t2*Vrms**2) - (t1*Vrms1**2))/(t2 - t1))
	return V_N


def dix_HN(Vrms,Hrms,Vi,Zi):
	"""
	Calculate the thickness of the Nth layer of an N-layered velocity model
	using estimates of th RMS velocity to a reflector and a defined shallow
	(layers 1 to N-1) velocity structure

	Manipulation of the Dix Equation that treats H_N = (t2 - t1)*V_N
	where V_N is the lower-most velocity in Vi (i.e., Vi[Zi == max(Zi)])
	t1 is calculated as the zero-offset two-way travel time using the 
	shallow velocity structure to define a Vrms1

	:: INPUTS ::
	:param Vrms: [float] RMS velocity for the reflector at the bottom of layer N
	:param Hrms: [float] RMS estimate of total column thickness from hyperbolic fitting to a reflector at the base of layer N
	:param Vi: [array-like] shallow velocity structure velocities assumed to be interval-averaged values
	:param Zi: [array-like] shallow velocity structure depths assumed to be bottom-interval depths

	:: OUTPUT ::
	:return H_N: [float] interval thickness for the Nth layer
	"""
	# Get lowermost velocity value for assumed value at depth
	V_N = Vi[Zi == np.nanmax(Zi)]
	if len(V_N) > 1:
		V_N = V_N[0]
	# Get Vrms for shallow structure
	Vrms1 = calc_Vrms_cont(Zi,Vi)
	# breakpoint()
	# Calculate zero-offset twtt...
	# ...for the shallow layer...
	t1 = np.nanmax(Zi)/Vrms1
	# ...and for the reflector.
	t2 = Hrms/Vrms
	# Main calculation
	H_N = V_N*(t2 - t1)
	return H_N

### WHB PROFILE RESAMPLING

def resample_WHB(Z,uD,method='incremental decrease',scalar=1.3):
	"""
	Resample Wiechert-Herglotz-Bateman vertical slowness profile in 
	a systematic way

	:: INPUTS ::
	:param Z: depth points in meters
	:param uD: slowness points in msec/m
	:param method: how to conduct resampling
					'incremental increase' - bin by a maximum change in slowness value
	:param scalar: scalar value to use with specified method

	:: OUTPUTS ::
	:return Zt: interval top-depth points for new model in meters
	:return Zb: interval bottom-depth points for new model in meters
	:return uRMS: RMS interval slowness values for new model in msec/m

	"""

	if method.lower() == 'incremental decrease':
		# Get surface coordinates
		u0 = uD[0]; z0 = 0.
		# Get interval where ui <= u0*scalar
		I1 = uD <= u0/scalar; z1 = Z[I1][-1]; u1 = uD[I1][-1]
		# Create holders for resampled slowness profile
		uRMS = []; Zt = []; Zb = [];
		# Calculate first interval
		ui = calc_Vrms_cont(Z[I1],uD[I1]**-1)**-1
		Zt.append(z0); Zb.append(z1)
		# While there is still a usable profile
		while z1 < Z[-1]:
			# Advance bounds and calculate new index
			z0 = z1; u0 = u1
			I1 = u0 < I1 <= u0*scalar
			z1 = Z[I1]
			# Calculate RMS slowness within new layer
			ui = calc_Vrms_cont(Z[I1],uD[I1]**-1)**-1
			# Save entries
			uRMS.append(ui)
			Zt.append(z0); Zb.append(z1)

	# elif method.lower() == 'uniform'
	# 	uRMS = [uD[0]]; Zt = [0]; Zb = [scalar]

	# mod_out = {'Ztop':z_top_int,'Zbot':Zb,'uRMS':uRMS}
	return Zt,Zb,uRMS


### 1D RAY TRACING ###
def generate_layercake_slow(Uv=[1/2500,1/3850],Zv=[0,10,4000]):
	"""
	Generate a 1-D layercake seismic velocity model with homogeneous interval 
	slownesses bounded by Zv horizons. For use with pyrocko.cake

	Note that Zv must have 1 additional entry compared to Uv

	:: INPUTS ::
	:param Uv: array-like set of slownesses in sec/m
	:param Zv: array-like set of horizon depths with z positive down

	:: OUTPUT ::
	:return model: pyrocko.cake.LayeredModel with specified layering
	"""

	model = cake.LayeredModel()
	for i_,S_ in enumerate(Uv):
		mi = cake.Material(vp=S_**-1)
		lay = cake.HomogeneousLayer(Zv[i_],Zv[i_+1],mi)
		model.append(lay)
		# if i_ < len(Uv) - 1:
		# 	mj = cake.Material(vp=Uv[i_+1]**-1)
		# 	ifce = cake.Interface(Zv[i_+1],mi,mj,)
		# elif i_ == len(Uv) - 1:
		# 	mj = cake.Material(vp=1.01*Uv[-1]**-1)
		# 	ifce = cake.Interface(Zv[-1],mi,mj)
	return model

def generate_layercake_vel(Vv=[2500,3850],Zv=[0,40,4000]):
	"""
	Wrapper for generate_layercake_slow that accepts interval velocities
	instead of slownesses

	:: INPUTS ::
	:param Vv: array-like set of velocities in m/sec
	:param Zz: array-like set of horizon depths with z positive down

	:: OUTPUT ::
	:return model: pyrocko.cake.LayeredModel with specified layering
	"""
	model = generate_layercake_slow(Sv=np.array(Vv)**-1,Zv=Zv)

	return model

def WHB2CakeMod(Uv,Zv,method='incremental increase',scalar=1.3):
	"""
	Convenience method for downsampling WHB outputs and converting into a 
	pyrocko.Cake.LayeredModel

	:: INPUTS ::
	:param Uv: modeled slownesses [msec/m]
	:param Zv: modeled depths [m BGS]
	:param method: resampling method, see resample_WHB()
	:param scalar: resampling scalar, see resample_WHB()

	:: OUTPUT ::
	:return model: pyrocko.Cake.LayeredModel 
	"""
	mod_out = resample_WHB(Zv,Uv,method=method,scalar=scalar)
	model = generate_layercake_slow(Sv=mod_out['uRMS']*1e-3,Zv=[0] + list(mod_out['Zbot']))

	return model

def add_halfspace(CakeMod,Vn,Hn=4000,verb=False):
	"""
	Add a very thick bottom layer to a copy of an exsiting pyrocko.Cake.LayeredModel
	"""
	NewMod = deepcopy(CakeMod)
	mi = cake.Material(vp=Vn)
	for i_ in CakeMod.elements():
		if verb:
			print('Old model')
			print(i_)
		else:
			pass
	zn = i_.zbot
	lay = cake.HomogeneousLayer(zn,zn+Hn,mi)
	NewMod.append(lay)
	if verb:
		print('New model')
		print(NewMod)
	return NewMod


def raytrace_single(CakeMod,rr,Zsrc,Phase=cake.PhaseDef('p'),pps=10):
	"""
	Run ray-tracing through a layered velocity model for specified receiver 
	locations, source depth, and phase type. This generally assumes that
	the source is at a reflector and the ray-paths are the up-going paths of
	rays in a common-midpoint (CMP) gather. 

	Double path-length and travel-time values to get the full ray-path length and twtt.

	:: INPUTS ::
	:param CakeMod: cake.Model layercake velocity model object
	:param rr: source-receiver offsets in meters
	:param Zsrc: source depth (z positive down) in meters
	:param Phase: cake.PhaseDef to use - default to cake.PhaseDef('p') to have an up-going P-wave
	:param pps: points per segment to estimate within layers

	:: OUTPUTS ::
	:return tt: arrival time at each receiver
	:return dd: travel path length from source to each receiver
	:return thetai: takeoff angle (incidence angle of reflection)

	"""
	# Have near-0 values allowed, but not 0-valued
	distances = [rr*cake.m2d]
	tt = []; dd = []; thetai = [];
	for arr_ in CakeMod.arrivals(distances,phases=Phase,zstart=Zsrc):
		z_,x_,t_ = arr_.zxt_path_subdivided(points_per_straight=pps)
		z_ = z_[0]
		x_ = x_[0]*cake.d2m
		t_ = t_[0]
		thetai.append(np.arctan((z_[0] - z_[2])/(x_[2] - x_[0])))
		tt.append(t_[-1])
		dd.append(np.sum(np.sqrt((z_[:-1] - z_[1:])**2 +\
								 (x_[:-1] - x_[1:])**2))*cake.d2m)
	tt = np.array(tt)
	dd = np.array(dd)/cake.d2m
	return tt,dd,thetai

def raytrace_explicit(CakeMod,rr,Zsrc,Phase=cake.PhaseDef('p'),pps=10):
	"""
	Run ray-tracing through a layered velocity model for specified receiver 
	locations, source depth, and phase type. Default PhaseDef assumes that
	the source is at a reflector and the ray-paths are the up-going paths of
	rays in a common-midpoint (CMP) gather. 

	:: INPUTS ::
	:param CakeMod: cake.Model layercake velocity model object
	:param rr: source-receiver offsets in meters
	:param Zsrc: source depth (z positive down) in meters
	:param Phase: cake.PhaseDef to use - default to cake.PhaseDef('p') to have an up-going P-wave
	:param pps: points per segment to estimate within layers

	:: OUTPUTS ::
	:return dict_out: dictionary with receiver offsets as keys and values of sub-dictionaries with
					keys: 'z m','x m','t s' and values corresponding to modeled rays
	"""

	dict_out = {}
	distances = rr*cake.m2d
	for arr_ in CakeMod.arrivals(distances,phases=Phase,zstart=Zsrc):
		z_,x_,t_ = arr_.zxt_path_subdivided(points_per_straight=pps)
		z_ = z_[0]; x_ = x_[0]*cake.d2m; t_ = t_[0]
		idict = {'z m':z_,'x m':x_,'t s':t_}
		dict_out.update({np.round(x_[-1],decimals=2):idict})
	return dict_out


def raytrace_summary(CakeMod,rr,Zsrc,Phase=cake.PhaseDef('p'),pps=10):
	"""
	Run ray-tracing through a layered velocity model for specified receiver 
	locations, source depth, and phase type. This generally assumes that
	the source is at a reflector and the ray-paths are the up-going paths of
	rays in a common-midpoint (CMP) gather. 

	Double path-length and travel-time values to get the full ray-path length and twtt.

	:: INPUTS ::
	:param CakeMod: cake.Model layercake velocity model object
	:param rr: source-receiver offsets in meters
	:param Zsrc: source depth (z positive down) in meters
	:param Phase: cake.PhaseDef to use - default to cake.PhaseDef('p') to have an up-going P-wave
	:param pps: points per segment to estimate within layers

	:: OUTPUTS ::
	:return tt: arrival time at each receiver
	:return dd: travel path length from source to each receiver
	:return thetai: takeoff angle (incidence angle of reflection)

	"""
	# Have near-0 values allowed, but not 0-valued
	rr[rr==0] += 0.001
	distances = rr*cake.m2d
	tt = []; dd = []; thetai = [];
	for arr_ in CakeMod.arrivals(distances,phases=Phase,zstart=Zsrc):
		z_,x_,t_ = arr_.zxt_path_subdivided(points_per_straight=pps)
		z_ = z_[0]
		x_ = x_[0]*cake.d2m
		t_ = t_[0]
		thetai.append(np.arctan((z_[0] - z_[2])/(x_[2] - x_[0])))
		tt.append(t_[-1])
		dd.append(np.sum(np.sqrt((z_[:-1] - z_[1:])**2 +\
								 (x_[:-1] - x_[1:])**2))*cake.d2m)
	tt = np.array(tt)
	dd = np.array(dd)/cake.d2m
	return tt,dd,thetai


def raytracing_NMO(CakeMod,xx,tt,Zref,dx=10,n_ref=1):
	"""
	Calculate data-model residuals for an up-going ray in a 1-D layered 
	velocity structure. Uses interpolation from regularly spaced modeled
	station location to estimate travel times at each provided station location,
	speeding up solving of Eikonal.

	:: INPUTS ::
	:param CakeMod: pyrocko.Cake.LayeredModel with maximum depth > Zsrc
	:param xx: station locations
	:param tt: arrival times
	:param Zref: guess depth to reflector
	:param dx: model receiver spacing
	:param n_ref: number of reflections, e.g.,
				0 = Just up-going ray (1 leg)
				1 = Primary reflection (single-bounce, 2 legs)
				2 = First multiple (double-bounce, 4 legs)
				3 = Second multiple (triple-bounce, 6 legs)
	:: OUTPUTS ::
	:return tt_cal: Calculated travel-travel times
	:return dd_cal: Calculated ray-path lengths
	:return theta_cal: Calculated incidence angle at the flat reflector

	:: REFLECTION SCALING RESULTS IN AN ERRANT 

	"""
	# Create model station locations
	xx_hat = np.arange(np.nanmin(xx),np.nanmax(xx) + dx, dx)
	# Conduct ray-tracing for upgoing ray
	tt_hat,dd_hat,theta_hat = raytrace_summary(CakeMod,xx_hat,Zsrc=Zref)
	# Multiply path-lengths and travel-times for specified reflection count
	tt_hat *= 2.*n_ref
	dd_hat *= 2.*n_ref
	# Conduct interpolation for datapoints
	tt_cal = np.interp(xx,xx_hat,tt_hat)
	dd_cal = np.interp(xx,xx_hat,dd_hat)
	if n_ref > 0:
		theta_cal = np.interp(xx/n_ref,xx_hat,theta_hat)
	else:
		theta_cal = np.interp(xx,xx_hat,theta_hat)

	return tt_cal,dd_cal,theta_cal


### GRID SEARCH FITTING FUNCTIONS ###

def hyperbolic_fitting(xx,tt,Zv,Uwhb,Zwhb,Vmax=3850,dV=1):
	"""
	Conduct a grid-search hyperbolic fitting using the guessed parameters
	of the thickness of the Nth layer and its velocity using the V_RMS for
	near-normal incidence assumption in Dix conversion analysis.

	:: INPUTS ::
	:param xx: reflected arrival source-receiver offsets in meters
	:param tt: reflected arrival two-way travel times in seconds
	:param Zv: Vector of ice-thickness to guess (firn + glacier ice thickness)
	:param Uwhb: Slowness values from WHB analysis for shallow velocity structure [in msec/m]
	:param Zwhb: Depth values from WHB analysis for shallow velocity structure [in m]
	:param Vmax: Maximum interval velocity for glacier ice [in m/sec]
	:param dv: Increment to discretize Velocity grid-search for bottom value from Uwhb to Vmax

	:: OUTPUT ::
	:return df_out: pandas.DataFrame with summary of models and data-model residuals
			Columns:
				Z m: depth points
				H1 m: thickness of overburden/firn layer
				H2 m: guessed thickness of remaining ice column
				V1rms: RMS velocity of overburden/firn layer
				V2: guessed ice layer velocity
				Vrms: Overall column RMS velocity
				res L2: L-2 norm of data-model residuals for travel-times
				res mean: Average data-model residual value
				res std: standard deviation of data-model residuals
				ndata: number of datapoints

	"""
	# Get bottom depth & velocity of shallow WHB profile
	Vsb = 1e3/Uwhb[-1]
	Vv = np.arange(Vsb,Vmax + dV,dV)
	Hsb = Zwhb[-1]
	# Get RMS Velocity from shallow profile
	Vrms = calc_Vrms_cont(Zwhb,1e3/Uwhb)
	# Iterate across guess-depth values
	MODS = []
	for Z_ in Zv:
		# Iterate across guess-velocity values
		for V_ in Vv:
			# Calculate Vrms for the total firn/ice column
			Vrmsi = calc_Vrms_2L(Vrms,V_,Hsb,Z_ - Hsb)
			# Calculate hyperbolic travel-times for guessed bottom depth
			tt_hat = hyperbolic_tt(xx,Z_,Vrmsi)
			# Calculate residuals
			res = tt - tt_hat
			# RES.append(res)
			# Get stats on residuals
			resL2 = np.linalg.norm(res)
			res_u = np.mean(res)
			res_o = np.std(res)
			# Summarize estimate
			line = [Z_,Hsb,Z_ - Hsb,Vrms,V_,Vrmsi,resL2,res_u,res_o,len(xx)]
			MODS.append(line)
	df_out = pd.DataFrame(MODS,columns=['Z m','H1 m','H2 m','V1rms','V2','Vrms',\
						  'res L2','res mean','res std','ndata'])
	return df_out


# def raytracing_gridsearch(xx,tt,Zv,Uwhb,Zwhb,Vmax=3850,dv=10,dx=10,n_ref=1):
# 	"""
# 	Conduct a grid-search fitting for a flat reflector in a layered medium
# 	where the thickness and velocity of the lowermost layer are unknowns and
# 	conduct a down-sampling of the input shallow velocity structure

# 	:: INPUTS ::
# 	:param xx: receiver locations
# 	:param tt: reflection two-way travel times
# 	:param Zv: vector of reflector depths to check
# 	:param Uwhb: slowness profile from WHB analysis [in msec/m]
# 	:param Zwhb: depth profile from WHB analysis [in m BGS]
# 	:param Vmax: maximum velocity to consider [in m/sec]
# 	:param dv: velocity increment for grid-search [in m/sec]
# 	:param dx: Model station spacing for raytrace_NMO()
# 	:param n_ref: Number of reflections for raytrace_NMO()

# 	:: OUTPUT ::
# 	:return df_out: pandas.DataFrame with summary of models and data-model residuals
# 			Columns:
# 				Z m: depth points
# 				H1 m: thickness of overburden/firn layer
# 				H2 m: guessed thickness of remaining ice column
# 				V1rms: RMS velocity of overburden/firn layer
# 				V2: guessed ice layer velocity
# 				Vrms: Overall column RMS velocity
# 				res L2: L-2 norm of data-model residuals for travel-times
# 				res mean: Average data-model residual value
# 				res std: standard deviation of data-model residuals
# 				ndata: number of datapoints

# 	"""
# 	# Zt,Zb,uRMS = resample_WHB(Zwhb,Uwhb,method=method,scalar=scalar)
# 	# breakpoint()

# 	CM0 = generate_layercake_slow(Uv=Uwhb*1e-3,Zv=[0]+list(Zwhb))
# 	CM0 = CM0.simplify()
# 	MODS = []
# 	Vv = np.arange(1e3/Uwhb[-1],Vmax + dv,dv)
# 	# Iterate across velocities for lowermost layer
# 	for V_ in Vv:
# 		# Append a "half-space" to the shallow model
# 		CMi = add_halfspace(CM0,V_)
# 		# Iterate across reflector depths
# 		for Z_ in Zv:
# 			print('%.2e -- %.2e'%(V_,Z_))
# 			# Calculate modeled values
# 			# NTS: the raytracing_NMO() method has an issue - switch to a lower complexity method...
# 			tti,ddi,thetai = raytracing_NMO(CMi,xx,tt,Z_,dx=dx,n_ref=n_ref)
# 			# Calculate residuals
# 			res = tt - tti
# 			# Get stats on residuals
# 			resL2 = np.linalg.norm(res)
# 			res_u = np.mean(res)
# 			res_o = np.std(res)
# 			# Summarize estimate
# 			line = [Z_,V_,resL2,res_u,res_o,len(xx)]
# 			MODS.append(line)

# 	df_out = pd.DataFrame(MODS,columns=['Z m','VN m/s','res L2','res mean','res std','ndata'])
# 	return df_out



def raytracing_Vsearch(xx,tt,Vv,Uwhb,Zwhb,ZN,dx=20,n_ref=0,Hhs=4000,full=False):
	"""
	Conduct a parameter-search fitting for a flat reflector in a layered medium
	where the thickness of the lowermost layer (Nth layer) is guessed 

	:: INPUTS ::
	:param xx: (n,) receiver locations
	:param tt: (n,) reflection two-way travel times
	:param Zv: (m,) vector of reflector depths to check
	:param Uwhb: slowness profile from WHB analysis [in msec/m]
	:param Zwhb: depth profile from WHB analysis [in m BGS]
	:param Vmax: maximum velocity to consider [in m/sec]
	:param dv: velocity increment for grid-search [in m/sec]
	:param dx: Model station spacing for raytrace_NMO()
	:param n_ref: Number of reflections for raytrace_NMO()
	:param Hhs: interval thickness of half-space (make larger than max(Zv))
	:param full: [BOOL] include array of data-model residuals as 2nd output?

	:: OUTPUT ::
	:return df_out: pandas.DataFrame with summary of models and data-model residuals
			Columns:
				Z m: depth points
				H1 m: thickness of overburden/firn layer
				H2 m: guessed thickness of remaining ice column
				V1rms: RMS velocity of overburden/firn layer
				V2: guessed ice layer velocity
				Vrms: Overall column RMS velocity
				res L2: L-2 norm of data-model residuals for travel-times
				res mean: Average data-model residual value
				res std: standard deviation of data-model residuals
				ndata: number of datapoints
	:return res_block: if full==True, returns an (m,n) array of data-model residuals
	"""
	# Create velocity model for shallow section
	CM0 = generate_layercake_slow(Uv=Uwhb*1e-3,Zv=[0]+list(Zwhb))
	# Use "simplify" method on shallow model to reduce number of layers
	CM0 = CM0.simplify()
	# If no explicit N^th layer velocity is provided, use the bottom value from the WHB profiled
	if full:
		res_block = []
	MODS = []
	# Iterate across V_N values
	for V_ in tqdm(Vv):
		# Append VN "half-space
		CMi = add_halfspace(CM0,V_,Hn=Hhs)
		# Conduct ray-tracing for specified for number of reflections
		# NTS: the raytracing_NMO() method has an issue - switch to a lower complexity method...
		# tti,ddi,thetai = raytracing_NMO(CMi,xx,tt,ZN,dx=dx,n_ref=n_ref)
		# Divide max by 2 to do half-path assumption in raytrace_summary()
		x_hat = np.arange(0,np.max(xx)/2+dx,dx)
		# Run up-going ray-tracing from common midpoint
		t_hat,d_hat,O_hat = raytrace_summary(CMi,x_hat,ZN)
		# Interpolate with double reference distance and double travel-time to get twtt
		tti = np.interp(xx,2*x_hat,2*t_hat)
		# # Interpolate with single reference distance for incidence angle (redundant here)
		# OOi = np.interp(xx,x_hat,O_hat)
		# breakpoint()
		# Calculate residuals
		res = tt - tti
		if full:
			res_block.append(res)
		# Get stats on residuals
		resL2 = np.linalg.norm(res)
		res_u = np.mean(res)
		res_o = np.std(res)
		# Summarize estimate
		line = [ZN,V_,resL2,res_u,res_o,len(xx)]
		MODS.append(line)

	df_out = pd.DataFrame(MODS,columns=['Z m','VN m/s','res L2','res mean','res std','ndata'])
	if full:
		res_block = np.array(res_block)
		return df_out, res_block
	else:
		return df_out


def raytracing_Zsearch(xx,tt,Zv,Uwhb,Zwhb,VN=None,dx=10,n_ref=1,Hhs=4000,full=False):
	"""
	Conduct a parameter-search fitting for a flat reflector in a layered medium
	where the thickness of the lowermost layer (Nth layer) is guessed 

	:: INPUTS ::
	:param xx: (n,) receiver locations
	:param tt: (n,) reflection two-way travel times
	:param Zv: (m,) vector of reflector depths to check
	:param Uwhb: slowness profile from WHB analysis [in msec/m]
	:param Zwhb: depth profile from WHB analysis [in m BGS]
	:param Vmax: maximum velocity to consider [in m/sec]
	:param dv: velocity increment for grid-search [in m/sec]
	:param dx: Model station spacing for raytrace_NMO()
	:param n_ref: Number of reflections for raytrace_NMO()
	:param Hhs: interval thickness of half-space (make larger than max(Zv))
	:param full: [BOOL] include array of data-model residuals as 2nd output?

	:: OUTPUT ::
	:return df_out: pandas.DataFrame with summary of models and data-model residuals
			Columns:
				Z m: depth points
				H1 m: thickness of overburden/firn layer
				H2 m: guessed thickness of remaining ice column
				V1rms: RMS velocity of overburden/firn layer
				V2: guessed ice layer velocity
				Vrms: Overall column RMS velocity
				res L2: L-2 norm of data-model residuals for travel-times
				res mean: Average data-model residual value
				res std: standard deviation of data-model residuals
				ndata: number of datapoints
	:return res_block: if full==True, returns an (m,n) array of data-model residuals
	"""
	# Create velocity model for shallow section
	CM0 = generate_layercake_slow(Uv=Uwhb*1e-3,Zv=[0]+list(Zwhb))
	# Use "simplify" method on shallow model to reduce number of layers
	CM0 = CM0.simplify()
	# If no explicit N^th layer velocity is provided, use the bottom value from the WHB profiled
	if VN is None:
		VN = 1e3/Uwhb[-1]
	# Append VN "half-space
	CM0 = add_halfspace(CM0,VN,Hn=Hhs)
	# Iterate across reflector depths
	if full:
		res_block = []
	MODS = []
	for Z_ in tqdm(Zv):
		# Conduct ray-tracing for specified for number of reflections
		# NTS: the raytracing_NMO() method has an issue - switch to a lower complexity method...
		# tti,ddi,thetai = raytracing_NMO(CM0,xx,tt,Z_,dx=dx,n_ref=n_ref)
		# Divide max by 2 to do half-path assumption in raytrace_summary()
		x_hat = np.arange(0,np.max(xx)/2+dx,dx)
		# Run up-going ray-tracing from common midpoint
		t_hat,d_hat,O_hat = raytrace_summary(CM0,x_hat,Z_)
		# Interpolate with double reference distance and double travel-time to get twtt
		tti = np.interp(xx,2*x_hat,2*t_hat)
		# # Interpolate with single reference distance for incidence angle (redundant here)
		# OOi = np.interp(xx,x_hat,O_hat)

		# Calculate residuals
		res = tt - tti
		if full:
			res_block.append(res)
		# Get stats on residuals
		resL2 = np.linalg.norm(res)
		res_u = np.mean(res)
		res_o = np.std(res)
		# Summarize estimate
		line = [Z_,VN,resL2,res_u,res_o,len(xx)]
		MODS.append(line)

	df_out = pd.DataFrame(MODS,columns=['Z m','VN m/s','res L2','res mean','res std','ndata'])
	if full:
		res_block = np.array(res_block)
		return df_out, res_block
	else:
		return df_out



def raytracing_gridsearch(xx,tt,Vv,Zv,Uwhb,Zwhb,dx=20,Hhs=4000,full=False):
	"""
	Conduct a parameter-search fitting for a flat reflector in a layered medium
	where the thickness of the lowermost layer (Nth layer) is guessed 

	:: INPUTS ::
	:param xx: (n,) receiver locations
	:param tt: (n,) reflection two-way travel times
	:param Zv: (m,) vector of reflector depths to check
	:param Uwhb: slowness profile from WHB analysis [in msec/m]
	:param Zwhb: depth profile from WHB analysis [in m BGS]
	:param Vmax: maximum velocity to consider [in m/sec]
	:param dv: velocity increment for grid-search [in m/sec]
	:param dx: Model station spacing for raytrace_NMO()
	:param n_ref: Number of reflections for raytrace_NMO()
	:param Hhs: interval thickness of half-space (make larger than max(Zv))
	:param full: [BOOL] include array of data-model residuals as 2nd output?

	:: OUTPUT ::
	:return df_out: pandas.DataFrame with summary of models and data-model residuals
			Columns:
				Z m: depth points
				H1 m: thickness of overburden/firn layer
				H2 m: guessed thickness of remaining ice column
				V1rms: RMS velocity of overburden/firn layer
				V2: guessed ice layer velocity
				Vrms: Overall column RMS velocity
				res L2: L-2 norm of data-model residuals for travel-times
				res mean: Average data-model residual value
				res std: standard deviation of data-model residuals
				ndata: number of datapoints
	:return res_block: if full==True, returns an (m,n) array of data-model residuals
	"""
	# Create velocity model for shallow section
	CM0 = generate_layercake_slow(Uv=Uwhb*1e-3,Zv=[0]+list(Zwhb))
	# Use "simplify" method on shallow model to reduce number of layers
	CM0 = CM0.simplify()
	# If no explicit N^th layer velocity is provided, use the bottom value from the WHB profiled
	if full:
		res_block = []
	MODS = []
	# Iterate across V_N values
	for i_,V_ in enumerate(Vv):
		# Append VN "half-space
		CMi = add_halfspace(CM0,V_,Hn=Hhs)
		for j_,Z_ in tqdm(enumerate(Zv)):
			# Conduct ray-tracing for specified for number of reflections
			# NTS: the raytracing_NMO() method has an issue - switch to a lower complexity method...
			# tti,ddi,thetai = raytracing_NMO(CMi,xx,tt,ZN,dx=dx,n_ref=n_ref)
			# Divide max by 2 to do half-path assumption in raytrace_summary()
			x_hat = np.arange(0,np.max(xx)/2+dx,dx)
			# Run up-going ray-tracing from common midpoint
			t_hat,d_hat,O_hat = raytrace_summary(CMi,x_hat,Z_)
			# Interpolate with double reference distance and double travel-time to get twtt
			tti = np.interp(xx,2*x_hat,2*t_hat)
			# # Interpolate with single reference distance for incidence angle (redundant here)
			# OOi = np.interp(xx,x_hat,O_hat)
			# breakpoint()
			# Calculate residuals
			res = tt - tti
			if full:
				res_block.append(res)
			# Get stats on residuals
			resL2 = np.linalg.norm(res)
			res_u = np.mean(res)
			res_o = np.std(res)
			# Summarize estimate
			if isinstance(Z_,np.ndarray):
				Z_ = Z_[0]
			if isinstance(V_,np.ndarray):
				V_ = V_[0]
			line = [i_,j_,Z_,V_,resL2,res_u,res_o,len(xx)]
			MODS.append(line)

	df_out = pd.DataFrame(MODS,columns=['indZ','indV','Z m','VN m/s','res L2','res mean','res std','ndata'])
	if full:
		res_block = np.array(res_block)
		return df_out, res_block
	else:
		return df_out



#### LEAST SQUARES METHODS ####

def hyperbolic_curvefit(xx,tt,tsig,p0=[400,3850],bounds=[(100,1000),(3600,4100)],method='trf',absolute_sigma=True):
	"""
	Wrapper for scipy.optimize.curve_fit for the hyperbolic travel-time equation
	This method provides options to bound parameters at the expense of not including
	uncertainties in station location (xsig) as is the case with the ODR implementation
	(see util.Dix_1D_Raytrace_Analysis.hyperbolic_ODR()).
	
	*For most imputs, see scipy.optimize.curve_fit() for further explanation

	:: INPUTS ::
	:param xx: station locations in meters
	:param tt: reflected arrival travel times in seconds
	:param tsig: picking errors in seconds
	:param p0: inital parameter guesses*
	:param bounds: parameter-bounding domains*
	:param method: solving method*
	:param absolute_sigma: Default to True - means tsig is used in an absolute sense for pcov

	:: OUTPUTS ::
	:return popt: Best-fit parameter estiamtes
	:return pcov: Best-fit parameter estimate covariance matrix

	"""
	popt,pcov = curve_fit(hyperbolic_tt,xx,tt,p0=p0,bounds=bounds,method=method,sigma=tsig,absolute_sigma=absolute_sigma)
	return popt,pcov

def hyperbolic_ODR(xx,tt,xsig,tsig,beta0=[550,3700],ifixb=None,fit_type=0):
	"""
	Fit a hyperbolic NMO curve to data using Orthogonal Distance Regression

	:: INPUTS ::
	:param xx: station-reciver offsets
	:param tt: two way travel times
	:param xsig: station-receiver offset uncertainties
	:param tsig: two way travel time uncertainties
	:param beta0: initial parameter estimates [ice thickness, average column velocity]
	:param ifixb: None or array-like of ints of rank-1 and equal length to beta: 
						0 = free parameter
						1 = fixed parameter
	:param fit_type: input to odr.set_job - 0 = ODR
											1 = ODR output including optional parameters
											2 = LSQ

	:: OUTPUT ::
	:param output: scipy.odr.Output - output produced by a scipy ODR run. 
					'beta' - parameter estimates (included in all fit_types)
					'cov_beta' - model covariance matrix (included in all fit_types)
	"""

	output = inv.curve_fit_2Derr(hyperbolic_tt_ODR,xx,tt,xsig,tsig,beta0=beta0,ifixb=ifixb,\
								 fit_type=fit_type)

	return output


