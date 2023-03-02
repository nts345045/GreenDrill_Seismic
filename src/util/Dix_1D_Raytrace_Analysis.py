"""
:module: Dix_Conversion.py
:purpose: Contains methods supporting Dix Conversion for estimating
interval velocities 

:: TODO ::
for hyperbolic_fit - update to include at least the 1D ray-tracer for layered structure
merge in RayTarcing1D.py methods


"""
import numpy as np
import sys
import os
from pyrocko import cake
from copy import deepcopy


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
	dt = dz/Vbar
	# Calculate the RMS velocity
	num = np.sum(dt*Vbar**2)
	den = np.sum(dt)
	Vrms = (num/den)**0.5
	return Vrms


### WHB PROFILE RESAMPLING

def resample_WHB(Z,uD,method='incremental increase',scalar=1.3):

	if method.lower() == 'incremental increase':
		# Get surface coordinates
		u0 = uD[0]; z0 = 0.
		# Get interval where ui <= u0*scalar
		I1 = uD <= u0*scalar; z1 = Z[I1][-1]; u1 = uD[I1][-1]
		# Create holders for resampled slowness profile
		u_int = []; z_top_int = []; z_bot_int = [];
		# Calculate first interval
		ui = calc_Vrms_cont(Z[I1],uD[I1]**-1)**-1
		z_top_int.append(z0); z_bot_int.append(z1)
		# While there is still a usable profile
		while z1 < Z:
			# Advance bounds and calculate new index
			z0 = z1; u0 = u1
			I1 = u0 < I1 <= u0*scalar
			z1 = Z[I1]
			# Calculate RMS slowness within new layer
			ui = calc_Vrms_cont(Z[I1],uD[I1]**-1)**-1
			# Save entries
			u_int.append(ui)
			z_top_int.append(z0); z_bot_int.append(z1)

	# elif method.lower() == 'uniform'
	# 	u_int = [uD[0]]; z_top_int = [0]; z_bot_int = [scalar]

	mod_out = {'Ztop':z_top_int,'Zbot':z_bot_int,'uRMS':u_int}
	return mod_out


### 1D RAY TRACING ###
def generate_layercake_slow(Sv=[1/2500,1/3750,1/5000],Zv=[0,10,490,4000]):
	"""
	Generate a 1-D layercake seismic velocity model with homogeneous interval 
	slownesses bounded by Zv horizons. For use with pyrocko.cake

	Note that Zv must have 1 additional entry compared to Sv

	:: INPUTS ::
	:param Sv: array-like set of slownesses in sec/m
	:param Zv: array-like set of horizon depths with z positive down

	:: OUTPUT ::
	:return model: pyrocko.cake.LayeredModel with specified layering
	"""

	model = cake.LayeredModel()
	for i_,S_ in enumerate(Sv):
		mi = cake.Material(vp=S_**-1)
		lay = cake.HomogeneousLayer(Zv[i_],Zv[i_+1],mi)
		model.append(lay)
		if i_ < len(Sv) - 1:
			mj = cake.Material(vp=Sv[i_+1]**-1)
			ifce = cake.Interface(Zv[i_+1],mi,mj,)
		elif i_ == len(Sv) - 1:
			mj = cake.Material(vp=1.01*Sv[-1]**-1)
			ifce = cake.Interface(Zv[-1],mi,mj)
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

def add_halfspace(CakeMod,Vn,Hn=4000):
	"""
	Add a very thick bottom layer to an exsiting 
	"""
	NewMod = deepcopy





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


### DIX FITTING ###

def hyperbolic_fitting(xx,tt,Zv,Uwhb,Zwhb,Vmax=3850,dv=10):
	"""
	Conduct a grid-search hyperbolic fitting using the guessed parameters
	of the thickness of the Nth layer and its velocity using the V_RMS for
	near-normal incidence assumption in Dix conversion analysis.

	:: INPUTS ::
	:param xx: reflected arrival source-receiver offsets in meters
	:param tt: reflected arrival two-way travel times in seconds
	:param Zv: Vector of ice-thickness to guess (firn + glacier ice thickness)
	:param Uwhb: Slowness values from WHB analysis for shallow velocity structure [in sec/m]
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
	Vsb = Uwhb[-1]**-1
	Vv = np.arange(Vsb,Vmax + dV,dV)
	Hsb = Zwhb[-1]
	# Get RMS Velocity from shallow profile
	Vrms = calc_Vrms_cont(Zwhb,Uwhb**-1)
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


def raytracing_gridsearch(CakeMod)

