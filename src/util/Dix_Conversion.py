"""
:module: Dix_Conversion.py
:purpose: Contains methods supporting Dix Conversion for estimating
interval velocities 

:: TODO ::
for hyperbolic_fit - update to include at least the 1D ray-tracer for layered structure
"""
import numpy as np
import sys
import os




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

### DIX FITTING ###

def hyperbolic_fit_simple(xx,tt,Viv,Ziv,ZNv,VNv):
	"""
	Conduct a grid-search hyperbolic fitting using the guessed parameters
	of the thickness of the Nth layer and its velocity

	:: INPUTS ::
	:param xx: reflected arrival source-receiver offsets in meters
	:param tt: reflected arrival two-way travel times in seconds
	:param iVv: interval Velocity values for the i = 0, N-1 layers
	:param iZv: bottom depths for the i = 0, N-1 layers
	:param ZNv: array of values to guess for the bottom depth of the Nth layer
	:param VNv: array of values to guess for the velocity in the Nth layer

	"""
	for i_,ZN_ in enumerate(ZNv):
		# Compile guessed vertical structure
		idZv = np.append(dZiv.copy(),dZN_)
		for j_,VN_ in enumerate(VNv):
			# Compile velocity profile
			iVv = np.append(Viv,VN_)
			# Calculate vRMS
			Vrms = calc_Vrms_cont(idZv,iVv)
			# Model travel times
			tt_hat = hyperbolic_tt(xx,dZN_)



	# Get bottom depth & velocity of shallow WHB profile
	Vsb = df_WHB['uD sec/m'].values[-1]**-1
	Hsb = df_WHB['z m'].values[-1]
	# Get RMS Velocity from shallow profile
	Vrms = calc_Vrms_cont(df_WHB['z m'].values,df_WHB['uD sec/m'].values**-1)
	# Iterate across guess-depth values
	MODS = []; #RES = []
	for Z_ in Zv:
		# Calculate Vrms for the guessed depth
		Vrmsi = calc_Vrms_2L(Vrms,Vsb,Hsb,Z_ - Hsb)
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
		line = [Z_,Hsb,Z_ - Hsb,Vrms,Vsb,Vrmsi,resL2,res_u,res_o,len(xx)]
		MODS.append(line)
	df_out = pd.DataFrame(MODS,columns=['Z m','H1 m','H2 m','V1rms','V2','Vrms',\
						  'res L2','res mean','res std','ndata'])
	# RES = np.array(RES)
	return df_out