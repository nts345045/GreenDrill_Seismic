"""
:module: RayTracing1D.py
:auth: Nathan T. Stevens
:purpose: Contains methods built on top of the pyrocko.cake layer-cake velocity model ray-tracing modules.
		Used for calculation of ray-path geometries and travel-times to support velocity model inversions.
:version: 1.0
:last update: 14. Feb 2023

:TODO: 
Update to incorporate ray-tracing methods from ttcrpy in V1.1
"""
import numpy as np
from pyrocko import cake
import matplotlib.pyplot as plt


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

def generate_layercake_vel(Vv=[2500,3750,5000],Zv=[0,10,490,4000]):
	model = cake.LayeredModel()
	for i_,V_ in enumerate(Vv):
		mi = cake.Material(vp=V_)
		lay = cake.HomogeneousLayer(Zv[i_],Zv[i_+1],mi)
		model.append(lay)
		if i_ < len(Vv-1):
			mj = cake.Material(vp=Vv[i_+1])
			ifce = cake.Interface(Zv[i_+1],mi,mj)
		elif i_ == len(Vv) - 1:
			mj = cake.Material(vp=1.01*Vv[-1])
			ifce = cake.Interface(Zv[-1],mi,mj)
	return model

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


def plot_rays(dict_out):
	for k_ in dict_out.keys():
		idict = dict_out[k_]



def run_example(rr=np.arange(0,300,50),Zsrc=490):
	CakeMod = generate_layercake_slow()
	dict_out = raytrace_explicit(CakeMod,rr,Zsrc)
	tt,dd,thetai = raytrace_summary(CakeMod,rr,Zsrc)
	return CakeMod,dict_out,tt,dd,thetai





