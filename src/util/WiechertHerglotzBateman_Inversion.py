"""
:module: WiechertHerglotzBateman_Inversion.py
:purpose: Methods for implementation of the Wiechert-Herglotz-Bateman inversion 
for vertical slowness profile models from apparent horizontal velocities.

:auth: Nathan T. Stevens
:last update: 10. February 2023
:email: ntstevens@wisc.edu | nts5045@psu.edu 

:: TODO ::
-Test scipy's integral estimation methods to solve for the WHB integral and compare to results from trapezoidal summation
"""
import numpy as np

def est_WHB_int(XX,dx=1.,abcde=np.ones(5),sig_rule='trap'):
	"""
	Conduct a summation approximate a single solution to the Wiechert-Herglotz-Bateman (WHB)
	integral that solves for turning depths associated with the maximum offset (XX). This 
	assumes the first kind of travel-time function as defined in Slichter (1932)

	:: INPUTS ::
	:param XX: maximum offset to associate output z(u_D)
	:param dx: spatial step for xv~np.linspace(0,XX,dx) -- NOTE See param sig_rule
	:param abcde: ordered array-like object of coefficients a,b,c,d,e for the derivative of the
				  double-exponential function of Kirchner & Bentley (1979)
	:param sig_rule: STRING - 'trap' - trapezoidal, or midpoint rule, set xv = np.arange(0.5*dx,XX + 0.5dx,dx)
							  'left' - lefthand Riemann, set xv = np.arange(0,XX,dx)
							  'right' - righthand Riemann, set xv = np.arange(dx,XX+dx,dx)
	:: OUTPUT ::
	:return z_uD: turning depth associated with the u(x) function defined by abcde and the maximum offset XX

	"""
	# Set grid for integration 
	if sig_rule == 'left':
		xv = np.arange(0,XX,dx)
	elif sig_rule == 'right':
		xv = np.arange(dx,XX+dx,dx)
	else:
		xv = np.arange(0.5*dx,XX + 0.5*dx, dx)
	# Calculate ray parameter at the max offset
	uD = KB79_exp_fun_deriv(xv[-1],*abcde)
	# Calculate apparent slownesses at all points
	# NOTE: Leave out last point to avoid a 0-valued / inf-valued result from arccosh
	uuv = KB79_exp_fun_deriv(xv[:-1],*abcde)
	# Create elements of integral sum
	z_ua_i = dx*np.arccosh(uuv/uD)
	# Approximate WHB integral for inversion
	z_uD = (1./np.pi)*np.sum(z_ua_i)
	# Return turning depth and associated slowness
	return z_uD, uD


def loop_WHB_int(XX,dx=1.,abcde=np.ones(5),sig_rule='trap'):
	"""
	Conduct Wiechert-Herglotz-Bateman inversion for a vertical slowness
	profile from a Kirchner & Bentley (1979) double-exponential model fit
	to travel-time(offset) data
	
	:: INPUTS ::
	:param XX: maximum offset for which to estimate effective slownesses
	:param dx: grid discretization for progressive integral estimates
	:param abcde: ordered array-like object of coefficients a,b,c,d,e for the derivative of the
				  double-exponential function of Kirchner & Bentley (1979)
	:param sig_rule: STRING - 'trap' - trapezoidal, or midpoint rule, set xv = np.arange(0.5*dx,XX + 0.5dx,dx)
							  'left' - lefthand Riemann, set xv = np.arange(0,XX,dx)
							  'right' - righthand Riemann, set xv = np.arange(dx,XX+dx,dx)
	:: OUTPUT ::
	:return z_uDv: turning depth vector associated with the u(x) functions defined by abcde and the maximum offset XX
	"""
	# Set grid for integration 
	if sig_rule == 'left':
		xv = np.arange(0,XX,dx)
	elif sig_rule == 'right':
		xv = np.arange(dx,XX+dx,dx)
	else:
		xv = np.arange(0.5*dx,XX + 0.5*dx, dx)
	z_uD = []; uD = []; XX = []
	for i_,X_ in enumerate(xv):
		if i_ > 0:
			xvi = xv[:i_]
			uDi = KB79_exp_fun_deriv(X_,*abcde)
			uavi = KB79_exp_fun_deriv(xvi,*abcde)
			z_uai = dx*np.arccosh(uavi/uDi)
			z_uDi = (1./np.pi)*np.sum(z_uai)
			if ~np.isfinite(z_uDi):
				breakpoint()
			XX.append(X_)
			z_uD.append(z_uDi)
			uD.append(uDi)
	z_uDv = {'X':XX,'z m':z_uD,'uD sec/m':uD}
	return z_uDv


	