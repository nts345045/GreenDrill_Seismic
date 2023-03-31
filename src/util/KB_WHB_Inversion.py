"""
:module: KirchnerBentley_Model.py
:auth: Nathan T. Stevens
:purpose: Contains methods for forward and inverse treatments of the 
		 double-exponential model for direct-wave arrival travel times 
		 from Kirchner & Bentley (1979) 

"""
import numpy as np
import scipy.optimize as spo
import scipy.odr as odr

def KB79_exp_fun(xx,aa,bb,cc,dd,ee):
	"""
	Calculate travel times (t) for the doube-exponential
	function from Kirchner & Bentley (1979)

	:: INPUTS ::
	:param xx: horizontal shot-receiver offsets
	:param aa: exponential prefactor a_1
	:param bb: exponential coefficient a_2
	:param cc: exponential prefactor a_3
	:param dd: exponential coefficient a_4
	:param ee: linear prefactor a_5

	:: OUTPUT ::
	:return tt: travel times
	"""

	tt = aa*(1. - np.exp(-bb*xx)) + \
		 cc*(1. - np.exp(-dd*xx)) + ee*xx
	return tt

def KB79_exp_ext_fun(xx,aa,bb,cc,dd,ee,t0):
 	"""
	Calculate travel times (t) for the double-exponential
	function from Kirchner & Bentley (1979) that includes
	a floating intercept (t0) to allow for errors unresolved
	by travel-time corrections
 	"""
 	tt = aa*(1. - np.exp(-bb*xx)) + cc*(1. - np.exp(-dd*xx)) + ee*xx + t0
 	return tt

def KB79_odr_fun(beta,xx):
	"""
	KB79_exp_fun formatted for use in Orthogonal Distance Regression
	inversion

	"""
	tt = KB79_exp_fun(xx,*beta)
	return tt

def KB79_ext_odr_fun(beta,xx):
	"""
	KB79_exp_ext_fun formatted use in Orthogonal Distance Regression
	inversion
	"""
	tt = KB79_exp_ext_fun(xx,*beta)
	return tt

def KB79_exp_fun_deriv(xx,aa,bb,cc,dd,ee):
	"""
	Calculate apparent slownesses (first spatial derivative) 
	for the doube-exponentialfunction from Kirchner & Bentley (1979)

	This also applies for the EXTENDED version of the function
	as \\partial_x t0 = 0

	:: INPUTS ::
	:param xx: horizontal shot-receiver offset(s)
	:param aa: exponential prefactor a_1
	:param bb: exponential coefficient a_2
	:param cc: exponential prefactor a_3
	:param dd: exponential coefficient a_4
	:param ee: linear prefactor a_5

	:: OUTPUT ::
	:return dtdx: apparent slowness(es)
	"""
	dtdx = aa*bb*np.exp(-bb*xx) + cc*dd*np.exp(-dd*xx) + ee
	return dtdx

def curvefit_KB79(xx,tt,p0=[10.,0.1,1.,0.1,0.1],bounds=(0,np.inf)):
	"""
	Wrapper for a trust-region-reflective (trf) least squares fitting
	of the Kirchner & Bentley (1979) double exponential function (KB79)
	to travel-time/offset data. Accepts initial guesses of parameters.

	:: INPUTS ::
	:param xx: offset data
	:param tt: diving ray travel time data in MILLISECONDS
	:param p0: initial parameter guesses - orders of magnitude scaled from 
	:param bounds: parameter limits - keep this as (0,inf) to ensure estimates
					are only positive valued.

	:: OUTPUT ::
	:return popt: parameter estimates
	:return pcov: parameter estimate covariance matrix
	"""
	popt,pcov = spo.curve_fit(KB79_exp_fun,xx,tt,p0=p0,method='trf',bounds=bounds)
	return popt,pcov

def curvefit_KB79_ext(xx,tt,p0=[10.,0.1,1.,0.1,0.1,0],bounds=(0,np.inf)):
	"""
	Wrapper for a trust-region-reflective (trf) least squares fitting
	of the Kirchner & Bentley (1979) double exponential function (KB79)
	to travel-time/offset data. Accepts initial guesses of parameters.

	EXTENDED version that allows for non-0 valued intercept

	:: INPUTS ::
	:param xx: offset data
	:param tt: diving ray travel time data in MILLISECONDS
	:param p0: initial parameter guesses - orders of magnitude scaled from 
	:param bounds: parameter limits - keep this as (0,inf) to ensure estimates
					are only positive valued.

	:: OUTPUT ::
	:return popt: parameter estimates
	:return pcov: parameter estimate covariance matrix
	"""
	popt,pcov = spo.curve_fit(KB79_exp_ext_fun,xx,tt,p0=p0,method='trf',bounds=bounds)
	return popt,pcov


def ODR_KB79(xx,tt,xsig,tsig,beta0=np.ones(5,),fit_type=0):
	"""
	Orthogonal Distance Regression treatment to estimate parameters for the 
	Kirchner & Bentley (1979) function - allows consideration of errors in the
	source-receiver offsets and travel times

	:: INPUTS ::
	:param xx: source-receiver offsets in meters
	:param tt: diving ray travel-times in MILLISECONDS
	:param xsig: standard deviations of xx - data weights become 1/xsig**2
	:param tsig: standard deviations of tt - data weights become 1/tsig**2
	:param beta0: initial parameter estimates
	:param fit_type: input to odr.set_job - 0 = full ODR
											2 = LSQ

	:: OUTPUT ::
	:param output: scipy.odr.Output - output produced by a scipy ODR run. 
					'beta' - parameter estimates
					'cov_beta' - model covariance matrix

	"""
	# Write data and standard errors for fitting to data object
	data = odr.RealData(xx,tt,xsig,tsig)
	# Define model
	model = odr.Model(KB79_odr_fun)
	# Compose Orthogonal Distance Regression 
	_odr_ = odr.ODR(data,model,beta0=beta0)
	# Set solution type
	_odr_.set_job(fit_type=fit_type)
	# Run regression
	output = _odr_.run()
	return output


def ODR_KB79_ext(xx,tt,xsig,tsig,beta0=[3,1,10,1e-2,0.1,0],fit_type=0):
	"""
	Orthogonal Distance Regression treatment to estimate parameters for the 
	Kirchner & Bentley (1979) function - allows consideration of errors in the
	source-receiver offsets and travel times

	EXTENDED: Design matrix includes possibility of non-zero intercept

	:: INPUTS ::
	:param xx: source-receiver offsets in meters
	:param tt: diving ray travel-times in MILLISECONDS
	:param xsig: standard deviations of xx - data weights become 1/xsig**2
	:param tsig: standard deviations of tt - data weights become 1/tsig**2
	:param beta0: initial parameter estimates
	:param fit_type: input to odr.set_job - 0 = full ODR
											2 = LSQ

	:: OUTPUT ::
	:param output: scipy.odr.Output - output produced by a scipy ODR run. 
					'beta' - parameter estimates
					'cov_beta' - model covariance matrix

	"""
	# Write data and standard errors for fitting to data object
	data = odr.RealData(xx,tt,xsig,tsig)
	# Define model
	model = odr.Model(KB79_ext_odr_fun)
	# Compose Orthogonal Distance Regression 
	_odr_ = odr.ODR(data,model,beta0=beta0)
	# Set solution type
	_odr_.set_job(fit_type=fit_type)
	# Run regression
	output = _odr_.run()
	return output



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
	to travel-time(offset) data.

	THIS METHOD ASSUMES abcde ARE INVERTED USING DATA WITH TIME UNITS OF MILLISECONDS
	
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
			if np.nanmin(uavi) <= 0:
				if len(uavi[uavi > 0]) > 0:
					uavi[uavi <= 0] = np.min(uavi[uavi > 0])
				else:
					uavi[uavi <= 0] = 1.
			if uDi > np.min(uavi):
				uDi = np.min(uavi)
			z_uai = dx*np.arccosh(uavi/uDi)
			z_uDi = (1./np.pi)*np.sum(z_uai)
			# if ~np.isfinite(z_uDi):
				# breakpoint()
			XX.append(X_)
			z_uD.append(z_uDi)
			uD.append(uDi)
	z_uDv = {'X':XX,'z m':z_uD,'uD ms/m':uD}
	return z_uDv





# def run_KB79_fit_analysis(df_picks,xdv,kinds=[1,2],p0=np.ones(5),bounds=(0,np.inf)):
# 	"""
# 	Run Kirchner & Bentley (1979) analysis on phase data, testing
# 	the model-fit quality as a function of maximum data offsets considered.
# 	Model-fit quality is assessed using the L-2 norm of model-data residuals.

# 	NTS: This doesn't see much use - from an earlier thought-process

# 	:: INPUTS ::
# 	:param df_picks: Compiled Picks from S1
# 	:param xdv: increment to increase maximum offset assessed for fit-quality
# 	:param kinds: list of kinds of picks to assess


# 	"""
# 	df_ = df_picks.copy().sort_values('SRoff m')
# 	MODELS = []
# 	for K_ in kinds:
# 		print('Running kind %d'%(K_))
# 		for xd in xdv:
# 			IND = (df_['kind']==K_) & \
# 				  (df_['SRoff m'] <= xd) & \
# 				  (df_['SRoff m'].notna())
# 			if len(df_[IND]['SRoff m'].unique()) > 6 and len(df_[~IND]['SRoff m'].unique()) > 3:
# 				# Pull data vectors
# 				xx = df_[IND]['SRoff m'].values
# 				tt = df_[IND]['tt sec'].values
# 				# Fit parameters for KB79 double-exponential model
# 				try:
# 					popt,pcov = curvefit_KB79(xx,tt,p0=p0,bounds=bounds)

# 					# Calculate model-data residuals
# 					tt_hat = KB79_exp_fun(xx,*popt)
# 					res = tt_hat - tt
# 					resL2 = np.linalg.norm(res)
# 					res_u = np.mean(res)
# 					res_o = np.std(res)


# 				except RuntimeError:
# 					popt = np.nan*np.ones(5)
# 					pcov = np.nan*np.ones((5,5))
# 					resL2 = np.nan
# 					res_u = np.nan
# 					res_o = np.nan
# 				modline = [K_,xd,len(xx),resL2,res_u,res_o]
# 				for i_ in range(5):
# 					modline.append(popt[i_])
# 				for i_ in range(5):
# 					for j_ in range(5):
# 						if i_<=j_:
# 							modline.append(pcov[i_,j_])
# 				MODELS.append(modline)
# 	df_out = pd.DataFrame(MODELS,columns=['kind','X m','npts','res L2','res u','res o',\
# 										  'aa','bb','cc','dd','ee',\
# 										  'caa','cab','cac','cad','cae',\
# 										  'cbb','cbc','cbd','cbe',\
# 										  'ccc','ccd','cce',\
# 										  'cdd','cde',\
# 										  'cee'])

# 	return df_out

# def curvefit_KB79DT(xx,tt,) - see TODO in project.Prudhoe.S1_Time_Geom_Processing.py


# def run_KB79_fit_analysis(df_PHZ,xdv,kinds=[1,2],p0=np.ones(5),bounds=(0,np.inf)):
# 	"""
# 	Run Kirchner & Bentley (1979) analysis on phase data, testing
# 	the model-fit quality as a function of maximum data offsets considered.
# 	Model-fit quality is assessed using the L-2 norm of model-data residuals.

# 	:: INPUTS ::
# 	:param df_PHZ: [pd.DataFrame] containing phase-pick data with fields including:
# 					'kind','SRoff m','tt sec'

# 	"""
# 	df_ = df_PHZ.copy().sort_values('SRoff m')
# 	MODELS = []
# 	for K_ in kinds:
# 		print('Running kind %d'%(K_))
# 		for xd in xdv:
# 			IND = (df_['kind']==K_) & \
# 				  (df_['SRoff m'] <= xd) & \
# 				  (df_['SRoff m'].notna())
# 			if len(df_[IND]['SRoff m'].unique()) > 6 and len(df_[~IND]['SRoff m'].unique()) > 3:
# 				# Pull data vectors
# 				xx = df_[IND]['SRoff m'].values
# 				tt = df_[IND]['tt sec'].values
# 				# Fit parameters for KB79 double-exponential model
# 				try:
# 					popt,pcov = curvefit_KB79(xx,tt,p0=p0,bounds=bounds)

# 					# Calculate model-data residuals
# 					tt_hat = KB79_exp_fun(xx,*popt)
# 					res = tt_hat - tt
# 					resL2 = np.linalg.norm(res)
# 					res_u = np.mean(res)
# 					res_o = np.std(res)


# 				except RuntimeError:
# 					popt = np.nan*np.ones(5)
# 					pcov = np.nan*np.ones((5,5))
# 					resL2 = np.nan
# 					res_u = np.nan
# 					res_o = np.nan
# 				modline = [K_,xd,len(xx),resL2,res_u,res_o]
# 				for i_ in range(5):
# 					modline.append(popt[i_])
# 				for i_ in range(5):
# 					for j_ in range(5):
# 						if i_<=j_:
# 							modline.append(pcov[i_,j_])
# 				MODELS.append(modline)
# 	df_out = pd.DataFrame(MODELS,columns=['kind','X m','npts','res L2','res u','res o',\
# 										  'aa','bb','cc','dd','ee',\
# 										  'caa','cab','cac','cad','cae',\
# 										  'cbb','cbc','cbd','cbe',\
# 										  'ccc','ccd','cce',\
# 										  'cdd','cde',\
# 										  'cee'])

# 	return df_out