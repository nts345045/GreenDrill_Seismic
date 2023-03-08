"""
:module: InvTools.py
:auth: Nathan T. Stevens
:purpose: Generalized inversion / error-propagation tools for use in 

"""
from pyDOE import lhs
import scipy.odr as odr
import numpy as np
from scipy.stats.distributions import norm

def norm_lhs(mean_vector,cov_matrix, n_samps = 1000, criterion='maximin'):
		"""
		Compilation of steps from Gou's article* on MC simulations using LHS
		to provide perturbed parameter estimates given a vector of expected parameter values
		and their corresponding covariance matrix

		*TowardsDataScience article by Shuai Gou: 
		https://towardsdatascience.com/performing-uncertainty-analysis-in-three-steps-a-hands-on-guide-9110b120987e
		
		:: INPUTS ::
		:type mean_vector: numpy.ndarray - n, or n,1 vector
		:param mean_vector: ordered vector of mean parameter estiamtes
		:type cov_matrix: numpy.ndarray - n x n array
		:param cov_matrix: ordered covariance matrix corresponding to entries in mean_vector
		:type n_samps: int
		:para n_samps: Number of target_samples to generate
		:type criterion: str
		:param criterion: See pyDOE.lhs

		:: OUTPUT ::
		:rtype target_samples: numpy.ndarray [m,n_samps]
		:return target_samples: Array of generated samples from the n-dimensional mean + cov_matrix normal distribution
		"""
		# 0) Detect the number of parameters & do sanity checks
		nfrommv = len(mean_vector)
		nfromcm,mfromcm = cov_matrix.shape
		if nfrommv == nfromcm == mfromcm:
				n_params = nfrommv
		elif nfrommv > nfromcm == nfromcm:
			print('In development, assume dropped COV matrix entries mean 0-(co)variance')
			n_params = nfrommv
		else:
			print('Poorly scaled input distributions, reassess')
			pass
		# 1) Conduct LHS to produce Uniform PDF samples
		uni_samples = lhs(n=n_params,samples=n_samps,criterion=criterion)
		# 2) Conduct inverse transformation sampling to create standard normal distributions
		std_norm_samples = np.zeros_like(uni_samples)
		for i_ in range(n_params):
			std_norm_samples[:,i_] = norm(loc=0,scale=1).ppf(uni_samples[:,i_])
		# 3) Convert standard normals into bivariate normals
		L = np.linalg.cholesky(cov_matrix)
		target_samples = mean_vector + std_norm_samples.dot(L.T)
		return target_samples


def lin_fun(beta,xx):
	return (beta[0]*xx) + beta[1]

def quad_fun(beta,xx):
	return (beta[0]*xx**2) + (beta[1]*xx) + beta[2]

def cube_fun(beta,xx):
	return (beta[0]*xx**3) + (beta[1]*xx**2) + (beta[2]*xx) + beta[3]


# def curve_fit_2Derr(fun,xx,yy,xsig,ysig,beta0=None,low_bnds=None,ifixb=None,fit_type=0):
def curve_fit_2Derr(fun,xx,yy,xsig,ysig,beta0=None,ifixb=None,fit_type=0):
	"""
	Orthogonal Distance Regression treatment to estimate parameters for the 
	Kirchner & Bentley (1979) function - allows consideration of errors in the
	source-receiver offsets and travel times

	:: INPUTS ::
	:param fun: function with which to generate global matrix for parameter estimation
	:param xx: independent data
	:param yy: dependent data
	:param xsig: standard deviations of xx - data weights become 1/xsig**2
	:param ysig: standard deviations of tt - data weights become 1/tsig**2
	:param beta0: initial parameter estimates
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

	Reference: https://stackoverflow.com/questions/26058792/correct-fitting-with-scipy-curve-fit-including-errors-in-x
	"""
	# Write data and standard errors for fitting to data object
	data = odr.RealData(xx,yy,xsig,ysig)
	# Define model
	model = odr.Model(fun)
	# Compose Orthogonal Distance Regression 
	_odr_ = odr.ODR(data,model,beta0=beta0)
	# Set solution type
	_odr_.set_job(fit_type=fit_type)
	# Set boundaries if needed
	# if low_bnds is not None:
	# 	_odf
	# 	if hi_bnds is not None:
	# 		_odr_.set_iprint(stpb=low_bnds,maxparm=hi_bnds)
	# 	else:
	# 		_odr_.set_iprint(stpb=low_bnds)
	# else:
	# 	if hi_bnds is not None:
	# 		_odr_.set_iprint(maxparm=hi_bnds)
	# 	else:
	# 		pass

	# Run regression
	output = _odr_.run()

	return output


