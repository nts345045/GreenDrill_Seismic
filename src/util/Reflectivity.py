"""
:module: Reflectivity.py
:purpose: Contains methods for estimating source amplitudes ($A_0$)
"""




def estimate_A0_direct(S_dir1,S_dir2,ATYPE='RMS Amp'):
	"""
	Estimate the source-amplitude ($A_0$) using the direct-wave method
	from Holland and Anandakrishnan (2009) [their equation ]
	"""
	B1 = np.abs(S_dir1[ATYPE].values[0])
	B2 = np.abs(S_dir2[ATYPE].values[0])
	d1 = np.abs(S_dir1['dd m'].values[0])
	d2 = np.abs(S_dir2['dd m'].values[0])
	y1 = d1**-0.5
	y2 = d2**-0.5
	A0 = (B1**2/B2)*(y2/y1**2)
	return A0