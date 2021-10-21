import numpy as np
from phys import plotLib as plib
import scipy.optimize as sop

def expFit(r, a, b):
	f = a * np.exp(b * r)
	return f

def getExchangePotential (density, functional="LDA"):
	"""
	The initial estimate for the exchange potential is calculated using the
	spin unpolarized local density approximation (LDA). For more information,
	see the following:

	Burke, K. (2007). The ABC of DFT. 92697.
	url: https://dft.uci.edu/doc/g1.pdf
	"""
	n = density
	vexc = - np.power(3.0 / np.pi * n, 1.0 / 3.0)
	return vexc

def getLB94 (density, r, D1, drdx, P, N_g):

	"""
	The LB94 generalized gradient approximation (GGA) is implemented here. For more information,
	see the following:

	van Leeuwen, R., and Baerends, E. J. Phys. Rev. A 49 2421 (1994)
	doi: https://doi.org/10.1103/PhysRevA.49.2421

	The parameter delta is a numerical parameter to soften the singularity encountered in x as
	F -> 0.
	"""

	delta = 1E-8

	xi = np.power(2.0, 1.0 / 3.0)

	G = (N_g * (N_g + 1)) / 2.0 * np.dot(D1, density) * P / np.sqrt(drdx)

	F = (N_g * (N_g + 1)) / 2.0 * density * P / np.sqrt(drdx) / np.power(r, 2.0)

	a = xi * np.absolute((- 2.0 * r * F + G) / np.power(r, 2.0))
	b = np.power(F, 4.0 / 3.0)

	"""
	The parameter x := a / b. However, numerical difficulty is encountered as b -> 0.
	Therefore, x = a * b / (b^2 + delta) is used. 
	"""

	x = a * b / (np.power(b, 2.0) + delta)

	beta = 0.05

	# x[r >= 10.0] = 0.0

	vexc = - beta * np.power(F / 2.0, 1.0 / 3.0) * np.power(x, 2.0) / (1.0 + 3.0 * beta * x * np.arcsinh(x))

	# popt, pcov = sop.curve_fit(expFit, r[np.logical_and(9 < r, r < 10)], vexc[np.logical_and(9 < r, r < 10)])

	# vexc[r > 9] = expFit(r[r > 9], * popt)

	return vexc

def getCorrelationPotential (density, functional="LDA"):

	"""
	The correlation potential is calculated from an approximation derived from second-order
	Moller-Plesset perturbation theory for the uniform electron gas. For more information,
	see the following:

	T. Chachiyo. J. Chem. Phys. 145, 021101 (2016)
	doi: https://doi.org/10.1063/1.4958669
	"""

	n = density

	a = (np.log(2.0) - 1.0) / (2.0 * np.power(np.pi, 2.0))
	b = 20.4562257

	vcor = (a * np.log(1.0 + b * np.power(4.0 * np.pi * n / 3.0, 1.0 / 3.0) + b * np.power(4.0 * np.pi * n / 3.0, 2.0 / 3.0)) \
			+ a * b * np.power(2.0 * np.pi, 1.0 / 3.0) * (np.power(6.0, 1.0 / 3.0) \
			+ 4.0 * np.power(np.pi * n, 1.0 / 3.0)) * np.power(n, 1.0 / 3.0) \
				/ (np.power(3.0, 2.0 / 3.0) * (3.0 + np.power(6.0, 2.0 / 3.0) * b * np.power(np.pi * n, 1.0 / 3.0) \
					+ 2.0 * np.power(6.0, 1.0 / 3.0) * np.power(np.pi * n, 2.0 / 3.0))))

	return vcor
