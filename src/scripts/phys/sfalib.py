import numpy as np
import scipy.optimize as sop

attosecondAU   = 0.0413413745758
femtosecondAU  = 41.3413745758
cAU            = 137.035999074
intensityAU    = 1.0 / 3.509E16
electronvoltAU = 1.0 / 27.2114
ev_2_Hz  	    = 2.417989E14

def loadDirectSolutions ():

	phi1rD = np.load("./data/directChannel/phi1rD.npy")
	phi1iD = np.load("./data/directChannel/phi1iD.npy")
	phi3rD = np.load("./data/directChannel/phi3rD.npy")
	phi3iD = np.load("./data/directChannel/phi3iD.npy")
	psrD   = np.load("./data/directChannel/psrD.npy")
	xuvD   = np.load("./data/directChannel/xuvD.npy")
	idxD   = np.load("./data/directChannel/idxD.npy")

	return phi1rD, phi1iD, phi3rD, phi3iD, psrD, xuvD, idxD

def loadC0Solutions ():

	phi1rC0 = np.load("./data/correlatedChannel/trivial/phi1rC0.npy")
	phi1iC0 = np.load("./data/correlatedChannel/trivial/phi1iC0.npy")
	phi2rC0 = np.load("./data/correlatedChannel/trivial/phi2rC0.npy")
	phi2iC0 = np.load("./data/correlatedChannel/trivial/phi2iC0.npy")
	phi3rC0 = np.load("./data/correlatedChannel/trivial/phi3rC0.npy")
	phi3iC0 = np.load("./data/correlatedChannel/trivial/phi3iC0.npy")
	psrC0   = np.load("./data/correlatedChannel/trivial/psrC0.npy")
	qsrC0   = np.load("./data/correlatedChannel/trivial/qsrC0.npy")
	qsiC0   = np.load("./data/correlatedChannel/trivial/qsiC0.npy")
	dqC0    = np.load("./data/correlatedChannel/trivial/dqC0.npy")
	xuvC0   = np.load("./data/correlatedChannel/trivial/xuvC0.npy")
	idxC0   = np.load("./data/correlatedChannel/trivial/idxC0.npy")

	return phi1rC0, phi1iC0, phi2rC0, phi2iC0, phi3rC0, phi3iC0, psrC0, qsrC0, qsiC0, dqC0, xuvC0, idxC0

def saddlePointEquationsDirect (x0, phi3r, Ip, A0, omega0):

	eqs = np.zeros(5, dtype=float)

	omega0 = 1.55 / 27.2114

	psr, psi, phi1r, phi1i, phi3i = x0

	gamma = np.sqrt(2.0 * Ip / np.power(A0, 2.0))

	# Tunnel Ionization

	eqs[0] = psr - np.cosh(phi1i)*np.sin(phi1r) + np.cosh(phi3i)*np.sin(phi3r)
	eqs[1] = -(np.sqrt(2)*np.sqrt(Ip/np.power(A0,2))) + psi - np.cos(phi1r)*np.sinh(phi1i) + np.cos(phi3r)*np.sinh(phi3i)

	eqs[2] = phi1i*psi - phi3i*psi - phi1r*psr + phi3r*psr - np.cos(phi1r)*np.cosh(phi1i) + np.cosh(phi3i)*(np.cos(phi3r) + (-phi1r + phi3r)*np.sin(phi3r)) + (phi1i - phi3i)*np.cos(phi3r)*np.sinh(phi3i)
	eqs[3] = -(phi1r*psi) + phi3r*psi - phi1i*psr + phi3i*psr + (-phi1i + phi3i)*np.cosh(phi3i)*np.sin(phi3r) + np.sin(phi1r)*np.sinh(phi1i) - ((phi1r - phi3r)*np.cos(phi3r) + np.sin(phi3r))*np.sinh(phi3i)

	eqs[4] = psi

	return eqs

def solveDirectSystem (phi3r, Ip, A0, omega0):

	N     = np.size(phi3r)
	phi1r = np.zeros(N)
	phi1i = np.zeros(N)
	phi3i = np.zeros(N)
	psr   = np.zeros(N)
	psi   = np.zeros(N)
	XUV   = np.zeros(N)

	for k in range(N):

		if (k == 0):
			res = sop.fsolve (saddlePointEquationsDirect, \
							  x0=(0, 0, 0, 0, 0), \
							  args=(phi3r[k], Ip, A0, omega0))
		else:
			res = sop.fsolve (saddlePointEquationsDirect, \
							  x0=(psr[k - 1], psi[k - 1], phi1r[k - 1], phi1i[k - 1], phi3i[k - 1]), \
							  args=(phi3r[k], Ip, A0, omega0))
		psr[k]   = res[0]
		psi[k]   = res[1]
		phi1r[k] = res[2]
		phi1i[k] = res[3]
		phi3i[k] = res[4]
		XUV[k]   = (2.0 * Ip + np.power(A0, 2.0) * np.power(psr[k], 2.0)) / 2.0 / electronvoltAU

	return phi1r, phi1i, phi3i, psr, XUV

def getDirectSolutionsMOL (N_t, PHI_MIN, PHI_MAX, Ip, A0, omega0, R, alpha, beta):

	phi3r = np.linspace(0.0, 1.0, N_t)

	idxMIN = np.argmin(np.absolute(phi3r - PHI_MIN))
	idxMAX = np.argmin(np.absolute(phi3r - PHI_MAX))

	phi3r *= 2.0 * np.pi

	phi1r = np.zeros(N_t)
	phi1i = np.zeros(N_t)
	phi3i = np.zeros(N_t)
	psr   = np.zeros(N_t)
	psi   = np.zeros(N_t)
	XUV   = np.zeros(N_t)

	phi1r_s, phi1i_s, phi3i_s, psr_s, psi_s, XUV_s = solveDirectSystemMOL (phi3r[idxMIN : idxMAX], Ip, A0, omega0, R, alpha, beta)

	phi1r[idxMIN : idxMAX] = phi1r_s[:]
	phi1i[idxMIN : idxMAX] = phi1i_s[:]
	phi3i[idxMIN : idxMAX] = phi3i_s[:]
	psr[idxMIN : idxMAX]   = psr_s[:]
	psi[idxMIN : idxMAX]   = psi_s[:]
	XUV[idxMIN : idxMAX] = XUV_s[:]

	idx = np.array([idxMIN, idxMAX])

	return phi1r, phi1i, phi3r, phi3i, psr, psi, XUV, idxMIN, idxMAX

def getDirectSolutions (N_t, PHI_MIN, PHI_MAX, Ip, A0, omega0):

	phi3r = np.linspace(0.0, 1.0, N_t)

	idxMIN = np.argmin(np.absolute(phi3r - PHI_MIN))
	idxMAX = np.argmin(np.absolute(phi3r - PHI_MAX))

	phi3r *= 2.0 * np.pi

	phi1r = np.zeros(N_t)
	phi1i = np.zeros(N_t)
	phi3i = np.zeros(N_t)
	psr   = np.zeros(N_t)
	XUV   = np.zeros(N_t)

	phi1r_s, phi1i_s, phi3i_s, psr_s, XUV_s = solveDirectSystem (phi3r[idxMIN : idxMAX], Ip, A0, omega0)

	phi1r[idxMIN : idxMAX] = phi1r_s[:]
	phi1i[idxMIN : idxMAX] = phi1i_s[:]
	phi3i[idxMIN : idxMAX] = phi3i_s[:]
	psr[idxMIN : idxMAX]   = psr_s[:]
	XUV[idxMIN : idxMAX] = XUV_s[:]

	idx = np.array([idxMIN, idxMAX])

	return phi1r, phi1i, phi3r, phi3i, psr, XUV, idxMIN, idxMAX
