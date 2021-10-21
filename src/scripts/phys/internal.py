import numpy as np
import scipy.special as sps
import sys as sys
import scipy as sp

def getIp(targetID):
	if targetID == 0:
		return 0.5
	elif targetID == 3:
		return 15.8 / 27.2114
	elif (targetID == 5):
		return 12.8 / 27.2114
	else:
		return 0.5

def getKSOL (targetID):
	if (targetID == 0):
		return np.array([0])
	elif (targetID == 1):
		return np.array([0])
	elif (targetID == 2):
		return np.array([1, 1])
	elif (targetID == 3):
		return np.array([0, 1, 1])
	elif (targetID == 4):
		return np.array([0, 1, 1])
	elif (targetID == 5):
		return np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])

def getKSOM (targetID):
	if (targetID == 0):
		return np.array([0])
	elif (targetID == 1):
		return np.array([0])
	elif (targetID == 2):
		return np.array([0, 1])
	elif (targetID == 3):
		return np.array([0, 0, 1])
	elif (targetID == 4):
		return np.array([0, 0, 0, 1, 0, 1, 0, 1, 2])
	elif (targetID == 5):
		return np.array([0, 0, 0, 1, 0, 1, 0, 1, 2])

def moving_average(x, w):

	idx = np.linspace(- np.pi, np.pi, w)
	fil = np.power(np.cos(idx), 2.0)

	return np.convolve(x, fil, 'same') / np.sum(fil)

def f3lm (l, m):
	if (l >= m):
		return 2.50 / (2.0 * l + 3.0) / (2.0 * l + 5.0) * np.sqrt((np.power(l + 1.0, 2.0) - np.power(m, 2.0)) * (np.power(l + 2.0, 2.0) - np.power(m, 2.0)) * (np.power(l + 3.0, 2.0) - np.power(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 7.0))
	else:
		return 0

def d2lm (l, m):
	return 1.50 / (2.0 * l + 3.0) * np.sqrt( (np.power(l + 1.0, 2.0) - np.power(m, 2.0)) * (np.power(l + 2.0, 2.0) - np.power(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 5.0))

def p3lm (l, m):
	if (l >= m):
		return 1.50 * (l * (l + 2.0) - 5.0 * np.power(m, 2.0)) / (2.0 * l - 1.0) / (2.0 * l + 5.0) * np.sqrt((np.power(l + 1.0, 2.0) - np.power(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 3.0))
	else:
		return 0.0

def p1lm (l, m):
	return np.sqrt((np.power(l + 1, 2.0) - np.power(m, 2.0)) / (2.0 * l + 1.0) / (2.0 * l + 3.0))

def s2lm (l, m):
	return (l * (l + 1.0) - 3.0 * np.power(m, 2.0)) / (2.0 * l - 1.0) / (2.0 * l + 3.0)

def q(l, m):

	if ((l >= 0) and (m <= l)):

		return 1.50 / (2.0 * l + 3.0) * np.sqrt((np.power(l + 1, 2.0) - np.power(m, 2.0)) * (np.power(l + 2, 2.0) - np.power(m, 2.0)) / ((2.0 * l + 1.0) * (2.0 * l + 5.0)))

	else:

		return 0

def c(l, m):
	if (m <= l and l >= 0):
		return np.sqrt((np.power(l + 1, 2.0) - np.power(m, 2.0)) / ((2 * l + 1) * (2 * l + 3)))
	else:
		return 0

def propagatorsLFP (N_l, M):
	U = np.zeros((N_l, N_l))
	for ldx in range(N_l - 1):
		U[ldx, ldx + 1] = p3lm(ldx, M)
		U[ldx + 1, ldx] = p3lm(ldx, M)
	return U

def propagatorsLFF (N_l, M):
	U = np.zeros((N_l, N_l))
	for ldx in range(N_l - 3):
		U[ldx, ldx + 3] = f3lm(ldx, M)
		U[ldx + 3, ldx] = f3lm(ldx, M)
	return U

def propagatorsLD (N_l, M):

	U = np.zeros((N_l, N_l))

	for ldx in range(N_l - 2):
		U[ldx, ldx + 2] = q(ldx, M)
		U[ldx + 2, ldx] = q(ldx, M)

	return U

def propagatorsLP (N_l, M):

	U = np.zeros((N_l, N_l))

	for ldx in range(N_l - 1):
		U[ldx, ldx + 1] = c(ldx, M)
		U[ldx + 1, ldx] = c(ldx, M)

	return U

def propagatorsL (m_active, N_l, LDX):

	for mdx in range(np.max(m_active) + 1):

		if (LDX == 1):

			# Dipole

			L  = np.zeros((np.max(m_active) + 1, N_l, N_l))
			LD = np.zeros((np.max(m_active) + 1, N_l))
			LV = np.zeros((np.max(m_active) + 1, N_l, N_l))

			L[mdx, :, :] = propagatorsLP (N_l, mdx)

			eigenD, eigenV = np.linalg.eigh(L[mdx, mdx : N_l, mdx : N_l])

			LD[mdx, mdx : N_l] = np.copy(eigenD)
			LV[mdx, mdx : N_l, mdx : N_l] = np.copy(eigenV)

			temp_tf = LV[mdx,:, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/001/V" + str(mdx).zfill(3) + ".bin")
			temp_tf = LD[mdx, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/001/D" + str(mdx).zfill(3) + ".bin")

			# Octupole

			L  = np.zeros((np.max(m_active) + 1, N_l, N_l))
			LD = np.zeros((np.max(m_active) + 1, N_l))
			LV = np.zeros((np.max(m_active) + 1, N_l, N_l))

			L[mdx, :, :] = propagatorsLFP (N_l, mdx)

			eigenD, eigenV = np.linalg.eigh(L[mdx, mdx : N_l, mdx : N_l])

			LD[mdx, mdx : N_l] = np.copy(eigenD)
			LV[mdx, mdx : N_l, mdx : N_l] = np.copy(eigenV)

			temp_tf = LV[mdx,:, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/003/V" + str(mdx).zfill(3) + ".bin")
			temp_tf = LD[mdx, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/003/D" + str(mdx).zfill(3) + ".bin")

		elif (LDX == 2):

			L  = np.zeros((np.max(m_active) + 1, N_l, N_l))
			LD = np.zeros((np.max(m_active) + 1, N_l))
			LV = np.zeros((np.max(m_active) + 1, N_l, N_l))

			L[mdx, :, :] = propagatorsLD (N_l, mdx)

			eigenD, eigenV = np.linalg.eigh(L[mdx, mdx : N_l, mdx : N_l])

			LD[mdx, mdx : N_l] = np.copy(eigenD)
			LV[mdx, mdx : N_l, mdx : N_l] = np.copy(eigenV)

			temp_tf = LV[mdx,:, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/002/V" + str(mdx).zfill(3) + ".bin")
			temp_tf = LD[mdx, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/002/D" + str(mdx).zfill(3) + ".bin")

		elif (LDX == 3):

			L  = np.zeros((np.max(m_active) + 1, N_l, N_l))
			LD = np.zeros((np.max(m_active) + 1, N_l))
			LV = np.zeros((np.max(m_active) + 1, N_l, N_l))

			L[mdx, :, :] = propagatorsLFF (N_l, mdx)

			eigenD, eigenV = np.linalg.eigh(L[mdx, mdx : N_l, mdx : N_l])

			LD[mdx, mdx : N_l] = np.copy(eigenD)
			LV[mdx, mdx : N_l, mdx : N_l] = np.copy(eigenV)

			temp_tf = LV[mdx,:, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/003/V" + str(mdx).zfill(3) + ".bin")
			temp_tf = LD[mdx, :].astype(np.float64)
			temp_tf.tofile("./workingData/hamiltonian/propagatorsL/" + str(LDX).zfill(3) + "/003/D" + str(mdx).zfill(3) + ".bin")


def printSimEstimate (E0, omega0, eigenValues, electronVoltAU, R_MAX, R_MASK, N_l):

	cutOff = (3.17 * np.power(E0 / 2.0 / omega0, 2.0) + np.absolute(np.max(eigenValues)))
	cutOffOrder = int(cutOff / omega0)
	print("\n     - Cutoff = %.2lf eV" % (electronVoltAU * cutOff))
	print("     - Cutoff Order = %d " % (cutOffOrder))
	print("     - Max. Radius  = %2lf a.u. " % (R_MAX))
	print("     - ABC. Radius  = %2lf a.u. " % (R_MASK))
	print("     - Est. Radius  = %2lf a.u. " % (E0 / np.power(omega0, 2.0)))


	if (cutOffOrder > N_l):
		print("")
		print("     ** Warning ** ")
		print("")
		print("     The number of angular momenta N_l < estimated cutoff harmonic order. Roughly, each")
		print("     each angular momentum l corresponds to the electron absorbing one photon and N_l")
		print("     should be larger than the estimated cutoff harmonic order for convergence.")
		print("")

def smootherLR (r, VMAX):

	A = np.exp(1) * VMAX
	sig = 1.0 / np.sqrt(VMAX)

	vc = np.power(r, - 2.0)

	vc[vc > VMAX] = A * np.exp(- np.power(r[vc > VMAX] / sig, 2.0))

	return vc

def propagatorsR (r, T, v0, vscf, maskR, l_active, N_r, N_l, dt, EXP_METHOD, R_MASK):

	print("     - Generating radial propagation matrices:")
	print("")

	CN  = 0
	CN2 = 1
	CN3 = 2
	EIG = 3

	MAX_L0 = 256
	MAX_R2 = 100000

	PROPAGATORS = np.zeros((N_r, N_r, N_l), dtype=complex)
	VSCF        = np.diag(vscf.astype(complex))

	for l in range(N_l):

		printProgressBar(l + 1, N_l)

		if (l <= MAX_L0):
			v = v0 + l * (l + 1) / 2.0 / np.power(r, 2.0) - 1j * maskR
		else:
			v = v0 - 1j * maskR

		V = np.diag(v) + VSCF

		hamiltonian1 = - 0.50 * T + V
		hamiltonian1 = hamiltonian1.astype(np.complex)

		hamiltonian2 = hamiltonian1.astype(np.complex)

		hamiltonian2 = - 0.50 * T + V
		hamiltonian2 = hamiltonian1.astype(np.complex)


		if (EXP_METHOD == CN):

			Wm  = np.eye(N_r) \
				- 0.5j * dt * hamiltonian1

			Wp  = np.eye(N_r) \
				+ 0.5j * dt * hamiltonian2

		elif (EXP_METHOD == CN2):

			hamiltonian2 = np.linalg.matrix_power(hamiltonian1, 2)

			Wm  = np.eye(N_r) \
				- 0.5j * dt * hamiltonian1 \
				- np.power(dt, 2.0) / 8.0 * hamiltonian2

			Wp  = np.eye(N_r) \
				+ 0.5j * dt * hamiltonian1 \
				- np.power(dt, 2.0) / 8.0 * hamiltonian2

		elif (EXP_METHOD == CN3):

			hamiltonian2 = np.linalg.matrix_power(hamiltonian1, 2)
			hamiltonian3 = np.matmul(hamiltonian1, hamiltonian2)

			Wm  = np.eye(N_r) \
				- 0.5j * dt * hamiltonian1 \
				- np.power(dt, 2.0) / 8.0 * hamiltonian2 \
				+ 1j / 48.0 * np.power(dt, 3.0) * hamiltonian3

			Wp  = np.eye(N_r) \
				+ 0.5j * dt * hamiltonian1 \
				- np.power(dt, 2.0) / 8.0 * hamiltonian2 \
				- 1j / 48.0 * np.power(dt, 3.0) * hamiltonian3

		elif (EXP_METHOD == EIG):

			eigenD, eigenV = np.linalg.eig(hamiltonian1)
			Wm = np.matmul(eigenV, np.matmul(np.diag(np.exp(- 0.5j * dt * eigenD)), np.linalg.inv(eigenV)))

			eigenD, eigenV = np.linalg.eig(hamiltonian2)
			Wp = np.matmul(eigenV, np.matmul(np.diag(np.exp(  0.5j * dt * eigenD)), (np.linalg.inv(eigenV))))

		else:

			Wm = sp.linalg.expm(- 0.5j * dt * hamiltonian1)
			Wp = sp.linalg.expm(  0.5j * dt * hamiltonian1)

		Wp = np.linalg.inv(Wp)

		PROP = np.dot(Wp, Wm)

		if (l > MAX_L0):
			vc = l * (l + 1) / 2.0 * smootherLR (r, MAX_R2 / l)
			PROPAGATORS[:, :, l] = np.dot(np.diag(np.exp(- 0.5j * dt * vc)), np.dot(PROP, np.diag(np.exp(- 0.5j * dt * vc))))
		else:
			PROPAGATORS[:, :, l] = PROP

	PROPAGATOR_FILE_PATHS = []

	for l in range(N_l):
		PROPAGATOR_FILE_PATHS.append("./workingData/hamiltonian/propagators/propagatorL" + str(l).zfill(3) + ".bin")
		TEMP = PROPAGATORS[:, :, l].astype(np.complex128)
		TEMP.tofile(PROPAGATOR_FILE_PATHS[l])

	return PROPAGATOR_FILE_PATHS

def rollingAverage(f, window):

	N_f = np.size(f)

	favg = np.zeros_like(f)

	for k in range(N_f):
		mindx = np.max(np.array([0, k - window]))
		maxdx = np.min(np.array([N_f - 1, k + window]))

		favg[k] = np.sum(f[mindx : maxdx]) / (maxdx - mindx)
	return favg

def dfdx (f, dx=1):

	df = np.zeros_like(f)

	for k in range(2, np.size(f) - 2):
		df[k] = - 1.0 / 12.0 * f[k + 2] \
				+ 2.0 / 3.0 * f[k + 1] \
				- 2.0 / 3.0 * f[k - 1] \
				+ 1.0 / 12.0 * f[k - 2]
	df[-1] = df[-2]
	df[0] = df[1]
	return df / dx

def d2fdx2 (f, dx):

	c04 = - 1.0 / 560.0
	c03 = 8.0 / 315.0
	c02 = - 0.20
	c01 = 8.0 / 5.0
	c00 = - 205.0 / 72.0

	N_f = np.size(f)
	df  = np.zeros_like(f)

	for k in range(4, N_f - 4):
		df[k] = c04 * (f[k + 4] + f[k - 4]) \
			  + c03 * (f[k + 3] + f[k - 3]) \
			  + c02 * (f[k + 2] + f[k - 2]) \
			  + c01 * (f[k + 1] + f[k - 1]) \
			  + c00 *  f[k]

	return df / np.power(dx, 2.0)

def saveEigenValues (targetID, eigenValues):
	if (targetID == 5):
		ev = np.zeros(6)
		ev[0] = eigenValues[4].real
		ev[1] = eigenValues[8].real
		ev[2] = eigenValues[8].real
		ev[3] = eigenValues[10].real
		ev[4] = eigenValues[10].real
		ev[5] = eigenValues[10].real
		np.save("./workingData/eigenStates/eigenValues.npy", ev)

def loadEigenValues ():

	ev = np.load("./workingData/eigenStates/eigenValues.npy")

	return ev

def filterIPforFFT (targetID):

	if targetID == 5:
		return np.array([19.3, 9.27, 9.27, 65.1, 65.1, 65.1])

def plotLabels (targetID):

	if targetID == 0:
		return [r'$n = 1, l = 0, m = 0$']
	elif targetID == 1:
		return ["n1l0m0"]
	elif targetID == 2:
		return ["n2l0m0", "n2l1m0", "n2l1m1"]
	elif targetID == 3:
		return ["n3l0m0", "n3l1m0", "n3l1m1"]
	elif targetID == 4:
		return ["n4l0m0", "n4l1m0", "n4l1m1", "n3l2m0", "n3l2m1", "n3l2m2"]
	elif targetID == 5:
		return [r'$n = 5, l = 0, m = 0$',\
				r'$n = 5, l = 1, m = 0$',\
				r'$n = 5, l = 1, m = 1$',\
				r'$n = 4, l = 2, m = 0$',\
				r'$n = 4, l = 2, m = 1$',\
				r'$n = 4, l = 2, m = 2$']
	else:
		return ["c60"]

def getMaxR (R_MASK, kmin=0.20):

	c 	 = 2.62206

	delta = 0.2

	R_MAX = c / (2.0 * delta * kmin) + R_MASK

	R_MAX = 2.0 * R_MASK

	return R_MAX

def getMaxRKmin (R_MASK):

	c 	 = 2.62206

	delta = 0.2

	kmin = c / (delta * R_MASK)

	return kmin

def getABCR (r, R_MASK, R_MAX, ABC_TYPE):

	N_r   = np.size(r)
	W     = np.zeros(N_r, dtype=np.complex)
	minDX = np.argmin(np.absolute(r - R_MASK))
	rabc  = r[minDX : np.size(r)] - R_MASK

	if ABC_TYPE == 0:

		alpha = 1.21E-5

		W[minDX : N_r] = alpha * np.power(rabc, 2.0)

	elif ABC_TYPE == 1:

		delta = 0.20
		alpha = 1.21E-5

		c  	  = 2.62206
		a     = (1.0 - 16.0 / np.power(c, 3.0))
		b     = (1.0 - 17.0 / np.power(c, 3.0)) / np.power(c, 2.0)

		kmin  = c / (delta * 2.0 * (R_MAX - R_MASK))
		EMIN  = 0.50 * np.power(kmin, 2.0)

		x 	  = 2.0 * delta * kmin * rabc
		y 	  = a * x - b * np.power(x, 3.0) + 4.0 / np.power(c - x, 2.0) - 4.0 / np.power(c + x, 2.0)
		alpha = 1.21E-3
		W[minDX : np.size(r)] = EMIN * y * rabc

	elif ABC_TYPE == 2:

		alpha12 = 0.298 * np.exp(0.104j)
		alpha22 = 0.71 * np.exp(- 0.0906j)
		beta 	= 1.97

		rabc = 1.0001 * r[-1] - r

		W = - 1j * alpha12 / (2.0 * np.sinh(rabc / (2.0 * beta))) + alpha22 / np.power(2.0 * np.sinh(rabc / (2.0 * beta)), 2.0)

	elif ABC_TYPE == 3:

		alpha1 = 1.07
		alpha2 = 8.37
		beta   = 1.79

		rabc = 1.001 * r[-1] - r

		W = - 1j * alpha1 * np.exp(- rabc / (2.0 * beta)) + alpha2 * np.exp(- rabc / beta)

	return W

def getABCMask (r, R_MASK, ETA_MASK, power=2.0):
	return ETA_MASK * np.heaviside(r - R_MASK, 1.0) * np.power(r - R_MASK, power)

def getABCMaskL (r, R_MASK, ETA_MASK, power=2.0):
	return (0.10 * ETA_MASK * np.heaviside(r - 0.95 * R_MASK, 1.0) * np.power(r - R_MASK, power))[::-1]

def getMaskT (t, factor=4):

	N_t = np.size(t)

	if (N_t < 32):
		return np.zeros(N_t)
	else:
		maskT 		= np.ones(N_t)
		maskT_width = N_t // factor
		tn 			= t[maskT_width]
		tp 			= t[maskT_width * (factor - 1)]

		maskT[0 : maskT_width] = np.power(np.cos(0.5 * np.pi * (t[0 : maskT_width] - tn) / (t[0] - tn)), 2.0)
		maskT[maskT_width * (factor - 1) : N_t] = np.power(np.cos(0.5 * np.pi * (tp - t[maskT_width * (factor - 1) : N_t]) / (t[-1] - tp)), 2.0)
		maskT[0] = 0.0
		maskT[-1] = 0.0

	return maskT

def generateGaussLobattoGrid(N_r, R_MASK, R_max, zeta):

	x = np.genfromtxt("./workingData/mesh/legendreRoots/roots" + str(N_r).zfill(5) + ".txt", delimiter=",")

	N_g  = N_r + 1

	r    = R_max * zeta / 2.0 * (x + 1.0) / (1.0 - x + zeta)

	drdx = R_max * zeta / 2.0 * (2.0 + zeta) / np.power(1.0 - x + zeta, 2.0)

	d = np.zeros((N_r, N_r))

	T = np.zeros((N_r, N_r), dtype=np.complex)

	D1 = np.zeros((N_r, N_r))

	for i in range(N_r):
		for j in range(N_r):

			if (i == j):
				d[i, j] = - N_g * (N_g + 1.0) / (3.0 * (1.0 - np.power(x[i], 2.0)))
			else:
				d[i, j] = - 2.0 / np.power(x[i] - x[j], 2.0)

			T[i, j] = d[i, j] / (drdx[i] * drdx[j])

			if (i != j):
				D1[i, j] = 1.0 / (x[i] - x[j]) / np.sqrt(drdx[i] * drdx[j])


	P = sps.eval_legendre(N_g, x)

	return r, drdx, d, T, N_g, x, P, D1

def generateGaussLobattoGrid2 (N_r, R_max, zeta):

	x = np.genfromtxt("./workingData/mesh/legendreRoots/roots" + str(N_r).zfill(5) + ".txt", delimiter=",")

	N_g  = N_r + 1

	r    = R_max * np.power(zeta / 2.0 * (x + 1.0) / (1.0 - x + zeta), 2.0)

	# drdx = R_max * zeta / 2.0 * (2.0 + zeta) / np.power(1.0 - x + zeta, 2.0)
	drdx = - (R_max * (1.0 + x) * np.power(zeta, 2.0) * (2.0 + zeta)) / (2.0 * np.power(- 1.0 + x - zeta, 3.0))

	d = np.zeros((N_r, N_r))

	T = np.zeros((N_r, N_r))

	D1 = np.zeros((N_r, N_r))

	for i in range(N_r):
		for j in range(N_r):

			if (i == j):
				d[i, j] = - N_g * (N_g + 1.0) / (3.0 * (1.0 - np.power(x[i], 2.0)))
			else:
				d[i, j] = - 2.0 / np.power(x[i] - x[j], 2.0)

			T[i, j] = d[i, j] / (drdx[i] * drdx[j])

			if (i != j):
				D1[i, j] = 1.0 / (x[i] - x[j]) / np.sqrt(drdx[i] * drdx[j])


	P = sps.eval_legendre(N_g, x)

	return r, drdx, d, T, N_g, x, P, D1

def printProgressBar (iteration, total, prefix = '    ', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix}  {bar}  {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("")

def welcomeScreen():

	print("")
	print(color.BOLD + "  GGB TD-DFT ASS" + color.END)
	print(color.BOLD + "  Time-Dependent Density Functional Theory for Azimuthally Symmetric Systems" + color.END)
	print("")
	print("  A TD-DFT program to calculate strong-field electron")
	print("  dynamics in azimuthally symmetric systems.")
	print("")

def printTargetParameters(targetString, Z, n, l, eConfig):

	print("")
	print("  1. System Description")
	print("")
	print("     - Atom   : " + targetString)
	print("     - Charge : " + str(int(Z)))
	print("")
	print("     - Electron Configuration: ")
	print("       " + eConfig)
	print("")

def printFigureUpdate():

	print("")
	print("  5. Generating Figures")
	print("")

def printGSUpdate():

	print("")
	print("  2. Obtaining Ground State and Calculation Parameters")
	print("")

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def getTargetLabel (targetID):

	if (targetID == 0):
		return "hydrogen"
	elif (targetID == 1):
		return "helium"
	elif (targetID == 2):
		return "neon"
	elif (targetID == 3):
		return "argon"
	elif (targetID == 4):
		return "krypton"
	else:
		return "xenon"

def getTargetID (target):

	target = target.lower()

	if ((target =="h") or (target == "hydrogen")):
		return 0
	elif ((target == "he") or (target == "helium")):
		return 1
	elif ((target == "ne") or (target == "neon")):
		return 2
	elif ((target == "ar") or (target == "argon")):
		return 3
	elif ((target == "kr") or (target == "krypton")):
		return 4
	elif ((target == "xe") or (target == "xenon")):
		return 5
	elif ((target == "c60") or (target == "carbon60")):
		return 10
	else:
		return - 1

def translateL (l):
	if (l == 0):
		return "s"
	elif (l == 1):
		return "p"
	elif (l == 2):
		return "d"
	elif (l == 3):
		return "f"
	elif (l == 4):
		return "g"
	elif (l == 5):
		return "h"
	elif (l == 6):
		return "i"
	elif (l == 7):
		return "k"
	elif (l == 8):
		return "l"
	elif (l == 9):
		return "m"
	elif (l == 10):
		return "n"
	elif (l == 11):
		return "o"
	elif (l == 12):
		return "q"
	elif (l == 13):
		return "r"
	elif (l == 14):
		return "t"
	elif (l == 15):
		return "u"
	else:
		return "v"

def getElectronConfiguration (targetID, targetString):
	print(targetID)
	if (targetID == 0):

		Z        = 1.0
		stringEC = "1s1"
		n 	     = np.array([1])
		l        = np.array([0])
		m 		 = np.array([0])
		occ      = np.array([1])
		N_kso    = np.size(n)

	elif (targetID == 1):

		Z        = 2.0
		stringEC = "1s2"
		n 	     = np.array([1])
		l        = np.array([0])
		m 		 = np.array([0])
		occ      = np.array([2])
		N_kso    = np.size(n)

	elif (targetID == 2):

		Z        = 10.0
		stringEC = "1s2, 2s2, 2p6"
		n 	     = np.array([1, 2, 2, 2])
		l        = np.array([0, 0, 1, 1])
		m 		 = np.array([0, 0, 0, 1])
		occ      = np.array([2, 2, 2, 4])
		N_kso    = np.size(n)

	elif (targetID == 3):

		Z        = 18.0
		stringEC = "1s2, 2s2, 2p6, 3s2, 3p6"
		n 	     = np.array([1, 2, 2, 2, 3, 3, 3])
		l        = np.array([0, 0, 1, 1, 0, 1, 1])
		m 		 = np.array([0, 0, 0, 1, 0, 0, 1])
		occ      = np.array([2, 2, 2, 4, 2, 2, 4])
		N_kso    = np.size(n)

	elif (targetID == 4):

		Z        = 36.0
		stringEC = "1s2, 2s2, 2p6, 3s2, 3p6, 3d10, 4s2, 4p6"
		n 	     = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4])
		l        = np.array([0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 0, 1, 1])
		m 		 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1])
		occ      = np.array([2, 2, 2, 4, 2, 2, 4, 2, 4, 4, 2, 2, 4])
		N_kso    = np.size(n)

	elif (targetID == 5):

		Z        = 54.0
		stringEC = "1s2, 2s2, 2p6, 3s2, 3p6, 3d10, 4s2, 4p6, 4d10, 5s2, 5p6"
		n 	     = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5])
		l        = np.array([0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 1])
		m 		 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 0, 1])
		occ      = np.array([2, 2, 2, 4, 2, 2, 4, 2, 4, 4, 2, 2, 4, 2, 4, 4, 2, 2, 4])
		N_kso    = np.size(n)

	else:
		print("Please choose a valid target.")
		sys.exit()

	printTargetParameters (targetString, Z, n, l, stringEC)

	return Z, n, l, m, occ, N_kso, stringEC

def getSphericalElectronConfiguration (targetID):

	if (targetID == 0):

		Z        = 0.97015
		n 	     = np.array([1])
		l        = np.array([0])
		occ      = np.array([1])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([1])

		return Z, n, l, occ, N_kso, NUM_L, active

	elif (targetID == 1):

		Z        = 2.0
		n 	     = np.array([1])
		l        = np.array([0])
		occ      = np.array([2])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([1])

		return Z, n, l, occ, N_kso, NUM_L, active

	elif (targetID == 2):

		Z        = 10.0
		n 	     = np.array([1, 2, 2])
		l        = np.array([0, 0, 1])
		occ      = np.array([2, 2, 6])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([0, 1, 1])

		return Z, n, l, occ, N_kso, NUM_L, active

	elif (targetID == 3):

		Z        = 18.0
		n 	     = np.array([1, 2, 3, 2, 3])
		l        = np.array([0, 0, 0, 1, 1])
		occ      = np.array([2, 2, 2, 6, 6])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([0, 0, 1, 0, 1])

		return Z, n, l, occ, N_kso, NUM_L, active

	elif (targetID == 4):

		Z        = 36.0
		n 	     = np.array([1, 2, 3, 4, 2, 3, 4, 3 ])
		l        = np.array([0, 0, 0, 0, 1, 1, 1, 2 ])
		occ      = np.array([2, 2, 2, 2, 6, 6, 6, 10])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([0, 0, 0, 1, 0, 0, 1, 0])

		return Z, n, l, occ, N_kso, NUM_L, active

	elif (targetID == 5):

		Z        = 54.0
		n 	     = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 3 , 4 ])
		l        = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2 , 2 ])
		occ      = np.array([2, 2, 2, 2, 2, 6, 6, 6, 6, 10, 10])
		N_kso    = np.size(n)
		NUM_L 	 = np.size(np.unique(l))
		active   = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0 , 1])

		return Z, n, l, occ, N_kso, NUM_L, active

	else:
		print("Please choose a valid target.")
		sys.exit()

def outputSCFInit (n_sph, l_sph, occ_sph, N_kso_sph):
	print("     - Beginning self-consistent field calculation\n")
	if (N_kso_sph > 1):
		print("     ┌────┬──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("───────────", end = '')
			else:
				print("──────────┬", end = '')
	else:
		print("     ┌────┬─────────┬", end = '')

	print("─────────────┐")

	if (N_kso_sph > 1):
		print("     │Step│Energies (eV)", end = '')
		for k in range(1, N_kso_sph):
			if (k == 1):
				print("        ", end = '')
			elif (k < N_kso_sph - 1):
				print("           ", end = '')
			else:
				print("          │", end = '')
	else:
		print("     │Step│Energies │", end = '')

	print("             │")

	if (N_kso_sph > 1):
		print("     ├────┼──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┬──────────", end = '')
			else:
				print("┬─────────┼", end = '')
	else:
		print("     ├────┼─────────┼", end = '')
	print("─────────────┤")

	print("     │    ", end = '')

	if (N_kso_sph > 1):
		for k in range(0, N_kso_sph):
			if (k < N_kso_sph - 1):
				if (occ_sph[k] < 10):
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "    ", end='')
				else:
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   ", end='')
			else:
				if (occ_sph[k] < 10):
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   │", end='')
				else:
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "  │", end='')
	else:
		print("│   " + str(n_sph[0]) + translateL(l_sph[0]) + str(occ_sph[0]) + "   │", end='')

	print(" Convergence │")

	if (N_kso_sph > 1):
		print("     ├────┼──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┼──────────", end = '')
			else:
				print("┼─────────┼", end = '')
	else:
		print("     ├────┼─────────┼", end = '')

	print("─────────────┤")

def formatFloat(x):
	if (np.absolute(x) < 10.0):
		return "    {:1.2f}".format(x)
	elif (np.absolute(x) < 100.0):
		return "   {:2.2f}".format(x)
	elif (np.absolute(x) < 1000.0):
		return "  {:3.2f}".format(x)
	elif (np.absolute(x) < 10000.0):
		return " {:4.2f}".format(x)
	elif (np.absolute(x) < 100000.0):
		return "{:5.2f}".format(x)
	elif (np.absolute(x) < 1000000.0):
		return "{:6.1f}".format(x)
	else:
		print("\n")
		print(x)
		return "FFF"

def outputSCFResultsIteration (n_sph, l_sph, energy, convergence, N_kso_sph, scfdx):
	energyScale = 27.211386245988
	if (N_kso_sph > 1):
		print("     │%04d" % scfdx, end='')
		for ksodx in range(N_kso_sph):
			if (N_kso_sph != 1):
				if (ksodx == N_kso_sph - 1):
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "│", end = '')
				elif (ksodx == 0):
					print("│" + formatFloat(energy[ksodx].real * energyScale), end = '')
				else:
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "", end = '')
			else:
				print("│%+.2E" % (energy[ksodx].real * energyScale), end = '')
		print(" %.5E │" % convergence)
	else:
		print("     │%04d│%+.2E│" % (scfdx, energy[0].real * energyScale), end='')
		print(" %.5E │" % convergence)

def outputSCFFinish (n_sph, l_sph, energy, convergence, N_kso_sph, occ_sph, scfdx):

	energyScale = 27.211386245988

	if (N_kso_sph > 1):
		print("     ├────┼──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┼──────────", end = '')
			else:
				print("┼─────────┼", end = '')
	else:
		print("     ├────┼─────────┼", end = '')

	print("─────────────┤")
	print("     │Iter", end = '')
	if (N_kso_sph > 1):
		for k in range(0, N_kso_sph):
			if (k < N_kso_sph - 1):
				if (occ_sph[k] < 10):
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "    ", end='')
				else:
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   ", end='')
			else:
				if (occ_sph[k] < 10):
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   │", end='')
				else:
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "  │", end='')
	else:
		print("│   " + str(n_sph[0]) + translateL(l_sph[0]) + str(occ_sph[0]) + "   │", end='')

	print(" Convergence │")

	if (N_kso_sph > 1):
		print("     ├────┼──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┼──────────", end = '')
			else:
				print("┼─────────┼", end = '')
	else:
		print("     ├────┼─────────┼", end = '')

	print("─────────────┤")

	if (N_kso_sph > 1):
		print("     │%04d" % scfdx, end='')
		for ksodx in range(N_kso_sph):
			if (N_kso_sph != 1):
				if (ksodx == N_kso_sph - 1):
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "│", end = '')
				elif (ksodx == 0):
					print("│" + formatFloat(energy[ksodx].real * energyScale), end = '')
				else:
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "", end = '')
			else:
				print("│%+.2E" % (energy[ksodx].real * energyScale), end = '')
		print(" %.5E │" % convergence)
	else:
		print("     │%04d│%+.2E│" % (scfdx, energy[0].real * energyScale), end='')
		print(" %.5E │" % convergence)

	if (N_kso_sph > 1):
		print("     └────┴──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┴──────────", end = '')
			else:
				print("┴─────────┴", end = '')
	else:
		print("     └────┴─────────┴", end = '')
	print("─────────────┘")

def outputSCFLoad (n_sph, l_sph, energy, convergence, N_kso_sph, occ_sph, scfdx):

	energyScale = 27.211386245988

	if (N_kso_sph > 1):
		print("     ┌────┬──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("───────────", end = '')
			else:
				print("──────────┬", end = '')
	else:
		print("     ┌────┬─────────┬", end = '')

	print("─────────────┐")

	print("     │    ", end = '')
	if (N_kso_sph > 1):
		for k in range(0, N_kso_sph):
			if (k < N_kso_sph - 1):
				if (occ_sph[k] < 10):
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "    ", end='')
				else:
					print("│   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   ", end='')
			else:
				if (occ_sph[k] < 10):
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "   │", end='')
				else:
					print("|   " + str(n_sph[k]) + translateL(l_sph[k]) + str(occ_sph[k]) + "  │", end='')
	else:
		print("│   " + str(n_sph[0]) + translateL(l_sph[0]) + str(occ_sph[0]) + "   │", end='')

	print(" Convergence │")

	if (N_kso_sph > 1):
		print("     ├────┼──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┼──────────", end = '')
			else:
				print("┼─────────┼", end = '')
	else:
		print("     ├────┼─────────┼", end = '')

	print("─────────────┤")

	if (N_kso_sph > 1):
		print("     │    ", end='')
		for ksodx in range(N_kso_sph):
			if (N_kso_sph != 1):
				if (ksodx == N_kso_sph - 1):
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "│", end = '')
				elif (ksodx == 0):
					print("│" + formatFloat(energy[ksodx].real * energyScale), end = '')
				else:
					print(" │" + formatFloat(energy[ksodx].real * energyScale) + "", end = '')
			else:
				print("│%+.2E" % (energy[ksodx].real * energyScale), end = '')
		print(" %.5E │" % convergence)
	else:
		print("     │%04d│%+.2E│" % (scfdx, energy[0].real * 27.2113), end='')
		print(" %.5E │" % convergence)

	if (N_kso_sph > 1):
		print("     └────┴──────────", end = '')
		for k in range(1, N_kso_sph):
			if (k < N_kso_sph - 1):
				print("┴──────────", end = '')
			else:
				print("┴─────────┴", end = '')
	else:
		print("     └────┴─────────┴", end = '')
	print("─────────────┘")

def estimateCutOff (E0, omega0, targetID):

	if targetID == 0:
		Ip = 13.6
	elif targetID == 1:
		Ip = 24.59
	elif targetID == 2:
		Ip = 21.56
	elif targetID == 3:
		Ip = 15.6
	elif targetID == 4:
		Ip = 14.0
	elif targetID == 5:
		Ip = 12.13
	else:
		Ip = 7.6

	Up = np.power(0.5 * E0 / omega0, 2.0) * 27.2114

	cutoff = 3.17 * Up + Ip
	radius = E0 / np.power(omega0, 2.0)

	print("")
	print("- Estimated Cutoff: %.2lf eV" % cutoff)
	print("- Estimated Radius: %.2lf a.u." % radius)
	print("")

def getSimulationStates (targetID):

	if targetID == 0:
		sdx = np.array([0])
		ldx = np.array([0])
		mdx = np.array([0])
		occ = np.array([1])
		N_e = 1
	elif targetID == 1:
		sdx = np.array([0])
		ldx = np.array([0])
		mdx = np.array([0])
		occ = np.array([1])
		N_e = 1
	elif targetID == 2:
		n 	     = np.array([1, 2, 2])
		sdx = np.array([1, 2, 2])
		ldx = np.array([0, 1, 1])
		mdx = np.array([0, 0, 1])
		occ = np.array([2, 2, 4])
		N_e = 3
	elif targetID == 3:
		n 	= np.array([1, 2, 3, 2, 3])
		sdx = np.array([2, 4, 4])
		ldx = np.array([0, 1, 1])
		mdx = np.array([0, 0, 1])
		occ = np.array([2, 2, 4])
		N_e = 3
	elif targetID == 4:
		n 	     = np.array([1, 2, 3, 4, 2, 3, 4, 3 ])
		l        = np.array([0, 0, 0, 0, 1, 1, 1, 2 ])
		sdx = np.array([3, 6, 6])
		ldx = np.array([0, 1, 1])
		mdx = np.array([0, 0, 1])
		occ = np.array([2, 2, 4])
		N_e = 3
	elif targetID == 5:
		sdx = np.array([3, 4, 7, 7, 8, 8, 10, 10, 10])
		ldx = np.array([0, 0, 1, 1, 1, 1,  2,  2,  2])
		mdx = np.array([0, 0, 0, 1, 0, 1,  0,  1,  2])
		occ = np.array([2, 2, 2, 4, 2, 4,  2,  4,  4])
		N_e = 9

	return sdx, ldx, mdx, occ, N_e
