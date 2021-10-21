import numpy as np
from misc import internal
import plotLib as plib
import dft as dft
import sys as sys
import scipy.interpolate as sin

def getTargetString (targetID):

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
	elif (targetID == 5):
		return "xenon"

def getIonicPotential (r, Z_sph, W, USE_C60, USE_MULLER):

	if (USE_MULLER == 0 and USE_C60 == 0):

		v0 = - Z_sph / r - 1j * W

	elif (USE_C60 == 1):

		v0 = -0.6 * ((1/np.sqrt(0.2 + np.power(-6.7 + r,2))) - 1/np.sqrt(0.2 + np.power(6.7 + r,2))) - 1j * W

	else:

		v0 = getMullerPotential(r) - 1j * W

	return v0

def getSCFRamp (N_scf):

	scfdx   = np.linspace(0.0, N_scf - 1, N_scf)
	scfRamp = np.ones(N_scf)

	if (N_scf > 512):
		MAX_RAMP = 128
	else:
		MAX_RAMP = N_scf // 4

	scfRamp[scfdx < np.max(np.array([MAX_RAMP, 8]))] = np.power(np.sin(0.5 * np.pi * scfdx[scfdx < np.max(np.array([MAX_RAMP, 8]))] / (np.max(np.array([MAX_RAMP, 8])))), 2.0)

	return scfRamp

def getLidx (l_sph, NUM_L):

	l_pop = np.zeros(NUM_L, dtype=int)
	l_idx = np.zeros(NUM_L, dtype=int)

	for ldx in range(NUM_L):

		l_pop[ldx] = np.size(l_sph[l_sph==ldx])
		if (ldx == 0):
			l_idx[ldx] = 0
		else:
			l_idx[ldx] = l_idx[ldx - 1] + l_pop[ldx - 1]

	return l_pop, l_idx

def getSortedEigenSystem (H):
	"""
	Numpy's eig method produces an unsorted list of eigenValues
	and eigenVectors. This sorts them in ascending order.
	"""

	epsilon_h, C = np.linalg.eig(H)
	idx = epsilon_h.argsort()
	epsilon_h = epsilon_h[idx]
	C = C[:,idx]

	return epsilon_h, C

def normalizeEigenstates (eigenStates):
	"""
	The eigenstates, u(r), are scaled versions of the true orbitals:

	u(r) = sqrt(drdx) phi(r).

	Consequently, the radial step is already integrated into the states and
	only the sum of |u(r)|^2 is required for normalization.
	"""

	N_states = np.shape(eigenStates)[1]

	for ksodx in range(N_states):

		norm = np.sum(np.power(np.absolute(eigenStates[:, ksodx]), 2.0))

def getDensityU (states, occ):

	"""
	The spherically averaged density is used:

	n(r) = (4 * pi)^(-1) * SUM(k = 1; N_kso) g_k * |psi_k(r)|^2 / r^2,

	g_k  : occupation number (= 2 for m = 0, = 4 for m > 0)
	psi_k: Kohn-Sham orbital for index k
	"""

	density = np.zeros(np.shape(states)[0])
	for i in range(np.shape(states)[1]):
		density += occ[i] * np.power(np.absolute(states[:, i]), 2.0)
	return density / 4.0 / np.pi

def getHartreePotentialU (density, r):
	"""
	Calculates the Hartree potential by solving the integral Poisson
	equation:

	v_h = int_0^R density(r') / max(r, r') dr'

	The factor of 4 * Pi arises from the multipole expansion of the Coulomb
	interaction (1 / |r-r'|). For more information, see Jackson's
	Electrodynamics.
	"""
	N_r = np.size(r)

	VHAR = np.zeros((N_r, N_r))

	for i in range(N_r):
		for j in range(N_r):
			VHAR[i,j] = density[i] / np.max(np.array([r[i], r[j]]))

	vhar = np.sum(VHAR, axis=0)

	return 4.0 * np.pi * vhar

def getMullerPotential(r):
	"""
	The Muller potential is a pseudopotential which acccurately describes Ar for
	single-active electron calculations. It is a fitting of the potential from a
	multielectron calculation of Ar.

	For more details, see the following:

	H. G. Muller. Phys. Rev. A 60, 1341
	"""

	Zc = 1.0
	a1 = 16.039
	a2 = 2.007
	a3 = - 25.543
	a4 = 4.525
	a5 = 0.961
	a6 = 0.443

	v  = - (Zc + a1 * np.exp(- a2 * r) + a3 * r * np.exp(- a4 * r) + a5 * np.exp(- a6 * r)) / r

	A = 5.4
	B = 1.0
	C = 3.682

	v = - (1.0 + A * np.exp(- B * r)+ (17.0 - A) * np.exp(- C * r)) / r

	return v

def getGroundState (r, x, T, D1, W, P, drdx, ETA_MIX, targetID, n, l, m, occ, N_kso, N_l, KSO_MIN, FROM_SCRATCH, METHOD, USE_MULLER, USE_C60, N_scf=65536, SCF_CONVERGENCE=1E-10, FIGS=1):
	"""
	This function runs through a self-consistent field (SCF) algorithm in order to find the ground state
	of the target system.
	"""

	# Print initial formatting output
	internal.printGSUpdate()

	N_r = np.size(r)
	N_g = N_r + 1

	# For the SCF calculation, the electron-electron interaction is gradually introduced.
	# Rapid introduction of the multielectron interaction results in 'charge sloshing' and
	# an unstable calculation. This is accomplished by defining scfRamp, which multiplies the
	# e-e interaction and gradually increases from zero to one.

	scfdx   = np.linspace(0.0, N_scf - 1, N_scf)
	scfRamp = np.ones(N_scf)

	if (N_scf > 512):
		MAX_RAMP = 128
	else:
		MAX_RAMP = N_scf // 4

	scfRamp[scfdx < np.max(np.array([MAX_RAMP, 8]))] = np.power(np.sin(0.5 * np.pi * scfdx[scfdx < np.max(np.array([MAX_RAMP, 8]))] / (np.max(np.array([MAX_RAMP, 8])))), 2.0)

	# The system is solved using the one-dimensional radial Schrodinger equation. The electron
	# configuration is set here for a spherical system.

	Z_sph, n_sph, l_sph, occ_sph, N_kso_sph, NUM_L, active = internal.getSphericalElectronConfiguration (targetID)

	# This loop determines the indices for each shell and angular momentum component
	# for the indexing of the eigenstates.

	l_pop = np.zeros(NUM_L, dtype=int)
	l_idx = np.zeros(NUM_L, dtype=int)

	for ldx in range(NUM_L):

		l_pop[ldx] = np.size(l_sph[l_sph==ldx])
		if (ldx == 0):
			l_idx[ldx] = 0
		else:
			l_idx[ldx] = l_idx[ldx - 1] + l_pop[ldx - 1]

	# The eigenstates can either be calculated from scratch, or loaded from a
	# previous calculation.

	if FROM_SCRATCH == 1:

		print("     - Beginning self-consistent field calculation ")

		internal.outputSCFInit(n_sph, l_sph, occ_sph, N_kso_sph)

		H  = np.zeros((NUM_L, N_r, N_r), dtype=complex)

		v0 = getIonicPotential (r, Z_sph, W, USE_C60, USE_MULLER)

		# The radial Hamiltonian for each l-component is defined here.

		for ldx in range(NUM_L):

			V = np.diag(v0 + 0.5 * ldx * (ldx + 1.0) * np.power(r, - 2.0))

			H[ldx, :, :] = - 0.50 * T + V

		eigenStates    = np.zeros((N_r, N_kso_sph), dtype=complex)
		eigenValues    = np.zeros((N_kso_sph), dtype=complex)
		energiesOld    = np.zeros((N_kso_sph), dtype=complex)

		# Initially, the eigenstates are approximated as the eigenstates of the
		# bare Hamiltonian (i.e. v(r) = v_0(r)).
		#
		# The variable idx keeps track of the memory index of the eigenstates while
		# the loop goes through the considered angular momentum components.

		idx = 0

		for ldx in range(NUM_L):

			eigenValues_h, eigenStates_h 			  = getSortedEigenSystem(H[ldx, :, :])
			eigenValues[   idx : idx + l_pop[ldx]]    = eigenValues_h[   0 : l_pop[ldx]]
			eigenStates[:, idx : idx + l_pop[ldx]]    = eigenStates_h[:, 0 : l_pop[ldx]]

			idx += l_pop[ldx]

		normalizeEigenstates (eigenStates)

		# The spherically averaged density:

		density = getDensityU (eigenStates, occ_sph)

		# The SCF loop begins here

		for scfdx in range(N_scf):

			idx = 0

			# To avoid charge sloshing, the density at the beginning of each iteration
			# is mixed with the previous. With each updated Hamiltonian, a new set of
			# eigenstates and density is found. This density is mixed with the density
			# which generated the potential of the Hamiltonian, with the previous density
			# remaining the dominant contribution and the system converging adiabatically
			# to the ground state.

			density = (1.0 - ETA_MIX) * getDensityU (eigenStates, occ_sph) + ETA_MIX * density

			# If the target is Hydrogen or uses the Muller pseudopotential, the e-e interaction
			# is zero.

			if ((targetID == 0) or (USE_MULLER == 1)):

				vhar = 0
				vexc = 0
				vcor = 0
				vscf = np.zeros(N_r)

			else:

				vhar = getHartreePotentialU (density, r)

				# Density must be scaled due to the use of Gauss-Lobatto quadrature. See
				# Manual.pdf.

				densityXC = (N_g * (N_g + 1) / 2.0) * np.power(P, 2.0) / drdx * density

				# The exchange-correlation potential is calculated using the local-density
				# approximation (LDA).

				vexc = dft.getExchangePotential (densityXC / np.power(r, 2.0))

				# Once the calculation is underway, the generalized-gradient approximation,
				# a supplement to the LDA, is introduced.

				if (scfdx > 10):
					vexc += dft.getLB94 (density * P / np.sqrt(drdx), r, D1, drdx, P, N_g)

				# The correlation potential used is found by an exact fitting of
				# Monte-Carlo simulations.

				vcor = dft.getCorrelationPotential (densityXC / np.power(r, 2.0))

				# The total self-consistent field potential is the sum of
				# v_har, v_exc, v_cor.

				vscf = np.diag(vhar + vexc + vcor)

			# With vscf, the eigenstates for each l-component are found and sorted into the
			# eigenstates array.

			for ldx in range(NUM_L):

				eigenValues_h, eigenStates_h 			  = getSortedEigenSystem(H[ldx, :, :] + scfRamp[scfdx] * vscf)
				eigenValues[   idx : idx + l_pop[ldx]]    = eigenValues_h[   0 : l_pop[ldx]]
				eigenStates[:, idx : idx + l_pop[ldx]]    = eigenStates_h[:, 0 : l_pop[ldx]]
				idx += l_pop[ldx]

			normalizeEigenstates (eigenStates)

			# Convergence is calculated as the maximum absolute difference in the current
			# and previous iteration's eigenvalues. This is a less rigorous way to calculate
			# it, but seems to be efficient. A more efficient way would be to calculate
			# difference between 1 and the inner product of the previous and current eigenstates.

			convergence = np.max(np.absolute(eigenValues - energiesOld))
			energiesOld = np.copy(eigenValues)

			if (convergence < SCF_CONVERGENCE):
				break
			else:
				internal.outputSCFResultsIteration (n_sph, l_sph, eigenValues, convergence, N_kso_sph, scfdx)

		internal.outputSCFFinish (n_sph, l_sph, eigenValues, convergence, N_kso_sph, occ_sph, scfdx)

		# The eigenstates are saved for future use.

		np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenStates.npy", eigenStates)
		np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenValues.npy", eigenValues)
		np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/density.npy", density)
		np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/vscf.npy", vscf)
		np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/v0.npy", v0)

		np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/eigenStates.npy", eigenStates)
		np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/r.npy", r)

	else:

		# If not calculating from scratch, the eigenstates are loaded here.

		print("     - Loading previous ground state data ")

		eigenStatesPath = "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenStates.npy"
		eigenStates = np.load(eigenStatesPath)

		eigenValuesPath = "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenValues.npy"
		eigenValues     = np.load(eigenValuesPath)

		densityPath = "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/density.npy"
		density = np.load(densityPath)

		vscfPath = "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/vscf.npy"
		vscf     = np.load(vscfPath)

		v0Path = "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/v0.npy"
		v0     = np.load(v0Path)

		np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/eigenStates.npy", eigenStates)
		np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/r.npy", r)

		print("     - Ground state susccessfuly loaded ")
		print("")

		internal.outputSCFLoad (n_sph, l_sph, eigenValues, 1.0, N_kso_sph, occ_sph, 1)

	# If FIGS == 1, radial linear plots and contour plots of the orbitals are generated.

	if FIGS == 1:
		print("     - Generating Ground State Plots ")
		print("       - Generating density plot ")
		plib.plotSystem (targetID, eigenStates, density, r, drdx, P, N_r, N_g, N_kso_sph, n_sph, l_sph, occ_sph)

	if (targetID > 0):

		densityXC = (N_g * (N_g + 1) / 2.0) * np.power(P, 2.0) / drdx * density

		vhar = getHartreePotentialU (density, r)
		vexcLDA = dft.getExchangePotential (densityXC / np.power(r, 2.0))
		vexcGGA = dft.getLB94 (density * P / np.sqrt(drdx), r, D1, drdx, P, N_g)
		vcor = dft.getCorrelationPotential (densityXC / np.power(r, 2.0))

		plib.linearPlot(r, vhar, xlabel="Radius (a.u.)", ylabel=r'$v_{h}[n](r)$ (a.u.)', xmin=0.0, xmax=10.0, \
						path="../figures/" + getTargetString(targetID) + "/single/static/groundState/vscf/vhar.png")
		plib.linearPlot(r, vexcLDA, xlabel="Radius (a.u.)", ylabel=r'$v_{h}[n](r)$ (a.u.)', xmin=0.0, xmax=10.0, \
						path="../figures/" + getTargetString(targetID) + "/single/static/groundState/vscf/vexcLDA.png")
		plib.linearPlot(r, vexcGGA, xlabel="Radius (a.u.)", ylabel=r'$v_{h}[n](r)$ (a.u.)', xmin=0.0, xmax=10.0, \
						path="../figures/" + getTargetString(targetID) + "/single/static/groundState/vscf/vexcGGA.png")
		plib.linearPlot(r, vexcLDA + vexcGGA, xlabel="Radius (a.u.)", ylabel=r'$v_{h}[n](r)$ (a.u.)', xmin=0.0, xmax=10.0, \
						path="../figures/" + getTargetString(targetID) + "/single/static/groundState/vscf/vexc.png")
		plib.linearPlot(r, vcor, xlabel="Radius (a.u.)", ylabel=r'$v_{h}[n](r)$ (a.u.)', xmin=0.0, xmax=10.0, \
						path="../figures/" + getTargetString(targetID) + "/single/static/groundState/vscf/vcor.png")

	# The eigenstates are taken and placed into the arrays to be saved and later used in the
	# full GPU simulation. In particular, the degenerate l-states are separated according to
	# their magnetic quantum numbers, m.

	Z_sph = np.sum(occ_sph)
	NUM_L = np.size(l_pop)
	N_r = np.size(r)

	sdx, l_active, m_active, occ_active, N_active = internal.getSimulationStates (targetID)

	# The density is partitioned into a frozen density and an active density. The frozen
	# density comes from the orbitals not propagated in time. The active density comprises
	# all the orbitals propagating in time. This is determined by setting which shells, n, which
	# are propagated.

	eigenStatesSIM = np.zeros((NUM_L, N_active, N_r), dtype=complex)
	eigenValuesSIM = np.zeros((N_active), dtype=complex)
	densityActive = np.zeros(N_r)

	for ksodx in range(N_active):
		eigenStatesSIM[l_active[ksodx], ksodx, :] = eigenStates[:, sdx[ksodx]]
		eigenValuesSIM[ksodx] = eigenValues[sdx[ksodx]]
		densityActive += occ_active[ksodx] * np.power(np.absolute(eigenStatesSIM[l_active[ksodx], ksodx, :]), 2.0) / 4.0 / np.pi
		eigenStatesSIM[l_active[ksodx], ksodx, :] = eigenStatesSIM[l_active[ksodx], ksodx, :] / np.sqrt(np.sum(np.power(np.absolute(eigenStatesSIM[l_active[ksodx], ksodx, :]), 2.0)))

	density = getDensityU (eigenStates, occ_sph)
	densityFrozen = density - densityActive

	densityXC = (N_g * (N_g + 1) / 2.0) * np.power(P, 2.0) / drdx * density

	internal.saveEigenValues(targetID, eigenValues)

	return eigenStatesSIM, eigenValuesSIM, v0, np.diag(vscf), N_active, l_active, m_active, occ_active, density, densityFrozen, np.max(m_active) + 1

# def getGroundState2 (r, x, T, D1, W, P, drdx, ETA_MIX, targetID, n, l, m, occ, N_kso, N_l, KSO_MIN, FROM_SCRATCH, METHOD, USE_MULLER, USE_C60, N_scf=65536, SCF_CONVERGENCE=1E-10, FIGS=0):
#
# 	internal.printGSUpdate()
#
# 	N_r = np.size(r)
# 	N_g = N_r + 1
#
# 	Z_sph, n_sph, l_sph, occ_sph, N_kso_sph, NUM_L, active = internal.getSphericalElectronConfiguration (targetID)
#
# 	l_pop = np.zeros(NUM_L, dtype=int)
# 	l_idx = np.zeros(NUM_L, dtype=int)
#
# 	for ldx in range(NUM_L):
#
# 		l_pop[ldx] = np.size(l_sph[l_sph==ldx])
# 		if (ldx == 0):
# 			l_idx[ldx] = 0
# 		else:
# 			l_idx[ldx] = l_idx[ldx - 1] + l_pop[ldx - 1]
#
# 	print("     - Beginning self-consistent field calculation ")
#
# 	H  = np.zeros((NUM_L, N_r, N_r), dtype=complex)
#
# 	if (USE_MULLER == 0 and USE_C60 == 0):
# 		v0 = - Z_sph / r - 1j * W
# 	elif (USE_C60 == 1):
# 		v0 = -0.6 * ((1/np.sqrt(0.2 + np.power(-6.7 + r,2))) - 1/np.sqrt(0.2 + np.power(6.7 + r,2))) - 1j * W
# 	else:
# 		v0 = getMullerPotential(r) - 1j * W
#
# 	for ldx in range(NUM_L):
#
# 		V = np.diag(v0 + 0.5 * ldx * (ldx + 1.0) * np.power(r, - 2.0))
#
# 		H[ldx, :, :] = - 0.50 * T + V
#
# 	internal.outputSCFInit(n_sph, l_sph, occ_sph, N_kso_sph)
#
# 	referenceStates = np.load("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/eigenStates.npy")
# 	referenceR      = np.load("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/r.npy")
# 	referenceR      = np.concatenate([[0.0], referenceR,[r[-1]]])
#
# 	eigenStates    = np.zeros((N_r, N_kso_sph), dtype=complex)
# 	eigenValues    = np.zeros((N_kso_sph), dtype=complex)
# 	energiesOld    = np.zeros((N_kso_sph), dtype=complex)
#
# 	for ksodx in range(N_kso_sph):
# 		temp = np.concatenate([[0.0], referenceStates[:, ksodx],[0.0]])
#
# 		f = sin.interp1d(referenceR, temp)
# 		eigenStates[:, ksodx] = f(r)
#
# 	normalizeEigenstates (eigenStates)
#
# 	density = getDensityU (eigenStates, occ_sph)
#
# 	for scfdx in range(N_scf):
#
# 		idx = 0
#
# 		density = (1.0 - ETA_MIX) * getDensityU (eigenStates, occ_sph) + ETA_MIX * density
#
# 		if ((targetID == 0) or (USE_MULLER == 1)):
#
# 			vhar = 0
# 			vexc = 0
# 			vcor = 0
# 			vscf = np.zeros(N_r)
#
# 		else:
#
# 			vhar = getHartreePotentialU (density, r)
#
# 			densityXC = (N_g * (N_g + 1) / 2.0) * np.power(P, 2.0) / drdx * density
#
# 			vexc = dft.getExchangePotential (densityXC / np.power(r, 2.0))
#
# 			if (scfdx > 10):
# 				vexc += dft.getLB94 (density * P / np.sqrt(drdx), r, D1, drdx, P, N_g)
#
# 			vcor = dft.getCorrelationPotential (densityXC / np.power(r, 2.0))
#
# 			vscf = np.diag(vhar + vexc + vcor)
#
# 		eigenStatesOld = np.copy(eigenStates)
#
# 		for ldx in range(NUM_L):
#
# 			eigenValues_h, eigenStates_h 			  = getSortedEigenSystem(H[ldx, :, :] + vscf)
# 			eigenValues[   idx : idx + l_pop[ldx]]    = eigenValues_h[   0 : l_pop[ldx]]
# 			eigenStates[:, idx : idx + l_pop[ldx]]    = eigenStates_h[:, 0 : l_pop[ldx]]
# 			idx += l_pop[ldx]
#
# 		normalizeEigenstates (eigenStates)
#
# 		convergence = np.max(np.absolute(eigenValues - energiesOld))
# 		energiesOld = np.copy(eigenValues)
#
# 		if (convergence < SCF_CONVERGENCE):
# 			break
# 		else:
# 			internal.outputSCFResultsIteration (n_sph, l_sph, eigenValues, convergence, N_kso_sph, scfdx)
#
# 	internal.outputSCFFinish (n_sph, l_sph, eigenValues, convergence, N_kso_sph, occ_sph, scfdx)
#
#
# 	np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenStates.npy", eigenStates)
# 	np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/eigenValues.npy", eigenValues)
# 	np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/density.npy", density)
# 	np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/vscf.npy", vscf)
# 	np.save("./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r).zfill(4) + "/v0.npy", v0)
#
# 	np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/eigenStates.npy", eigenStates)
# 	np.save("./workingData/reference/groundState/" + str(targetID).zfill(3) + "/r.npy", r)
#
# 	if FIGS == 1:
# 		print("     - Generating Ground State Plots ")
# 		print("       - Generating density plot ")
# 		plotSystem (targetID, eigenStates, density, r, drdx, P, N_r, N_g, N_kso_sph, n_sph, l_sph, occ_sph)
#
# 	Z_sph = np.sum(occ_sph)
# 	NUM_L = np.size(l_pop)
# 	N_r = np.size(r)
#
# 	sdx, l_active, m_active, occ_active, N_active = getSimulationStates (targetID)
#
# 	eigenStatesSIM = np.zeros((NUM_L, N_active, N_r), dtype=complex)
# 	densityActive = np.zeros(N_r)
#
# 	for ksodx in range(N_active):
# 		eigenStatesSIM[l_active[ksodx], ksodx, :] = eigenStates[:, sdx[ksodx]]
# 		densityActive += occ_active[ksodx] * np.power(np.absolute(eigenStatesSIM[l_active[ksodx], ksodx, :]), 2.0) / 4.0 / np.pi
# 		eigenStatesSIM[l_active[ksodx], ksodx, :] = eigenStatesSIM[l_active[ksodx], ksodx, :] / np.sqrt(np.sum(np.power(np.absolute(eigenStatesSIM[l_active[ksodx], ksodx, :]), 2.0)))
#
# 	density = getDensityU (eigenStates, occ_sph)
# 	densityFrozen = density - densityActive
#
# 	densityXC = (N_g * (N_g + 1) / 2.0) * np.power(P, 2.0) / drdx * density
#
# 	internal.saveEigenValues(targetID, eigenValues)
#
# 	return eigenStatesSIM, eigenValues, v0, np.diag(vscf), N_active, l_active, m_active, occ_active, density, densityFrozen, np.max(m_active) + 1
