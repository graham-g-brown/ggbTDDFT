from importlib.machinery import SourceFileLoader
import numpy as np
from phys import internal
from phys import constants
from phys import plotLib as plib
from phys import sfalib as sfa
import matplotlib.colors as colors
import scipy.signal as sig
import scipy.interpolate as sin
import sys

from scipy.signal import windows

import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.colors
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

mpl.rcParams['agg.path.chunksize'] = 1000000

cmap = 'hot'
cmap2 = mpl.colors.LinearSegmentedColormap.from_list("", ["blue",(0,0,0,1),"red"])

SMALL_SIZE  = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10
FIG_DPI     = 300

font_path = './scripts/fonts/Helvetica.ttc'
prop = font_manager.FontProperties(fname=font_path)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.switch_backend('agg')

def STFT (dipoleMoment, E, t, path):

	w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(np.size(t), t[1] - t[0]))
	wMinDX  = np.argmin(np.absolute(w - 15. / 27.2114))
	wMaxDX  = np.argmin(np.absolute(w - 200 / 27.2114))
	wPlot   = w[wMinDX : wMaxDX] * 27.2114

	STFT = np.zeros((np.size(t), np.size(wPlot)))

	tp = t / 41.341374575751

	for k in range(np.size(t)):

		filter = np.exp(- np.power((t - t[k]) / 15., 2.0))

		spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(filter * totalDipole)))
		spectrum = np.power(np.absolute(spectrum), 2.0)

		STFT[k, :] = np.power(spectrum[wMinDX : wMaxDX], 4.0)

	maxdx = np.zeros(np.size(wPlot), dtype=np.int)

	for m in range(np.size(wPlot)):
		if (np.sum(STFT[:, m]) > 1E-60):
			# STFT[:, m] = STFT[:, m] / np.sum(STFT[:, m])
			maxdx[m] = np.argmax(STFT[tp < 2.65, m])
		else:
			# STFT[:, m] = 1E-15
			maxdx[m] = - 1

	GD = tp[maxdx]
	GD[maxdx == np.size(tp) - 1] = np.nan

	STFT = STFT / np.max(STFT)

	T, Wp = np.meshgrid(tp, wPlot)
	# , norm=colors.LogNorm(vmin=1E-5, vmax=1.0)
	fig = plt.figure(figsize=(14.5 / 2.54, 8.96 / 2.54))
	plt.pcolormesh (Wp, T, np.log10(STFT.T), vmin=-20, vmax=- 11, rasterized=True, cmap='gist_heat', shading='auto')
	plt.plot(wPlot, GD, color="tab:red")
	plt.ylabel("Time (fs)")
	plt.xlabel("Energy (eV)")
	cb = plt.colorbar()
	cb.set_label("Intensity (norm.)")
	plt.ylim(- 4.5, 4.5)
	plt.xlim(wPlot[0], wPlot[-1])
	plt.tight_layout()
	plt.savefig(dipMomentSpecFigurePath + "/gaborTransform.png", dpi=300, transparent=True, bbox_inches="tight")
	plt.close(fig)

	plt.figure(figsize=(14.5 / 2.54, 8.96 / 2.54))
	plt.plot(wPlot, GD, color="tab:red")
	plt.ylabel("Time (fs)")
	plt.xlabel("Energy (eV)")
	plt.ylim(0, 4)
	plt.xlim(wPlot[0], wPlot[-1])
	plt.tight_layout()
	plt.savefig(dipMomentSpecFigurePath + "/gaborTransformGD.png", dpi=300, transparent=True, bbox_inches="tight")

internal.printFigureUpdate ()

params = SourceFileLoader("params", "./scripts/phys/parseParams.py").load_module()

SIM_DATE = int(params.SIM_DATE[1 : -1])
SIM_IDX  = int(params.SIM_INDEX[1 : -1])

print(SIM_IDX)
print(SIM_DATE)

params 	 = SourceFileLoader(str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5) + "/params", "./scripts/phys/parseParams2.py").load_module()

TD_SIM   = int(params.TD_SIMULATION[1 : -1])
IN_SITU  = int(params.IN_SITU[1 : -1])

N_r   	 = int(params.N_r[1 : -1])
N_t   	 = int(params.N_t[1 : -1])
N_l   	 = int(params.N_l[1 : -1])
Lext     = N_l - 4

N_har 	 = int(params.N_har[1 : -1])
N_kso    = int(params.N_kso[1 : -1])
N_scan   = int(params.N_scan[1 : -1])

E0       = float(params.E0[1 : -1])
omega0   = float(params.omega0[1 : -1])
tau0     = float(params.tau0[1 : - 1])
targetID = int(params.targetID[1 : -1])

R_MASK   = float(params.R_MASK[1 : -1])
R_MAX    = float(params.R_MAX[1 : -1])
ETA_MASK = float(params.ETA_MASK[1 : -1])

CUTOFF = 1.5000 * (3.17 * np.power(E0 / (2.0 * omega0), 2.0) + 0.5)
W0 	   = 0
PLOT_HO = False
MINI = 1E-25
MAXI = 1E2
N_pad = 8

AVG_WIDTH = 2

UNITS_T  = 1
UNITS_E  = 1

eigenValuesPath = "./workingData/eigenStates/active/" + str(targetID).zfill(3) + "/eigenValues.bin"
eigenValues     = np.fromfile(eigenValuesPath, dtype=np.complex).real

if UNITS_T == 1:
	scalingT = 1.0 / constants.femtosecondAU
else:
	scalingT = 1.0

if UNITS_E == 1:
	scalingE = constants.electronVoltAU
else:
	scalingE = 1.0

outputPath 			= "../output/" + str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5)

meshFilePath 		= outputPath + "/static/mesh"
ksoFilePath  		= outputPath + "/static/kohnShamOrbitals"
dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleMoment"
dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleAcceleration"
vharFilePath		= outputPath + "/timeDependent/observables/vhar"
electricFieldPath   = outputPath + "/timeDependent/field"

if (IN_SITU == 0):
	meshFigurePath 	 	    = "../figures/" + internal.getTargetLabel(targetID) + "/single/static/mesh"
	dipMomentFigurePath  	= "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleMoment/time"
	dipAccFigurePath        = "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleAcceleration/time"
	vharFigurePath  	    = "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/hartreePotential"
	dipMomentSpecFigurePath = "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleMoment/spectrum"
	dipAccSpecFigurePath    = "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleAcceleration/spectrum"
	electricFieldFigurePath = "../figures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/fields"

elif (IN_SITU == 1):
	meshFigurePath 	 	    = "../figures/" + internal.getTargetLabel(targetID) + "/scan/static/mesh"
	dipMomentFigurePath  	= "../figures/" + internal.getTargetLabel(targetID) + "/scan/timeDependent/observables/dipoleMoment/time"
	dipMomentSpecFigurePath = "../figures/" + internal.getTargetLabel(targetID) + "/scan/timeDependent/observables/dipoleMoment/spectrum"
	electricFieldFigurePath = "../figures/" + internal.getTargetLabel(targetID) + "/scan/timeDependent/fields"

r    = np.fromfile(meshFilePath + "/r.bin")
drdx = np.fromfile(meshFilePath + "/drdx.bin")
P    = np.fromfile("./workingData/mesh/P.bin")

t   	   = np.fromfile(meshFilePath + "/time.bin")
dt  	   = t[1] - t[0]
dw 		   = 2.0 * np.pi * np.fft.fftfreq(N_t, dt)

# filterDIP  = np.power(np.sin(np.pi * (t - t[0]) / (t[-1] - t[0])), 4.0) # sig.windows.kaiser(N_t, beta=10, sym=False)
filterDIP = sig.windows.kaiser(N_t, beta=24, sym=False)
# filterDIP = np.exp(- np.power(t / 0.35 / tau0, 10.0))

occ = np.fromfile("./workingData/stateParameters/occ_active.bin", dtype=np.intc)
occ = np.array([2, 2, 2, 4, 2, 4, 2, 4, 4])

print("      \u2022 Mesh:")

xxx = np.linspace(0, N_r - 1, N_r)

print("         \u2022 Radius")

r = np.fromfile(meshFilePath + "/r.bin")
plib.linearPlot (xxx, r, \
		xlabel=r'Memory Index', ylabel=r'Radius (a.u.)', \
		path=meshFigurePath + "/r.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

plib.linearPlot (xxx, internal.dfdx(r), \
		xlabel=r'Memory Index', ylabel=r'$r$', \
		path=meshFigurePath + "/drdx.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

print("         \u2022 Complex Absorbing Potential")

ABC = internal.getABCR(r, R_MASK, R_MAX, 1)

plib.linearPlot (r, ABC.real, \
		xlabel=r'Radius (a.u.)', ylabel=r'$W(r)$', \
		path=meshFigurePath + "/abc.png", log=True, logx=False, ymin=1E-15, ymax=1E12, figsize=plib.cm2inch(14.5, 8.96), dot=False, thesisStyle=False)

if IN_SITU == 1:

	phi     = np.linspace(0.0, 2.0 * np.pi, N_scan)

	N_t_pad = (2 * N_pad + 1) * N_t
	t_pad   = np.linspace(- 0.5 * N_t_pad, 0.5 * N_t_pad - 1, N_t_pad) * dt
	w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t_pad, dt))
	dw      = w[1] - w[0]

	phi     = np.linspace(0.0, 2.0 * np.pi, N_scan)
	wMinDX  = np.argmin(np.absolute(w))
	wMaxDX  = np.argmin(np.absolute(w - CUTOFF))

	wPlot   = w[wMinDX : wMaxDX]

	WPLOT, PHI = np.meshgrid(wPlot, phi)

	N_wp    = np.size(wPlot)

	SPECTRA = np.zeros((N_scan, N_wp), dtype=np.complex)

	outputPath0 = "../output/" + str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5)

	for scandx in range(N_scan):
		print(scandx)

		E = np.fromfile(outputPath0 + "/timeDependent/field/E" + str(scandx).zfill(3) + ".bin", dtype=float)
		plib.linearPlot (t * scalingT, E, \
							xlabel=r'$t$ (a.u.)', ylabel=r'$d(t)$', \
							xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
							path=electricFieldFigurePath + "/E" + str(scandx).zfill(3) + ".png", \
							log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		dipoleMoment = np.zeros((N_kso, N_t), dtype=float)
		dipoleAcc    = np.zeros((N_kso, N_t), dtype=float)

		for ksodx in range(N_kso):

			for ldx in range(N_l):
				dipoleMomentTemp = np.fromfile(outputPath0 + "/timeDependent/observables/dipoleMoment/scan" \
												+ str(scandx).zfill(3) + "/kso" + str(ksodx).zfill(3) \
												+ "/dipL" + str(ldx).zfill(3) + ".bin", dtype=complex)

				dipoleMoment[ksodx, :] += dipoleMomentTemp.real

			dipoleMoment[ksodx, :] *= occ[ksodx]

		totalDipole = np.sum(dipoleMoment[:, :], axis=0) * filterDIP

		totalDipole   = np.pad(internal.d2fdx2(totalDipole, dt), N_pad * N_t, mode='constant')
		spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipole)))
		SPECTRA[scandx, :] = spectrum[wMinDX : wMaxDX]

	np.save("./SPECTRA.npy", SPECTRA)
	np.save("./w.npy", wPlot)

	# SPEC0 = np.zeros(np.size(spectrum[wMinDX : wMaxDX]), dtype=np.complex)
	# SPEC0[:] = spectrum[wMinDX : wMaxDX]
	#
	# for scandx in range(N_scan):
	#
	# 	print(scandx)
	#
	# 	electricFieldPath   = outputPath + "/timeDependent/field/" + "E" + str(scandx).zfill(3) + ".bin"
	# 	dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleMoment/" + str(scandx).zfill(3) + ".bin"
	# 	dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleAcceleration/" + str(scandx).zfill(3) + ".bin"
	#
	# 	E    = np.fromfile(electricFieldPath, dtype=np.float)
	#
	# 	dipoleMoment = np.zeros((N_kso, N_t), dtype=np.complex)
	# 	dipoleAcc    = np.zeros((N_kso, N_t), dtype=np.complex)
	#
	# 	for ksodx in range(N_kso):
	#
	# 		for ldx in range(N_l):
	# 			dipoleMomentTemp = np.zeros(N_t, dtype=np.complex)
	# 			dipoleMomentTemp[:] = np.fromfile(outputPath + "/timeDependent/observables/dipoleMoment/scan" \
	# 											+ str(scandx).zfill(3) + "/kso" + str(ksodx).zfill(3) \
	# 											+ "/dipL" + str(ldx).zfill(3) + ".bin", dtype=complex)
	#
	# 			dipoleMoment[ksodx, :] += dipoleMomentTemp * filterDIP
	#
	# 		dipoleMoment[ksodx, :] *= occ[ksodx]
	#
	# 	totalDipole = np.sum(dipoleMoment[:, :], axis=0)
	#
	# 	totalDipole *= filterDIP
	#
	# 	for ksodx in range(N_kso):
	# 		plib.linearPlot (t * scalingT, dipoleMoment[ksodx, :].real, \
	# 					xlabel=r'$t$ (a.u.)', ylabel=r'$d(t)$', \
	# 					xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
	# 					path=dipMomentFigurePath + "/full/kso" + str(ksodx).zfill(3) + "/linearPlots/scan" + str(scandx).zfill(3) + ".png", \
	# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
	#
	#
	# 	totalDipole   = np.pad(internal.d2fdx2(totalDipole, dt), N_pad * N_t, mode='constant')
	# 	spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipole)))
	# 	spectrumTotal = np.power(np.absolute(spectrum), 2.0)
	# 	spectrumPhase = np.unwrap(np.angle(spectrum))
	# 	groupDelay    = - internal.dfdx(spectrumPhase, dw)
	#
	# 	plib.linearPlot (w[wMinDX : wMaxDX] * scalingE, spectrumTotal[wMinDX : wMaxDX].real, \
	# 				xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
	# 				xmin=W0, xmax=CUTOFF * scalingE, \
	# 				ymin=MINI, ymax=MAXI,
	# 				path=dipMomentSpecFigurePath + "/total/linearPlots/scan" + str(scandx).zfill(3) + ".png", log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False, thesisStyle=False)
	#
	# 	SPECTRA[scandx, :] = spectrum[wMinDX : wMaxDX] - np.sum(np.conj(SPEC0) * spectrum[wMinDX : wMaxDX]) * SPEC0 / np.sum(np.power(np.absolute(SPEC0), 2.0))
	#
	# SPECTRA = np.power(np.absolute(SPECTRA), 2.0)
	#
	# phiInt = np.linspace(phi[0], phi[-1], 8192)
	#
	# WPLOT, PHI = np.meshgrid(wPlot, phiInt)
	#
	# SPECTRA_INT = np.zeros((np.size(phiInt), np.shape(SPECTRA)[1]))
	# maxPhi = np.zeros(np.shape(SPECTRA_INT)[1])
	#
	# for k in range(np.shape(SPECTRA)[1]):
	# 	SPECTRA[:, k] -= np.min(SPECTRA[:, k])
	# 	SPECTRA[:, k] /= np.max(np.absolute(SPECTRA[:, k]))
	# 	SPECTRA[:, k] -= 0.50
	# 	SPECTRA[:, k] *= 2.0
	#
	# 	specTemp = sin.splrep(phi, SPECTRA[:, k])
	# 	specInterp = sin.splev(phiInt, specTemp)
	# 	SPECTRA_INT[:, k] = specInterp[:]
	# 	# maxPhi[k] = phiInt[4096 + np.argmax(SPECTRA_INT[4096 : 8192, k])]
	# SPECTRA_INT = SPECTRA_INT[::-1, :]
	# SPEC_T = np.fft.fftshift(np.fft.fft(np.fft.fftshift(SPECTRA_INT), axis=0))
	# SPEC_T_SUM = np.sum(np.absolute(SPEC_T), axis=1)
	# maxPhi = np.unwrap(np.angle(SPEC_T[4094, :]))
	# maxPhi[:] /= 2.0
	#
	# fig = plt.figure(figsize=(14.5 / 2.54, 8.96 / 2.54))
	# plt.pcolormesh (scalingE * WPLOT, PHI, SPECTRA_INT[:, :].real, vmin=-1.0, vmax=1.0, rasterized=True, cmap='gist_heat', shading='auto')
	# plt.ylabel("$r_0$ (a.u.)", fontsize=10)
	# plt.xlabel("Energy (eV)", fontsize=10)
	# plt.xlim(20., scalingE * CUTOFF)
	# cb = plt.colorbar()
	# cb.set_label("Intensity (norm.)", fontsize=10)
	# plt.tight_layout()
	# plt.savefig(dipMomentSpecFigurePath + "/total/spectrogram2.png", dpi=300, transparent=True, bbox_inches="tight")
	# plt.close(fig)
	#
	# theta = 30E-3
	# k = 1.0 / 1800E-9

if IN_SITU == 0:

	if TD_SIM == 1:

		N_t_pad = (2 * N_pad + 1) * N_t
		t_pad   = np.linspace(- 0.5 * N_t_pad, 0.5 * N_t_pad - 1, N_t_pad) * dt
		w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t_pad, dt))
		dw      = w[1] - w[0]

		print("      \u2022 Electric Field")

		SCANDXDX = 0

		E    = np.fromfile(electricFieldPath + "/E" + str(SCANDXDX).zfill(3) + ".bin")

		plib.linearPlot (t * scalingT, E, x2=t * scalingT, f2=filterDIP * np.max(E), \
					xlabel=r'$t$ (a.u.)', ylabel=r'$E(t)$', \
					xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
					path=electricFieldFigurePath + "/E.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		print("      \u2022 Observables")

		dipoleMoment = np.zeros((N_kso, N_t), dtype=complex)
		dipoleAcc    = np.zeros((N_kso, N_t), dtype=float)
		VHAR         = np.zeros((N_kso, N_t), dtype=complex)

		print("         \u2022 Dipole Moment:")

		KSOL = internal.getKSOL(targetID)
		KSOM = internal.getKSOM(targetID)
		print(N_kso)
		for ksodx in range(N_kso):

			if (KSOL[ksodx] == 0 or KSOL[ksodx] == KSOM[ksodx]):
				mindx = KSOL[ksodx]
			else:
				mindx = KSOM[ksodx]

			print(ksodx, mindx)

			Larray = np.linspace(mindx, Lext - 1, Lext + 1 - mindx).astype(np.int)

			Larray = np.delete(Larray, np.where(Larray == KSOL[ksodx]))

			Larray = np.unique(Larray)

			for ldx in Larray:

				dipoleMomentTemp = np.fromfile(outputPath + "/timeDependent/observables/dipoleMoment/scan" \
												+ str(0).zfill(3) + "/kso" + str(ksodx).zfill(3) \
												+ "/dipL" + str(ldx).zfill(3) + ".bin", dtype=complex)

				dipoleMoment[ksodx, :] += dipoleMomentTemp.real

			dipoleMoment[ksodx, :] *= occ[ksodx] * filterDIP

		totalDipole = np.sum(dipoleMoment, axis=0)

		print("            \u2022 Time")

		for ksodx in range(N_kso):
			plib.linearPlot (t * scalingT, - dipoleMoment[ksodx, :], \
						xlabel=r'$t$ (a.u.)', ylabel=r'$d(t)$', \
						xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
						path=dipMomentFigurePath + "/full/kso" + str(ksodx).zfill(3) + ".png", \
						log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		# Total

		dDipoleF      = np.pad(totalDipole, N_pad * N_t, mode='constant')
		dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
		Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))

		print("            \u2022 Gabor Transform")

		STFT (dDipoleF, E, t, dipMomentFigurePath)

		print("            \u2022 Spectra")

		totalDipole   = np.pad(internal.d2fdx2(totalDipole, dt), N_pad * N_t, mode='constant')
		spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipole)))
		spectrumTotal = np.power(np.absolute(spectrum), 2.0)
		spectrumPhase = np.unwrap(np.angle(spectrum))

		groupDelay    = - internal.dfdx(spectrumPhase, dw)
		groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)

		plib.linearPlot (w * scalingE, spectrumTotal, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
					xmin=W0, xmax=CUTOFF * scalingE, \
					ymin=MINI, ymax=MAXI, \
					path=dipMomentSpecFigurePath + "/total.png", log=True, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False, thesisStyle=False)

		plib.linearPlot (w * scalingE, spectrumTotal, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
					xmin=W0, xmax=CUTOFF * scalingE,
					path=dipMomentSpecFigurePath + "/totalAmp.png", ymin=0.0, ymax=4E-4, log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False, thesisStyle=False)

		plib.linearPlot (w * scalingE, spectrumTotal, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
					xmin=W0, xmax=CUTOFF * scalingE, \
					ymin=MINI, ymax=MAXI,
					path=outputPath + "/total.png", log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		spectrumPhase = spectrumPhase - np.min(spectrumPhase)

		plib.linearPlot (w * scalingE, spectrumPhase, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
					xmin=W0, xmax=CUTOFF * scalingE,\
					path=dipMomentSpecFigurePath + "/phase.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
					xmin=10, xmax=CUTOFF * scalingE,\
					ymin=0.0, ymax=5.0,
					path=dipMomentSpecFigurePath + "/groupDelay.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False)

		plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
					xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
					xmin=10, xmax=CUTOFF * scalingE,\
					ymin=0.0, ymax=5.0,
					path=outputPath + "/groupDelay.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False)

		# Orbital

		np.save("./orbitalXe/w.npy", w * scalingE)

		for ksodx in range(N_kso):

			print(N_kso, ksodx)

			totalDipoleKSO = np.pad(internal.d2fdx2(dipoleMoment[ksodx, :], dt), N_pad * N_t, mode='constant')
			spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipoleKSO)))
			spectrumTotal = np.power(np.absolute(spectrum), 2.0)
			spectrumPhase = np.unwrap(np.angle(spectrum))

			groupDelay    = - internal.dfdx(spectrumPhase, dw)
			groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)

			np.save("./orbitalXe/spec" + str(ksodx).zfill(3) + ".npy", spectrum)
			np.save("./orbitalXe/gd" + str(ksodx).zfill(3) + ".npy", groupDelay)

			plib.linearPlot (w * scalingE, spectrumTotal,\
							 xlabel=r'$\Omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
							 xmin=W0, xmax=CUTOFF * scalingE, \
							 ymin=MINI, ymax=MAXI, \
							 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/spectrum.png", \
							 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, spectrumPhase, \
							 xlabel=r'$\Omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
							 xmin=W0, xmax=CUTOFF,\
							 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/phase.png", \
							 log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, groupDelay * scalingT, \
							 xlabel=r'$\Omega$ (eV)', ylabel=r'Group Delay (fs)', \
							 xmin=W0, xmax=CUTOFF * scalingE,\
							 ymin=0, ymax=6.0,\
							 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/groupDelay.png", \
							 log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

		if (targetID == 5):

			# Shell

			# # 4

			dipoleShell = dipoleMoment[0, :] \
						+ dipoleMoment[2, :] \
						+ dipoleMoment[3, :] \
						+ dipoleMoment[6, :] \
						+ dipoleMoment[7, :] \
						+ dipoleMoment[8, :]

			spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.pad(internal.d2fdx2(dipoleShell, dt), N_pad * N_t, mode='constant'))))
			spec = np.power(np.absolute(spectrum), 2.0)
			spectrumPhase = np.unwrap(np.angle(spectrum))

			dDipoleF      = np.pad(dipoleShell, N_pad * N_t, mode='constant')
			dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
			Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))

			groupDelay	  = (dDipole + 1j * internal.dfdx(np.absolute(Dipole), dx=dw)) / (Dipole + 1E-6j)
			groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)

			plib.linearPlot (w * scalingE, spec,\
							 xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
							 xmin=W0, xmax=CUTOFF * scalingE, \
							 ymin=MINI, ymax=MAXI, \
							 path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/spectrum.png", \
							 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, spectrumPhase, \
						xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Phase (rad.)', \
						xmin=W0, xmax=CUTOFF * scalingE,\
						path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/phase.png", \
						log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
						xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
						xmin=W0, xmax=CUTOFF * scalingE,\
						ymin=0, ymax=6.0,\
						path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/groupDelay.png", \
						log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			# # 5

			dipoleShell = dipoleMoment[1, :] \
						+ dipoleMoment[4, :] \
						+ dipoleMoment[5, :]

			spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.pad(internal.d2fdx2(dipoleShell, dt), N_pad * N_t, mode='constant'))))
			spec = np.power(np.absolute(spectrum), 2.0)
			spectrumPhase = np.unwrap(np.angle(spectrum))

			dDipoleF      = np.pad(dipoleShell, N_pad * N_t, mode='constant')
			dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
			Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))

			groupDelay    = (dDipole + 1j * internal.dfdx(np.absolute(Dipole), dx=dw)) / (Dipole + 1E-6j)
			groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)

			plib.linearPlot (w * scalingE, spec,\
							 xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
							 xmin=W0, xmax=CUTOFF * scalingE, \
							 ymin=MINI, ymax=MAXI, \
							 path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/spectrum.png", \
							 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, spectrumPhase, \
						xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Phase (rad.)', \
						xmin=W0, xmax=CUTOFF * scalingE,\
						path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/phase.png", \
						log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

			plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
						xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
						xmin=W0, xmax=CUTOFF * scalingE,\
						ymin=0, ymax=6.0,\
						path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/groupDelay.png", \
						log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)

print("")
print("  6. Program Finished")
print("")
