from importlib.machinery import SourceFileLoader
import numpy as np
from phys import internal
from phys import constants
from phys import plotLib as plib
from phys import sfalib as sfa
from phys import dft
import matplotlib.colors as colors
import scipy.signal as sig
import scipy.interpolate as sin
import sys
import tikzplotlib

from scipy.signal import windows

import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.colors
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable


mpl.rcParams['agg.path.chunksize'] = 1000000

cmap = 'hot'
cmap2 = mpl.colors.LinearSegmentedColormap.from_list("", ["blue",(0,0,0,1),"red"])

SMALL_SIZE  = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 10
FIG_DPI     = 300

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# mpl.rcParams['agg.path.chunksize'] = 1000000

cmap = 'hot'

# SMALL_SIZE  = 10
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 10
# FIG_DPI     = 300
#
# font_path = './scripts/fonts/Helvetica.ttc'
# prop = font_manager.FontProperties(fname=font_path)
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#
plt.switch_backend('agg')

def STFT (dipoleMoment, E, t, path):

	w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(np.size(t), t[1] - t[0]))
	wMinDX  = np.argmin(np.absolute(w - 15. / 27.2114))
	wMaxDX  = np.argmin(np.absolute(w - 150 / 27.2114))
	wPlot   = w[wMinDX : wMaxDX] * 27.2114

	STFT = np.zeros((np.size(t), np.size(wPlot)))

	tp = t / 41.341374575751

	for k in range(np.size(t)):

		filter = np.exp(- np.power((t - t[k]) / 30., 2.0))

		spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(filter * totalDipole)))
		spectrum = np.power(np.absolute(spectrum), 2.0)

		STFT[k, :] = np.power(spectrum[wMinDX : wMaxDX], 4.0)

	maxdx = np.zeros(np.size(wPlot), dtype=np.int)

	for m in range(np.size(wPlot)):
		if (np.sum(STFT[:, m]) > 1E-60):
			STFT[:, m] = STFT[:, m] / np.sum(STFT[:, m])
			maxdx[m] = np.argmax(STFT[:, m])
		else:
			STFT[:, m] = 1E-15
			maxdx[m] = - 1

	GD = tp[maxdx]
	GD[maxdx == np.size(tp) - 1] = np.nan

	STFT = STFT / np.max(STFT)

	T, Wp = np.meshgrid(tp, wPlot)

	fig = plt.figure(figsize=(14.5 / 2.54, 8.96 / 2.54))
	plt.pcolormesh (Wp, T, STFT.T, norm=colors.LogNorm(vmin=1E-5, vmax=1.0), rasterized=True, cmap='gist_heat', shading='auto')
	plt.plot(wPlot, GD, color="tab:red")
	plt.ylabel("Time (fs)")
	plt.xlabel("Energy (eV)")
	cb = plt.colorbar()
	cb.set_label("Intensity (norm.)")
	plt.ylim(0, 4)
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

SIM_DATE = 20210816# int(params.SIM_DATE[1 : -1])
SIM_IDX  = 5 # int(params.SIM_INDEX[1 : -1]) - 1

params 	 = SourceFileLoader(str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5) + "/params", "./scripts/phys/parseParams2.py").load_module()

TD_SIM   = int(params.TD_SIMULATION[1 : -1])
IN_SITU  = int(params.IN_SITU[1 : -1])

N_r   	 = int(params.N_r[1 : -1])
N_t   	 = int(params.N_t[1 : -1])
N_l   	 = int(params.N_l[1 : -1])
Lext     = 64

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

AVG_WIDTH = 4

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
ksoFilePath  		= "./workingData/eigenStates/fullGroundState/" + str(targetID).zfill(3) + "/" + str(N_r)
dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleMoment"
dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleAcceleration"
vharFilePath		= outputPath + "/timeDependent/observables/vhar"
electricFieldPath   = outputPath + "/timeDependent/field"


meshFigurePath 	 	    = "../thesisFigures/" + internal.getTargetLabel(targetID) + "/single/static/mesh"
groundStateFigurePath   = "../thesisFigures/" + internal.getTargetLabel(targetID) + "/single/static/groundState"
dipMomentFigurePath  	= "../thesisFigures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleMoment/time"
dipMomentSpecFigurePath = "../thesisFigures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/observables/dipoleMoment/spectrum"
electricFieldFigurePath = "../thesisFigures/" + internal.getTargetLabel(targetID) + "/single/timeDependent/fields"

r    = np.fromfile(meshFilePath + "/r.bin")
drdx = np.fromfile(meshFilePath + "/drdx.bin")
P    = np.fromfile("./workingData/mesh/P.bin")
D1   = np.fromfile("./workingData/mesh/D1.bin")
D1   = D1.reshape((N_r, N_r), order="C")

density = np.load(ksoFilePath + "/density.npy")
density = (N_r + 1) * (N_r + 2) / 2.0 * density / drdx * P * P

eigenStates = np.load(ksoFilePath + "/eigenStates.npy")
vxc = dft.getExchangePotential (density / np.power(r, 2.0)) + dft.getCorrelationPotential (density / np.power(r, 2.0))
vgga = dft.getLB94 (density * P / np.sqrt(drdx), r, D1, drdx, P, N_r + 1)

t   	   = np.fromfile(meshFilePath + "/time.bin")
dt  	   = t[1] - t[0]
dw 		   = 2.0 * np.pi * np.fft.fftfreq(N_t, dt)

filterDIP  = sig.windows.kaiser(N_t, beta=14, sym=False)

occ = np.fromfile("./workingData/stateParameters/occ_active.bin", dtype=np.intc)

print("      \u2022 Mesh:")

xxx = np.linspace(0, N_r - 1, N_r)

print("         \u2022 Radius")

r = np.fromfile(meshFilePath + "/r.bin")

figs, ax = plt.subplots(1, 1, figsize=(89 / 25.4, 55 / 25.4))
ax.plot(xxx, r)
ax.set_xlabel("Memory Index")
ax.set_ylabel("Radius (a.u.)")
tikzplotlib.save("mytikz.tex")

figs, axes = plt.subplots(2, 2)
ax = axes[0,0]
ax.semilogx(r, density)
ax.set_xlabel("Radius (a.u.)")
ax.set_ylabel("Density (norm.)")
ax = axes[0,1]
ax.semilogx(r, eigenStates[:, 8].real * np.sqrt((N_r + 1) * (N_r + 2) / 2.0 / drdx) * P, label=r'$5p$')
ax.semilogx(r, eigenStates[:, 10].real * np.sqrt((N_r + 1) * (N_r + 2) / 2.0 / drdx) * P, label=r'$4d$')
ax.set_xlabel("Radius (a.u.)")
ax.set_ylabel("Wavefunction (norm.)")
ax.legend()
ax = axes[1, 0]
ax.plot(r, vxc)
ax.set_xlim(0.0, 5.0)
ax.set_ylim(- 20.00, 5.0)
ax = axes[1, 1]
ax.plot(r, vgga)
ax.set_xlim(0.0, 5.0)
ax.set_ylim(- 20.00, 5.0)
tikzplotlib.save(groundStateFigurePath + "/5p.tex")

w = np.load("./xenonIntensityScan/w.npy")

spec0 = np.load("./xenonIntensityScan/spec2021081600000.npy")
spec1 = np.load("./xenonIntensityScan/spec2021081600001.npy")
spec2 = np.load("./xenonIntensityScan/spec2021081600002.npy")
spec3 = np.load("./xenonIntensityScan/spec2021081600003.npy")
spec4 = np.load("./xenonIntensityScan/spec2021081600004.npy")
spec5 = np.load("./xenonIntensityScan/spec2021081600005.npy")
w = w * 27.2114
minwdx = np.argmin(np.absolute(w - 0))
maxwdx = np.argmin(np.absolute(w - 160.0))
figs, ax = plt.subplots(1, 1, figsize=(89 / 25.4, 55 / 25.4))
ax.semilogy(w[minwdx : maxwdx], spec0[minwdx : maxwdx], label=r'$0.5 \times 10^{14}$ W/cm$^{2}$')
ax.semilogy(w[minwdx : maxwdx], spec1[minwdx : maxwdx], label=r'$0.6 \times 10^{14}$ W/cm$^{2}$')
ax.semilogy(w[minwdx : maxwdx], spec2[minwdx : maxwdx], label=r'$0.7 \times 10^{14}$ W/cm$^{2}$')
ax.semilogy(w[minwdx : maxwdx], spec3[minwdx : maxwdx], label=r'$0.8 \times 10^{14}$ W/cm$^{2}$')
ax.semilogy(w[minwdx : maxwdx], spec4[minwdx : maxwdx], label=r'$0.9 \times 10^{14}$ W/cm$^{2}$')
ax.semilogy(w[minwdx : maxwdx], spec5[minwdx : maxwdx], label=r'$1.0 \times 10^{14}$ W/cm$^{2}$')
ax.set_xlim(0.0, 160.0)
ax.set_ylim(1E-8, 1E0)
ax.set_xlabel("Energy (eV)",  fontsize=9)
ax.set_ylabel("Intensity (arb.)", fontsize=9)
# ax.legend()
plt.tight_layout()
plt.savefig(dipMomentSpecFigurePath + "/intensityScan.pdf", format="PDF")
# tikzplotlib.save(dipMomentSpecFigurePath + "/specIntensityScan.tex")

# if TD_SIM == 1:
#
# 	N_t_pad = (2 * N_pad + 1) * N_t
# 	t_pad   = np.linspace(- 0.5 * N_t_pad, 0.5 * N_t_pad - 1, N_t_pad) * dt
# 	w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t_pad, dt))
# 	dw      = w[1] - w[0]
#
# 	print("      \u2022 Electric Field")
#
# 	SCANDXDX = 0
#
# 	E    = np.fromfile(electricFieldPath + "/E" + str(SCANDXDX).zfill(3) + ".bin")
#
# 	plib.linearPlot (t * scalingT, E, x2=t * scalingT, f2=filterDIP * np.max(E), \
# 				xlabel=r'$t$ (a.u.)', ylabel=r'$E(t)$', \
# 				xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
# 				path=electricFieldFigurePath + "/E.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 	print("      \u2022 Observables")
#
# 	dipoleMoment = np.zeros((N_kso, N_t), dtype=complex)
# 	dipoleAcc    = np.zeros((N_kso, N_t), dtype=float)
# 	VHAR         = np.zeros((N_kso, N_t), dtype=complex)
#
# 	print("         \u2022 Dipole Moment:")
#
# 	KSOL = internal.getKSOL(targetID)
# 	KSOM = internal.getKSOM(targetID)
#
# 	for ksodx in range(N_kso):
#
# 		if (KSOL[ksodx] == 0 or KSOL[ksodx] == KSOM[ksodx]):
# 			mindx = KSOL[ksodx]
# 		else:
# 			mindx = KSOM[ksodx]
#
# 		Larray = np.linspace(mindx, KSOL[ksodx] + Lext, KSOL[ksodx] + Lext + 1 - mindx).astype(np.int)
#
# 		Larray = np.delete(Larray, np.where(Larray == KSOL[ksodx]))
# 		# Larray = np.delete(Larray, np.where(Larray == KSOL[ksodx] - 1))
#
# 		# if (ksodx == 2):
# 		# 	sys.exit()
#
# 		for ldx in Larray:
#
# 			dipoleMomentTemp = np.fromfile(outputPath + "/timeDependent/observables/dipoleMoment/scan" \
# 											+ str(0).zfill(3) + "/kso" + str(ksodx).zfill(3) \
# 											+ "/dipL" + str(ldx).zfill(3) + ".bin", dtype=complex)
#
# 			dipoleMoment[ksodx, :] += dipoleMomentTemp.real
#
# 		dipoleMoment[ksodx, :] *= occ[ksodx] * filterDIP
#
# 	totalDipole = np.sum(dipoleMoment, axis=0)
#
# 	print("            \u2022 Time")
#
# 	for ksodx in range(N_kso):
# 		plib.linearPlot (t * scalingT, - dipoleMoment[ksodx, :], \
# 					xlabel=r'$t$ (a.u.)', ylabel=r'$d(t)$', \
# 					xmin=t[0] * scalingT, xmax=t[-1] * scalingT, \
# 					path=dipMomentFigurePath + "/full/kso" + str(ksodx).zfill(3) + ".png", \
# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 	# Total
#
# 	dDipoleF      = np.pad(totalDipole, N_pad * N_t, mode='constant')
# 	dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
# 	Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))
#
# 	print("            \u2022 Gabor Transform")
#
# 	STFT (dDipoleF, E, t, dipMomentFigurePath)
#
# 	print("            \u2022 Spectra")
#
# 	totalDipole   = np.pad(internal.d2fdx2(totalDipole, dt), N_pad * N_t, mode='constant')
# 	spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipole)))
# 	spectrumTotal = np.power(np.absolute(spectrum), 2.0)
# 	spectrumPhase = np.unwrap(np.angle(spectrum))
#
# 	groupDelay    = - internal.dfdx(spectrumPhase, dw)
#
# 	plib.linearPlot (w * scalingE, spectrumTotal, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 				xmin=W0, xmax=CUTOFF * scalingE, \
# 				ymin=MINI, ymax=MAXI, \
# 				path=dipMomentSpecFigurePath + "/total.png", log=True, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False, thesisStyle=False)
#
# 	plib.linearPlot (w * scalingE, spectrumTotal, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 				xmin=W0, xmax=CUTOFF * scalingE,
# 				path=dipMomentSpecFigurePath + "/totalAmp.png", ymin=0.0, ymax=4E-4, log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False, thesisStyle=False)
#
# 	plib.linearPlot (w * scalingE, spectrumTotal, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 				xmin=W0, xmax=CUTOFF * scalingE, \
# 				ymin=MINI, ymax=MAXI,
# 				path=outputPath + "/total.png", log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 	spectrumPhase = spectrumPhase - np.min(spectrumPhase)
#
# 	plib.linearPlot (w * scalingE, spectrumPhase, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 				xmin=W0, xmax=CUTOFF * scalingE,\
# 				path=dipMomentSpecFigurePath + "/phase.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 	plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
# 				xmin=10, xmax=CUTOFF * scalingE,\
# 				ymin=0.0, ymax=5.0,
# 				path=dipMomentSpecFigurePath + "/groupDelay.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False)
#
# 	plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
# 				xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
# 				xmin=10, xmax=CUTOFF * scalingE,\
# 				ymin=0.0, ymax=5.0,
# 				path=outputPath + "/groupDelay.png", log=False, logx=False, figsize=plib.cm2inch(14.5, 7.0), dot=False)
#
# 	# Orbital
#
# 	for ksodx in range(N_kso):
#
# 		totalDipoleKSO = np.pad(internal.d2fdx2(dipoleMoment[ksodx, :], dt), N_pad * N_t, mode='constant')
# 		spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipoleKSO)))
# 		spectrumTotal = np.power(np.absolute(spectrum), 2.0)
# 		spectrumPhase = np.unwrap(np.angle(spectrum))
#
# 		groupDelay    = - internal.dfdx(spectrumPhase, dw)
#
# 		plib.linearPlot (w * scalingE, spectrumTotal,\
# 						 xlabel=r'$\Omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 						 xmin=W0, xmax=CUTOFF * scalingE, \
# 						 ymin=MINI, ymax=MAXI, \
# 						 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/spectrum.png", \
# 						 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, spectrumPhase, \
# 						 xlabel=r'$\Omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 						 xmin=W0, xmax=CUTOFF,\
# 						 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/phase.png", \
# 						 log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, groupDelay * scalingT, \
# 						 xlabel=r'$\Omega$ (eV)', ylabel=r'Group Delay (fs)', \
# 						 xmin=W0, xmax=CUTOFF * scalingE,\
# 						 ymin=0, ymax=6.0,\
# 						 path=dipMomentSpecFigurePath + "/orbitals/" + str(ksodx).zfill(3) + "/groupDelay.png", \
# 						 log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 	if (targetID == 5):
#
# 		# Shell
#
# 		# # 4
#
# 		dipoleShell = dipoleMoment[0, :] \
# 					+ dipoleMoment[2, :] \
# 					+ dipoleMoment[3, :] \
# 					+ dipoleMoment[6, :] \
# 					+ dipoleMoment[7, :] \
# 					+ dipoleMoment[8, :]
#
# 		spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.pad(internal.d2fdx2(dipoleShell, dt), N_pad * N_t, mode='constant'))))
# 		spec = np.power(np.absolute(spectrum), 2.0)
# 		spectrumPhase = np.unwrap(np.angle(spectrum))
#
# 		dDipoleF      = np.pad(dipoleShell, N_pad * N_t, mode='constant')
# 		dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
# 		Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))
#
# 		groupDelay	  = (dDipole + 1j * internal.dfdx(np.absolute(Dipole), dx=dw)) / (Dipole + 1E-6j)
# 		# groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)
#
# 		plib.linearPlot (w * scalingE, spec,\
# 						 xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 						 xmin=W0, xmax=CUTOFF * scalingE, \
# 						 ymin=MINI, ymax=MAXI, \
# 						 path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/spectrum.png", \
# 						 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, spectrumPhase, \
# 					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Phase (rad.)', \
# 					xmin=W0, xmax=CUTOFF * scalingE,\
# 					path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/phase.png", \
# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
# 					xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
# 					xmin=W0, xmax=CUTOFF * scalingE,\
# 					ymin=0, ymax=6.0,\
# 					path=dipMomentSpecFigurePath + "/shells/" + str(4).zfill(2) + "/groupDelay.png", \
# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		# # 5
#
# 		dipoleShell = dipoleMoment[1, :] \
# 					+ dipoleMoment[4, :] \
# 					+ dipoleMoment[5, :]
#
# 		spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.pad(internal.d2fdx2(dipoleShell, dt), N_pad * N_t, mode='constant'))))
# 		spec = np.power(np.absolute(spectrum), 2.0)
# 		spectrumPhase = np.unwrap(np.angle(spectrum))
#
# 		dDipoleF      = np.pad(dipoleShell, N_pad * N_t, mode='constant')
# 		dDipole       = np.fft.fftshift(np.fft.fft(np.fft.fftshift(t_pad * dDipoleF)))
# 		Dipole   	  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dDipoleF)))
#
# 		groupDelay    = (dDipole + 1j * internal.dfdx(np.absolute(Dipole), dx=dw)) / (Dipole + 1E-6j)
# 		groupDelay	  = internal.moving_average(groupDelay * spectrumTotal, AVG_WIDTH) / internal.moving_average(spectrumTotal, AVG_WIDTH)
#
# 		plib.linearPlot (w * scalingE, spec,\
# 						 xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Intensity (a.u.)', \
# 						 xmin=W0, xmax=CUTOFF * scalingE, \
# 						 ymin=MINI, ymax=MAXI, \
# 						 path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/spectrum.png", \
# 						 log=True, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, spectrumPhase, \
# 					xlabel=r'$\omega$ (eV)', ylabel=r'Spectral Phase (rad.)', \
# 					xmin=W0, xmax=CUTOFF * scalingE,\
# 					path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/phase.png", \
# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# 		plib.linearPlot (w * scalingE, groupDelay.real * scalingT, \
# 					xlabel=r'$\omega$ (eV)', ylabel=r'Group Delay (fs)', \
# 					xmin=W0, xmax=CUTOFF * scalingE,\
# 					ymin=0, ymax=6.0,\
# 					path=dipMomentSpecFigurePath + "/shells/" + str(5).zfill(2) + "/groupDelay.png", \
# 					log=False, logx=False, figsize=plib.cm2inch(14.5, 8.96), dot=False)
#
# print("")
# print("  6. Program Finished")
# print("")
