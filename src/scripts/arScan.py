from importlib.machinery import SourceFileLoader
import numpy as np
from phys import internal
from phys import constants
from phys import plotLib as plib
from phys import sfalib as sfa
import scipy.signal as sig
import scipy.interpolate as sin

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

internal.printFigureUpdate ()

params = SourceFileLoader("params", "./scripts/phys/parseParams.py").load_module()

SIM_DATE = int(params.SIM_DATE[1 : -1])
SIM_IDX  = int(params.SIM_INDEX[1 : -1])

TD_SIM   = int(params.TD_SIMULATION[1 : -1])
IN_SITU  = int(params.IN_SITU[1 : -1])

N_r   	 = int(params.N_r[1 : -1])
N_t   	 = int(params.N_t[1 : -1])
N_l   	 = int(params.N_l[1 : -1])

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

eigenValuesPath = "./workingData/eigenStates/active/" + str(targetID).zfill(3) + "/eigenValues.bin"
eigenValues     = np.fromfile(eigenValuesPath, dtype=np.complex).real

UNITS_T  = 1
UNITS_E  = 1

if UNITS_T == 1:
	scalingT = 1.0 / constants.femtosecondAU
else:
	scalingT = 1.0

if UNITS_E == 1:
	scalingE = constants.electronVoltAU
else:
	scalingE = 1.0

tau_filter = 0.7 * tau0

MAX_L_ANAL = N_l
CUTOFF = 80.0 / 27.2114
W0 	   = 0
PLOT_HO = False
MINI = 1E-20
MAXI = 1E2
N_pad = 2

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

t   	   = np.fromfile(meshFilePath + "/time.bin")
dt  	   = t[1] - t[0]
dw 		   = 2.0 * np.pi * np.fft.fftfreq(N_t, dt)

filter 	   = sig.windows.kaiser(N_t, beta=14, sym=False)
filter 	   = np.power(np.sin(np.pi * (t - t[0]) / (t[-1] - t[0])), 2.0)
filter 	   = np.exp(- np.power(1.6 * (t  / tau_filter), 6.0))

occ = np.fromfile("./workingData/stateParameters/occ_active.bin", dtype=np.intc)

SIM_IDX = 4

SIM_DATE = 20210503

I = np.linspace(0.5, 2.0, 30, endpoint=True)
I = np.append(I, [2.75, 3.5])
for sdx in range(32):
	print(SIM_DATE, SIM_IDX)
	if (sdx == 8):
		SIM_DATE = 20210504
		SIM_IDX = 0

	outputPath 			= "../output/" + str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5)

	meshFilePath 		= outputPath + "/static/mesh"
	ksoFilePath  		= outputPath + "/static/kohnShamOrbitals"
	dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleMoment"
	dipoleMomentPath 	= outputPath + "/timeDependent/observables/dipoleAcceleration"
	vharFilePath		= outputPath + "/timeDependent/observables/vhar"
	electricFieldPath   = outputPath + "/timeDependent/field"


	omega0 = np.sqrt(I[sdx] / 2.75) * 1.55 / 27.2114
	tau_filter = 0.7 * 2.0 * np.pi / omega0

	t   	   = np.fromfile(meshFilePath + "/time.bin")
	dt  	   = t[1] - t[0]

	dw 		   = 2.0 * np.pi * np.fft.fftfreq(N_t, dt)

	filter 	   = sig.windows.kaiser(N_t, beta=14, sym=False)
	filter 	   = np.power(np.sin(np.pi * (t - t[0]) / (t[-1] - t[0])), 2.0)
	filter 	   = np.exp(- np.power(1.6 * (t  / tau_filter), 6.0))

	N_t_pad = (2 * N_pad + 1) * N_t
	t_pad   = np.linspace(- 0.5 * N_t_pad, 0.5 * N_t_pad - 1, N_t_pad) * dt
	w       = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t_pad, dt))
	dw      = w[1] - w[0]


	dipoleMoment = np.zeros((N_kso, N_t), dtype=float)

	for ksodx in range(N_kso):
		for ldx in range(MAX_L_ANAL):
			if (ldx <= MAX_L_ANAL):
				dipoleMomentTemp = np.fromfile(outputPath + "/timeDependent/observables/dipoleMoment/scan" \
												+ str(0).zfill(3) + "/kso" + str(ksodx).zfill(3) \
												+ "/dipL" + str(ldx).zfill(3) + ".bin", dtype=complex)
				dipoleMoment[ksodx, :] += dipoleMomentTemp.real

		dipoleMoment[ksodx, :] *= occ[ksodx] * filter

	totalDipole = np.sum(dipoleMoment[:, :], axis=0)

	# Total

	totalDipole   = np.pad(internal.d2fdx2(totalDipole, dt), N_pad * N_t, mode='constant')
	spectrum      = np.fft.fftshift(np.fft.fft(np.fft.fftshift(totalDipole)))
	spectrumTotal = np.power(np.absolute(spectrum), 2.0)


	np.save("./arScanData/w" + str(sdx).zfill(3) + ".npy", w * scalingE)
	np.save("./arScanData/s" + str(sdx).zfill(3) + ".npy", spectrumTotal)

	SIM_IDX += 1
