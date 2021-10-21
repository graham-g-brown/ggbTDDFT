import numpy as np
import scipy as sp
from datetime import date
import os
from phys import groundState as gs
from phys import internal
from phys import plotLib as plib
import sys

attosecondAU   = 0.041341374575751
femtosecondAU  = 41.3413745758
cAU            = 137.035999074
intensityAU    = 3.509E16
electronVoltAU = 27.2114

atomicUnit   = 0
electronVolt = 1
attosecond   = 2
femtosecond  = 3
angstrom     = 4

H = "H"
h = "H"
Hydrogen = "H"
hydrogen = "H"

He = "He"
he = "He"
Helium = "He"
helium = "he"

Ne = "Ne"
ne = "Ne"
Neon = "Ne"
neon = "Ne"

Ar = "Ar"
ar = "Ar"
Argon = "Ar"
argon = "Ar"

Kr = "Kr"
kr = "Kr"
Krypton = "Kr"
krypton = "Kr"

Xe = "Xe"
xe = "Xe"
Xenon = "Xe"
xenon = "Xe"

DFT = 0
HF  = 1

CN  = 0
CN2 = 1
CN3 = 2
EIG = 3

NOXC = 0
ALDA = 1

MONOMIAL     = 0
ABC_MANOLOPOULOS = 1
DOUBLE_SINH  = 2
DOUBLE_EXP   = 3

EV = 1
FS = 1
AS = 2
AU = 0

internal.welcomeScreen ()

with open("init") as f:
    for line in f:
        exec(line)

"""
Set Simulation Parameters:

If a parameter was not set in the "init" file, it is set to a default value.
"""

try:
	UNITS_ENERGY
except NameError:
	UNITS_ENERGY = electronVolt

try:
	USE_C60
except NameError:
	USE_C60 = 0

try:
	USE_MULLER
except NameError:
	USE_MULLER = 0

try:
	GS_FIGS
except NameError:
	GS_FIGS = False

try:
	UNITS_TIME
except NameError:
	UNITS_TIME = attosecond

try:
	THETA
except NameError:
	THETA = 30E-3

COS_THETA = np.cos(THETA)

try:
	UNITS_LENGTH
except NameError:
	UNITS_LENGTH = atomicUnit

try:
	FROM_SCRATCH
except NameError:
	FROM_SCRATCH = 1

try:
	GS_FIGS
except NameError:
	GS_FIGS = False

try:
	METHOD
except NameError:
	METHOD = DFT

try:
	FUNCTIONAL
except NameError:
	FUNCTIONAL = ALDA

try:
	TD_SIMULATION
except NameError:
	TD_SIMULATION = 0

try:
	IN_SITU
except NameError:
	IN_SITU = 0

if (IN_SITU == 0):
	PERTURBATION_AMPLITUDE = 0.0
else:
	PERTURBATION_AMPLITUDE = 0.01

try:
	N_r
except NameError:
	N_r = 64

try:
	N_scan
except NameError:
	N_scan = 64

dphi = 2.0 * np.pi / N_scan

if (IN_SITU == 0):
	N_scan = 1
try:
	zeta
except NameError:
	zeta = 0.20

try:
	ETA_MASK
except NameError:
	ETA_MASK = 1E-3

try:
	N_l
except NameError:
	N_l = 32

try:
	EXP_METHOD
except NameError:
	EXP_METHOD = EIG

try:
	KSO_MIN
except NameError:
	KSO_MIN = 0

try:
	USE_HARTREE
except NameError:
	USE_HARTREE = 1

try:
	HARTREE_FACTOR
except NameError:
	HARTREE_FACTOR = 1.0

# Target Parameters

try:
	targetString = atom
except NameError:
	targetString = atom

targetID = internal.getTargetID (targetString)

# Time Propagation Parameters

try:
	N_t
except NameError:
	N_t = 2048

try:
	dt
except NameError:
	dt = 10.0

dt *= attosecondAU

try:
	I0
except NameError:
	I0 = 1E10

try:
	tau0 *= femtosecondAU
except NameError:
	tau0 = 0.25 * N_t * dt

try:
	CEP0
except NameError:
	CEP0 = np.pi / 2.0

try:
	ETA_MIX
except NameError:
	ETA_MIX = 1E-1

try:
	N_har
except NameError:
	N_har = 2


I0 = I0 / intensityAU
E0 = np.sqrt(I0)

if (N_scan > 1):
	dE = 1.0 / (N_scan - 1)
else:
	dE = 1.0

try:
	omega0 = omega0
except NameError:
	CO = 70. / 27.2114
	Ip = 12.8 / 27.2114

	omega0 = (0.890224690738243 * np.sqrt(I0)) / np.sqrt(CO - Ip)

T0 = 2.0 * np.pi / omega0
TD = np.linspace(- T0, T0, N_scan) * 4.0

try:
	ABC_R_SCALE
except NameError:
	ABC_R_SCALE = 1.0

try:
	R_MASK
except NameError:
	R_MASK = np.max(np.array([ABC_R_SCALE * np.sqrt(I0) / np.power(omega0, 2.0), 16.0]))

R_MAX = 1.50 * R_MASK

try:
	ABC_TYPE
except NameError:
	ABC_TYPE = ABC_MANOLOPOULOS

try:
	CEP0
except NameError:
	CEP0 = 0

"""
Initialize field parameters
"""

"""
Initialize Mesh
"""

r, drdx, d, T, N_g, x, P, D1 = internal.generateGaussLobattoGrid (N_r, R_MASK, R_MAX, zeta)

maskR = internal.getABCR(r, R_MASK, R_MAX, 1)

"""
Determine Target Ground State Properties
"""

Z, n, l, m, occ, N_kso, stringEC = internal.getElectronConfiguration (targetID, targetString)

eigenStates, eigenValues, v0, vscf, N_active, l_active, m_active, occ_active, density, densityFrozen, N_m = gs.getGroundState (r, x, T, D1, maskR, P, drdx, ETA_MIX, targetID, n, l, m, occ, N_kso, N_l, KSO_MIN, FROM_SCRATCH, METHOD, USE_MULLER, USE_C60, FIGS=GS_FIGS)

"""
Initialize Time-Dependent Parameters
"""

t = np.linspace(- 0.5 * N_t, 0.5 * N_t - 1, N_t) * dt

"""
Generate Propagation Matrices
"""

# Propagators involving radial derivatives

PROPAGATOR_FILE_PATHS = internal.propagatorsR (r, T, v0, vscf, maskR, l_active, N_r, N_l, dt, EXP_METHOD, R_MASK)

# Propagators diagonal in r, but not diagonal in l

for k in range(1, N_har):
	internal.propagatorsL (m_active, N_l, k)

# Print estimated cutoff, radial extent of trajectories, etc...

internal.printSimEstimate (E0, omega0, eigenValues, electronVoltAU, R_MAX, R_MASK, N_l)

# 2. Write Data to Disk and Write Parameters File

WORKING_DATA_MESH_R    = "./workingData/mesh/r.bin"
WORKING_DATA_MESH_DRDX = "./workingData/mesh/drdx.bin"
WORKING_DATA_MESH_P    = "./workingData/mesh/P.bin"
WORKING_DATA_MESH_E    = "./workingData/mesh/E.bin"
WORKING_DATA_MESH_TIME = "./workingData/mesh/time.bin"
WORKING_DATA_MESH_D1 = "./workingData/mesh/D1.bin"
WORKING_DATA_MESH_TD = "./workingData/mesh/TD.bin"

r_tf     = r.astype(np.float64)
drdx_tf  = drdx.astype(np.float64)
P_tf     = P.astype(np.float64)
time_tf  = t.astype(np.float64)
D1_tf    = D1.astype(np.float64)
TD_tf    = TD.astype(np.float64)

r_tf.tofile(WORKING_DATA_MESH_R)
drdx_tf.tofile(WORKING_DATA_MESH_DRDX)
P_tf.tofile(WORKING_DATA_MESH_P)
time_tf.tofile(WORKING_DATA_MESH_TIME)
D1_tf.tofile(WORKING_DATA_MESH_D1)
TD_tf.tofile(WORKING_DATA_MESH_TD)

eigenStates_tf = eigenStates.astype(np.complex128)
eigenValues_tf = eigenValues.astype(np.complex128)

occ_tf         = occ_active.astype(np.intc)
l_tf 		   = l_active.astype(np.intc)
m_tf 		   = m_active.astype(np.intc)
densityFrozen_tf = densityFrozen.astype(np.float64)

eigenValues_tf.tofile("./workingData/eigenStates/active/" + str(targetID).zfill(3) + "/eigenValues.bin")

for ldx in range(int(np.min(l_active)), int(np.max(l_active)) + 1):
	temp_tf = eigenStates[ldx, :, :].astype(np.complex128)
	temp_tf.tofile("./workingData/eigenStates/active/" + str(targetID).zfill(3) + "/eigenStatesL" + str(ldx).zfill(3) + ".bin")

l_tf.tofile("./workingData/stateParameters/l_active.bin")
m_tf.tofile("./workingData/stateParameters/m_active.bin")
occ_tf.tofile("./workingData/stateParameters/occ_active.bin")
densityFrozen_tf.tofile("./workingData/stateParameters/densityFrozen.bin")

# 3. Set Output Directories

date = date.today()
date_today = int(str(int(date.year)) + str(int(date.month)).zfill(2) + str(int(date.day)).zfill(2))

localOutputFilePath = "../output"

idx = 0
con = True

localDateFilePath = localOutputFilePath + "/" + str(date_today)

if not os.path.exists(localDateFilePath):
	os.mkdir(localDateFilePath)

while (con):
	if not os.path.exists(localOutputFilePath + "/" + str(date_today) + "/" + str(idx).zfill(5)):
		con = False
	else:
		idx += 1

# Define Simulation Output Directories

localSimIndexFilePath = localOutputFilePath + "/" + str(date_today) + "/" + str(idx).zfill(5)

localStaticFilePath = localSimIndexFilePath + "/static"
localMeshFilePath = localStaticFilePath + "/mesh"

localTimeDependentFilePath = localSimIndexFilePath + "/timeDependent"
localTimeDependentKohnShamOrbitalsFilePath = localTimeDependentFilePath + "/kohnShamOrbitals"
localTimeDependentFieldFilePath = localTimeDependentFilePath + "/field"
localTimeDependentObservablesFilePath = localTimeDependentFilePath + "/observables"
localTimeDependentDipoleMomentFilePath = localTimeDependentObservablesFilePath + "/dipoleMoment"
localTimeDependentDipoleAccelerationFilePath = localTimeDependentObservablesFilePath + "/dipoleAcceleration"
localTimeDependentVHARFilePath = localTimeDependentObservablesFilePath + "/vhar"
localTimeDependentVEXCFilePath = localTimeDependentObservablesFilePath + "/vexc"

path = localSimIndexFilePath
if not os.path.exists(path):
	os.mkdir(path)

os.system("cp ./init " + localSimIndexFilePath + "/init")

path = localStaticFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localMeshFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localTimeDependentFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localTimeDependentKohnShamOrbitalsFilePath
if not os.path.exists(path):
	os.mkdir(path)

for ldx in range(0, 8):
	path = localTimeDependentKohnShamOrbitalsFilePath + "/" + str(ldx).zfill(3)
	if not os.path.exists(path):
		os.mkdir(path)

path = localTimeDependentObservablesFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localTimeDependentDipoleMomentFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localTimeDependentDipoleAccelerationFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localTimeDependentVHARFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localTimeDependentVEXCFilePath
if not os.path.exists(path):
	os.mkdir(path)
path = localTimeDependentFieldFilePath
if not os.path.exists(path):
	os.mkdir(path)


for scandx in range(N_scan + 1):

	path = localTimeDependentDipoleMomentFilePath + "/scan" + str(scandx).zfill(3)
	if not os.path.exists(path):
		os.mkdir(path)

	path = localTimeDependentDipoleAccelerationFilePath + "/scan" + str(scandx).zfill(3)
	if not os.path.exists(path):
		os.mkdir(path)

	path = localTimeDependentVHARFilePath + "/scan" + str(scandx).zfill(3)
	if not os.path.exists(path):
		os.mkdir(path)

	for ksodx in range(N_active):

		path = localTimeDependentDipoleMomentFilePath + "/scan" + str(scandx).zfill(3) + "/kso" + str(ksodx).zfill(3)
		if not os.path.exists(path):
			os.mkdir(path)

		path = localTimeDependentDipoleAccelerationFilePath + "/scan" + str(scandx).zfill(3) + "/kso" + str(ksodx).zfill(3)
		if not os.path.exists(path):
			os.mkdir(path)

		path = localTimeDependentVHARFilePath + "/scan" + str(scandx).zfill(3) + "/kso" + str(ksodx).zfill(3)
		if not os.path.exists(path):
			os.mkdir(path)

def writeParametersFile(filepath):
	f = open(filepath, "w")
	f.write("// Simulation Parameters \n")
	f.write("\n")
	f.write("#define SIM_DATE (%d)  \n" % date_today)
	f.write("#define SIM_INDEX (%d)  \n" % idx)
	f.write("\n")
	f.write("#define TD_SIMULATION (%d)\n" % TD_SIMULATION)
	f.write("#define IN_SITU (%d)\n" % IN_SITU)
	f.write("\n")
	f.write("// System Settings \n")
	f.write("\n")
	f.write("#define CUDA_DEVICE (0)   \n")
	f.write("#define MTPB (32)   \n")
	f.write("\n")
	f.write("#define attosecondAU (%.14E) \n" % attosecondAU)
	f.write("#define femtosecondAU (%.14E) \n" % femtosecondAU)
	f.write("#define cAU (%.14E) \n" % cAU)
	f.write("#define intensityAU (%.14E) \n" % intensityAU)
	f.write("#define electronVoltAU (%.14E) \n" % electronVoltAU)
	f.write("\n")
	f.write("\n")
	f.write("// Spatial and Temporal Grid Parameters \n")
	f.write("\n")
	f.write("// // Spatial Parameters \n")
	f.write("\n")
	f.write("#define N_r (%d) \n" % N_r)
	f.write("#define N_g (%d) \n" % N_g)
	f.write("#define zeta (%.15E) \n" % zeta)
	f.write("#define R_MAX (%.15E) \n" % R_MAX)
	f.write("\n")
	f.write("#define R_MASK (%.15E) \n" % R_MASK)
	f.write("#define ETA_MASK (%.15E) \n" % ETA_MASK)
	f.write("\n")
	f.write("#define N_l (%d) \n" % N_l)
	f.write("#define N_m (%d) \n" % N_m)
	f.write("\n")
	f.write("#define Z (%.15E) \n" % Z)
	f.write("#define L0 (%d) \n" % int(np.max(l_active)))
	f.write("#define targetID (%d) \n" % targetID)
	f.write("#define N_kso (%d) \n" % N_active)
	f.write("\n")
	f.write("#define N_t (%d) \n" % N_t)
	f.write("#define dt (%.15E) \n" % dt)
	f.write("\n")
	f.write("#define E0 (%.15E) \n" % E0)
	f.write("#define omega0 (%.15E) \n" % omega0)
	f.write("#define omegaX (%.15E) \n" % (13.0 / 27.2114))
	f.write("#define tau0 (%.15E) \n" % tau0)
	f.write("#define CEP0 (%.15E) \n" % CEP0)
	f.write("\n")
	f.write("#define PERTURBATION_AMPLITUDE (%.15E) \n" % PERTURBATION_AMPLITUDE)
	f.write("#define dphi (%.15E) \n" % dphi)
	f.write("#define COS_THETA (%.15E) \n" % COS_THETA)
	f.write("#define THETA (%.15E) \n" % THETA)
	f.write("\n")
	f.write("#define N_scan (%d) \n" % N_scan)
	f.write("\n")
	f.write("#define N_har (%d) \n" % N_har)
	f.write("#define USE_HARTREE (%d) \n" % USE_HARTREE)
	f.write("#define HARTREE_FACTOR (%.14E) \n" % HARTREE_FACTOR)
	f.write("#define FUNCTIONAL (%d) \n" % FUNCTIONAL)
	f.write("\n")
	f.write("#define WORKING_DATA_MESH_R \"" + WORKING_DATA_MESH_R + "\"\n")
	f.write("#define WORKING_DATA_MESH_P \"" + WORKING_DATA_MESH_P + "\"\n")
	f.write("#define WORKING_DATA_MESH_DRDX \"" + WORKING_DATA_MESH_DRDX + "\"\n")
	f.write("#define WORKING_DATA_MESH_E \"" + WORKING_DATA_MESH_E + "\"\n")
	f.write("#define WORKING_DATA_MESH_TIME \"" + WORKING_DATA_MESH_TIME + "\"\n")
	f.write("#define WORKING_DATA_MESH_D1 \"" + WORKING_DATA_MESH_D1 + "\"\n")
	f.write("#define WORKING_DATA_MESH_TD \"" + WORKING_DATA_MESH_TD + "\"\n")
	f.write("\n")
	for l in range(N_l):
		f.write("#define WORKING_DATA_PROPAGATOR_L_" + str(l).zfill(3) + " \"" + PROPAGATOR_FILE_PATHS[l] + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_OUTPUT_MESH_R \"" + localMeshFilePath + "/r.bin\"\n")
	f.write("#define LOCAL_OUTPUT_MESH_DRDX \"" + localMeshFilePath + "/drdx.bin\"\n")
	f.write("#define LOCAL_OUTPUT_MESH_TIME \"" + localMeshFilePath + "/time.bin\"\n")
	f.write("#define LOCAL_OUTPUT_MESH_E \"" + localTimeDependentFieldFilePath + "/E.bin\"\n")
	f.write("#define LOCAL_OUTPUT_VHAR_VHAR \"" + localTimeDependentVHARFilePath + "/vhar.bin\"\n")
	f.write("#define LOCAL_OUTPUT_VHAR_VHAR0 \"" + localTimeDependentVHARFilePath + "/vhar0.bin\"\n")
	f.write("#define LOCAL_OUTPUT_VEXC_VEXC \"" + localTimeDependentVEXCFilePath + "/vexc.bin\"\n")
	f.write("#define LOCAL_OUTPUT_VEXC_VEXC0 \"" + localTimeDependentVEXCFilePath + "/vexc0.bin\"\n")
	f.write("\n")
	f.write("// Load Library Files \n")
	f.write("\n")
	f.write("#include \"./subFunctions/system/h_checkCUDADevice.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/output/h_writeMesh.cuh\"\n")
	f.write("#include \"./subFunctions/output/h_writeDipoleMoment.cuh\"\n")
	f.write("#include \"./subFunctions/output/h_writeHartreePotential.cuh\"\n")
	f.write("#include \"./subFunctions/output/h_writeExchangePotential.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/mesh/h_initialiseMesh.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/field/h_initialiseField.cuh\"\n")
	f.write("#include \"./subFunctions/field/h_resetField.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/potentials/hartree/angularCoefficients.cuh\"\n")
	f.write("#include \"./subFunctions/potentials/hartree/hartreePotentialDensity.cuh\"\n")
	f.write("#include \"./subFunctions/potentials/hartree/h_calculateHartreePotential.cuh\"\n")
	f.write("#include \"./subFunctions/potentials/exchangeCorrelation/h_calculateExchangeCorrelationPotential.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/timePropagation/h_initialisePropagators.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/h_initialiseTDKSOSet.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/observables/dipoleMoment.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/observables/dipoleAcceleration.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/externalFieldCoupling/lengthGauge/couplingS.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/externalFieldCoupling/lengthGauge/couplingP.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/externalFieldCoupling/lengthGauge/couplingD.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/externalFieldCoupling/lengthGauge/couplingF.cuh\"\n")
	f.write("#include \"./subFunctions/timePropagation/h_timePropagation.cuh\"\n")
	f.close()

writeParametersFile ("./params.cuh")

writeParametersFile (localSimIndexFilePath + "/params.cuh")
