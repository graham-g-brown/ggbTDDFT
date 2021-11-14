import numpy as np
import numpy.linalg as nla
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.colors
from matplotlib import rcParams
import scipy.interpolate as sin
import scipy.special as sps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.interpolate as sin

electronvoltAU = 0.036749405469679

COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dipoleSpec = np.load("./data/dipoleSpec.npy")
dipoleSpec0 = np.load("./data/dipoleSpec0.npy")
dipoleSpec0 = dipoleSpec0 / np.max(np.absolute(dipoleSpec0))
N_sim = 64

omega0 = np.load("./data/omega0.npy")[0]
w = np.load("./data/w.npy")

H = np.linspace(12, 46, 18).astype(int)

Hp   = H + 0.5
Hn   = H - 0.5

Hdx = np.zeros(np.size(H), dtype=int)
Hpdx = np.zeros(np.size(H), dtype=int)
Hndx = np.zeros(np.size(H), dtype=int)

for hdx in range(np.size(H)):
	Hdx[hdx] = np.argmin(np.absolute(H[hdx] * omega0 - w))
	Hpdx[hdx] = np.argmin(np.absolute(Hp[hdx] * omega0 - w))
	Hndx[hdx] = np.argmin(np.absolute(Hn[hdx] * omega0 - w))

filter = np.linspace(0.0, np.size(w) - 1, np.size(w))

dipoleSPEC = np.zeros_like(dipoleSpec)

RANGE = 100

phase = np.linspace(0.0, 2.0 * np.pi, N_sim)


for j in range(N_sim):
	for k in range(np.size(Hdx)):
		dipoleSPEC[j, :] += np.exp(- np.power((filter - Hdx[k]) / (0.250 * (Hpdx[k] - Hndx[k])), 32.0)) * (dipoleSpec[j, :] - dipoleSpec0[0, :])

minimizingPhaseDX = np.zeros(np.size(Hdx), dtype=int)

for k in range(np.size(Hdx)):

	if k == 0:

		b = Hdx[k]
		c = Hdx[k + 1]

		maxRdx = Hdx[k] + (Hdx[k + 1] - Hdx[k]) // 2
		minRdx = Hdx[k] - (Hdx[k + 1] - Hdx[k]) // 2
	elif k == np.size(Hdx) - 1:
		maxRdx = Hdx[k] + (Hdx[k] - Hdx[k - 1]) // 2
		minRdx = Hdx[k] - (Hdx[k] - Hdx[k - 1]) // 2 + 1

	else:
		a = Hdx[k - 1]
		b = Hdx[k]
		c = Hdx[k + 1]

		minRdx = Hdx[k - 1] + (Hdx[k] - Hdx[k - 1]) // 2 + 1
		maxRdx = Hdx[k] + (Hdx[k + 1] - Hdx[k]) // 2

	dipoleSPEC[:, minRdx : maxRdx] -= np.min(dipoleSPEC[:, minRdx : maxRdx])
	dipoleSPEC[:, minRdx : maxRdx] /= np.max(np.absolute(dipoleSPEC[:, minRdx : maxRdx]))
	minimizingPhaseDX[k] = 8 + np.argmin(np.sum(dipoleSPEC[8 : N_sim // 2, Hdx[k] - 128 : Hdx[k] + 128], axis=1))

N_int = 256

phase = np.linspace(0.0, 2.0 * np.pi, 65)

dipoleSPECNEW = np.zeros((65, np.size(w)))
dipoleSPECNEW[0 : 64, :] = dipoleSPEC[:, :]
dipoleSPECNEW[64, :] = dipoleSPEC[0, :]

PHASENEW = np.linspace(0.0, 2.0 * np.pi, 512)

SPECFINAL = np.zeros((512, np.size(w)))

for k in range(np.shape(dipoleSPEC)[1]):
	# print(k)
	f = sin.interp1d(phase, dipoleSPECNEW[:, k])
	SPECFINAL[:, k] = f(PHASENEW)

W, PHASE = np.meshgrid(w, PHASENEW)

fig, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 50 / 25.4))
ax = axes[0]
ax.semilogy(w / electronvoltAU, np.absolute(dipoleSpec[0, :]), linewidth=0.50)
ax.set_xlim(0, 72)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Intensity (arb.)")
ax = axes[1]
im = ax.pcolormesh(W / electronvoltAU, PHASE / np.pi, SPECFINAL[:,:], cmap='hot', vmin=0.00, vmax=1.0, shading='auto', rasterized = True)
ax.plot(w[Hdx] / electronvoltAU, phase[minimizingPhaseDX] / np.pi, color=(200 / 255.0, 33 / 255.0, 23 / 255.0), linewidth=0.50)
cb = fig.colorbar(im, pad=0.05)
cb.set_label("Intensity (norm.)")
cb.ax.tick_params()
ax.set_xlim(20.0, 72.0)
ax.set_ylim(0.0, 2.0)
ax.set_ylabel(r'Relative Phase ($\pi$-rad.)')
ax.set_xlabel("Energy (eV)")
plt.tight_layout()
plt.savefig("./figures/spectrogram.png", dpi=1200, bbox_inches="tight", format="PNG", transparent=False)
plt.close(fig)
