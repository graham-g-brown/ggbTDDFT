import numpy as np
import matplotlib.pylab as plt
import matplotlib.font_manager as font_manager

SMALL_SIZE  = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10
FIG_DPI     = 300

font_path = './scripts/fonts/Helvetica.ttc'
prop = font_manager.FontProperties(fname=font_path)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.switch_backend('agg')

N_scan = 256

td = np.linspace(- 4.0, 4.0, N_scan)

SPECTRA = np.load("./SPECTRA.npy")
w       = np.load("./w.npy")

SPECTRA = SPECTRA[:, w > 20.0 / 27.2114]
w = w[w > 20.0 / 27.2114]

W, TD = np.meshgrid(w, td)

SPECTRA /= np.max(SPECTRA)

fig, ax = plt.subplots(1, 1, figsize=(160 / 25.4, 60 / 25.4))
im = ax.pcolormesh(W, TD, np.power(np.absolute(SPECTRA), 2.0), cmap="hot", vmin=0.0, vmax=1.0)
ax.set_xlabel("Energy (eV)")
ax.set_xlim(25.0 / 27.2114, w[-1])
ax.set_ylabel("Delay (opt. cycles)")
fig.colorbar(im)
plt.savefig("./spectrogram.png", dpi=1200, bbox_inches="tight")
plt.close(fig)

SPECTRA *= np.exp(- np.power(TD, 2.0))
