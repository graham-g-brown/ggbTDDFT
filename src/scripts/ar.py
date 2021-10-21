from importlib.machinery import SourceFileLoader
import numpy as np
from phys import internal
from phys import constants
from phys import plotLib as plib
from phys import sfalib as sfa
import scipy.signal as sig
from matplotlib.colors import LogNorm
import scipy.interpolate as sin
import matplotlib.pylab as plt
import scipy.signal as sig
import matplotlib as mpl

mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])

N = 32

startDX = 0

w = np.load("./arScanData/w000.npy")
i = np.linspace(0.5, 0.5 + (N - 1) * 0.05, N, endpoint=True)
i = np.append(i, [2.75, 3.5])
W_MIN = 40.
W_MAX = 60.
S = np.zeros((N, np.size(w[np.logical_and(w > W_MIN, w < W_MAX)])))

argmin = np.argmin(np.absolute(w - 50.))

for k in range(N):
	S[k, :] = np.load("./arScanData/s" + str(k).zfill(3) + ".npy")[np.logical_and(w > W_MIN, w < W_MAX)]

W, I = np.meshgrid(w[np.logical_and(w > W_MIN, w < W_MAX)], i)
w = w[np.logical_and(w > W_MIN, w < W_MAX)]
argmin = np.argmin(np.absolute(w - W_MIN))
minE = np.zeros(N)

for k in range(N):
	S[k, :] = S[k, :] / np.sum(S[k, :])
	print(np.min(S[k, :]))
	minEdx = np.argmin(np.absolute(w - W_MIN)) + np.argmin(S[k, np.logical_and(w > W_MIN, w < W_MAX)])
	minE[k] = minEdx

fig = plt.figure()
plt.pcolormesh(W[startDX : N, :], I[startDX : N, :], S[startDX : N, :], cmap='gist_heat', shading='gouraud', norm=LogNorm(vmin=0.9*S.min(), vmax=S.max()))
plt.plot(w[minE.astype(np.int)][startDX : N], i[startDX : N])
plt.colorbar()
plt.xlim(W_MIN, W_MAX)
plt.xlabel("Energy (eV)")
plt.ylabel(r'Intensity ($10^{14}$ W cm$^{-2}$)')
plt.savefig("Scan.png", dpi=300)
plt.close(fig)

Z = [[0,0],[0,0]]
levels = range(0,N-1,1)
CS3 = plt.contourf(Z, i, cmap=mymap)
plt.clf()
f = np.zeros(N)
fig = plt.figure()
for k in range(N - 1, startDX - 1, - 1):
	print(N - 1 - k)
	r = (float(k))/(N-1)
	g = 0
	b = 1-r
	plt.semilogy(w, np.power(10, 0.5 * (N - 1 - k)) * S[k, :], color=(r,g,b))

for k in range(N - 1, startDX- 1, - 1):
	f[k] = np.power(10, 0.5 * (N - 1 - k)) * S[k, int(minE[k])]
print(f)
# f[f < 1E-20] = np.nan
plt.semilogy(w[minE[startDX : N].astype(np.int)], f[startDX : N], color="tab:red")
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity (arb.)")
plt.ylim(1E-8, 1E20)
plt.xlim(W_MIN, W_MAX)
cbar = plt.colorbar(CS3)
cbar.set_label(r'Driving Field Intensity ($10^{14}$ W/cm$^2$)')
plt.tight_layout()
plt.savefig("Scan2.png", dpi=300)
plt.close(fig)


fff = w[minE[startDX : N].astype(np.int)]
# fff[2] =
fig = plt.figure()
plt.plot(i[startDX : N], fff, color="tab:red")
plt.ylabel("Energy (eV)")
plt.xlabel("Intensity (arb.)")
# plt.xlim(W_MIN, W_MAX)
plt.tight_layout()
plt.savefig("Scan3.png", dpi=300)
plt.close(fig)
