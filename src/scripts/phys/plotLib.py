import numpy as np
import numpy.linalg as nla
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.colors
from phys import internal
from matplotlib import rcParams
import matplotlib.colors
import scipy.interpolate as sin
import scipy.special as sps
from mpl_toolkits.axes_grid1 import make_axes_locatable

COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

mpl.rcParams['agg.path.chunksize'] = 1000000

cmap = 'gist_heat'

SMALL_SIZE  = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
FIG_DPI     = 300

# font_path = './scripts/fonts/Helvetica.ttc'
# prop = font_manager.FontProperties(fname=font_path)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.switch_backend('agg')


MIN_W = 50
MAX_W = 57

def dfdx(f):
	df = np.zeros_like(f)
	df[0] = 0.0
	df[1] = 0.0
	df[-1] = 0.0
	df[-2] = 0.0
	N = np.size(f)
	for k in range(2, N - 2):
		df[k] = 0.50 * (f[k + 1] - f[k - 1]) \

	return df

def cm2inch (*tupl):
	inch = 2.54
	if isinstance(tupl[0], tuple):
		return tuple(i/inch for i in tupl[0])
	else:
		return tuple(i/inch for i in tupl)

def padFunction(f, scale=18, pad=0.0):

	N = np.size(f)

	N_new = scale * N

	f_new = np.ones(N_new, dtype=f.dtype) * pad

	f_new[N_new // 2 - N // 2 : N_new // 2 + N // 2] = f[:]

	return f_new

def shortTimeFourierTransformFilter (t, t0, sigma):

	return np.exp(- np.power((t - t0) / sigma, 2.0))

def fourierTransform(f, w=None, w_min=None):

	if (w_min is None):
		return np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))
	else:
		spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))

		filter = np.power(np.sin(0.50 * np.pi * w / w_min), 4.0)

		spectrum[np.absolute(w) < w_min] *= filter[np.absolute(w) < w_min]

		return filter * spectrum


def shortTimeFourierTransform (f, t):

	N = np.size(t)
	dt = t[1] - t[0]
	w = np.linspace(- 0.50 * N, 0.50 * N - 1, N) * 2.0 * np.pi / dt / N * 27.212

	t = t / attosecond_au / 1000.0

	t_plot = t[t > - 3.0]
	t_plot = t_plot[t_plot < 3.0]

	N_plot = np.size(t_plot)

	sdx = np.argmin(np.absolute(t + 3.0))

	reduction = 8
	sigma = 8000 * attosecond_au

	STFT = np.zeros((N_plot, N))

	for tdx in range(N_plot):
		STFT[tdx, :] = np.log10(np.power(np.absolute(np.fft.fftshift(np.fft.fft(np.fft.fftshift(f * shortTimeFourierTransformFilter(t, t[sdx + tdx], t[128 * reduction] - t[0]))))), 2.0) + 1E-30)

	return STFT, t_plot

def applyFourierTransformWindow (f):

	N = np.size(f)

	idx = np.linspace(- 0.5 * N, 0.5 * N - 1, N)

	filter = np.ones(N)

	x0 = idx[0]
	xf = idx[-1]
	L = (1 + xf - x0)

	filter = np.power(np.sin((idx - x0) / L * np.pi), 2.0)

	return f * filter

def contourPlot (f, X, Y, x, y, fmin=None, fmax=None, xmin=None, xmax=None, ymin=None, ymax=None, path=None, xlabel=None, ylabel=None, log=False, T=False, figsize=cm2inch(14.5, 8.96)):

	if xmin is None:
		yMindx = 0
	else:
		yMindx = np.argmin(np.absolute(y - ymin))

	if xmax is None:
		yMaxdx = -1
	else:
		yMaxdx = np.argmin(np.absolute(y - ymax))

	if ymin is None:
		xMindx = 0
	else:
		xMindx = np.argmin(np.absolute(x - xmin))

	if ymax is None:
		xMaxdx = -1
	else:
		xMaxdx = np.argmin(np.absolute(x - xmax))

	print(y[yMindx], y[yMaxdx])

	if (log):

		figure = plt.figure(figsize=figsize)

		if (fmax is None):
			fmax = np.max(np.absolute(f[xMindx : xMaxdx, yMindx : yMaxdx]))
		if (fmin is None):
			fmin = np.min(np.absolute(f[xMindx : xMaxdx, yMindx : yMaxdx]))

		if (T):
			plt.pcolormesh(Y[xMindx : xMaxdx, yMindx : yMaxdx].T,\
	                   	   X[xMindx : xMaxdx, yMindx : yMaxdx].T, \
	                   	   f[xMindx : xMaxdx, yMindx : yMaxdx].T / fmax, cmap=cmap, vmin=fmin, vmax=fmax, shading='auto', norm=colors.LogNorm(vmin=fmin, vmax=fmax))
		else:
			plt.pcolormesh(X[xMindx : xMaxdx, yMindx : yMaxdx],\
	                   	   Y[xMindx : xMaxdx, yMindx : yMaxdx], \
	                   	   f[xMindx : xMaxdx, yMindx : yMaxdx], cmap=cmap, ymax=140., vmin=fmin, vmax=fmax, shading='auto')

	else:
		figure = plt.figure(figsize=figsize)

		if (fmax is None):
			fmax = np.max(np.absolute(f[xMindx : xMaxdx, yMindx : yMaxdx]))
		if (fmin is None):
			fmin = - fmax

		if (T):
			plt.pcolormesh(Y[xMindx : xMaxdx, yMindx : yMaxdx].T,\
	                   	   X[xMindx : xMaxdx, yMindx : yMaxdx].T, \
	                   	   f[xMindx : xMaxdx, yMindx : yMaxdx].T / fmax, cmap=cmap, vmin=fmin, vmax=fmax, shading='auto')
		else:
			plt.pcolormesh(X[xMindx : xMaxdx, yMindx : yMaxdx],\
	                   	   Y[xMindx : xMaxdx, yMindx : yMaxdx], \
	                   	   f[xMindx : xMaxdx, yMindx : yMaxdx], cmap=cmap, vmin=fmin, vmax=fmax, shading='auto')

		plt.colorbar(format='%.2E')

	if (xlabel is not None):
		plt.xlabel(xlabel, fontsize=12, color=COLOR)
	if (ylabel is not None):
		plt.ylabel(ylabel, fontsize=12, color=COLOR)
	if (path is None):
		plt.show()
	else:
		plt.savefig(path, transparent=True, bbox_inches="tight", dpi=FIG_DPI)
	plt.close(figure)

def linearPlot(x, f, x2=None, f2=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, path=None, log=False, logx=False, figsize=cm2inch(14.5, 8.96), dot=False, title=None, minorTicks=False, thesisStyle=False, xticks=None):

	if thesisStyle:
		figsize = cm2inch(11.80737, 7.37960625)
		FONT_SIZE = 8
		COLOR = 'white'
		mpl.rcParams['text.color'] = COLOR
		mpl.rcParams['axes.labelcolor'] = COLOR
		mpl.rcParams['xtick.color'] = COLOR
		mpl.rcParams['ytick.color'] = COLOR
		FILL_BETWEEN = False
	else:
		FONT_SIZE = 8
		FILL_BETWEEN = False

	if (xmin is None):
		if logx:
			xmin = 0.9 * np.min(np.absolute(x))
		else:
			xmin = x[0]
			xminDX = 0
	else:
		xminDX = np.argmin(np.absolute(x - xmin))
	if (xmax is None):
		if logx:
			xmax = 10.0 * np.max(np.absolute(x))
		else:
			xmax = x[-1]
			xmaxDX = -1
	else:
		xmaxDX = np.argmin(np.absolute(x - xmax))
	if (ymin is None):
		if (log):
			ymin = 1.1 * np.min(np.absolute(f[xminDX : xmaxDX]))
		else:
			ymin = np.min(f[xminDX : xmaxDX])
			if ymin < 0.0:
				ymin *= 1.1
			else:
				ymin *= 0.9
	if (ymax is None):
		if (log):
			ymax = 10.0 * np.max(np.absolute(f[xminDX : xmaxDX]))
		else:
			ymax = np.max(f[xminDX : xmaxDX])
			if ymax < 0.0:
				ymax *= 0.9
			else:
				ymax *= 1.1

	if (np.absolute(ymin - ymax) < 1E-16):
		ymin = -1.0
		ymax = 1.0
	if (figsize is None):
		figure = plt.figure(figsize=cm2inch(14.5, 8.96))
	else:
		figure = plt.figure(figsize=figsize)
	if (log):
		plt.semilogy(x, f, linewidth=1.0)
		plt.fill_between(x, f, alpha=0.50)
		if (f2 is not None):
			plt.semilogy(x2, f2, linewidth=1.0)
	elif (logx):
		plt.semilogx(x, f, linewidth=1.0, color="tab:red")
		if (FILL_BETWEEN):
			plt.fill_between(x, f, alpha=0.50)
	else:
		if dot:
			plt.plot(x, f, linewidth=1.0)
		else:
			plt.plot(x, f, linewidth=1.0)
			if (f2 is None and FILL_BETWEEN):
				jjk = 1 # plt.fill_between(x, f, alpha=0.50)
			else:
				jjk = 1 # plt.plot(x2, f2)
	if (minorTicks):
		plt.grid(which='both')
	else:
		plt.grid()
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	if (xticks is not None):
		plt.xticks(xticks)
	if (title is not None):
		plt.title(title, fontsize=12)
	if (xlabel is not None):
		plt.xlabel(xlabel, fontsize=8)
	if (ylabel is not None):
		plt.ylabel(ylabel, fontsize=8)
	if (path is None):
		plt.show()
	else:
		plt.tight_layout()
		plt.savefig(path, format="PNG", bbox_inches="tight", transparent=True, dpi=FIG_DPI)
		# plt.savefig(path, format="PDF", bbox_inches="tight")
	plt.close(figure)

def scatterPlot(x, f, t, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, path=None, log=False, figsize=cm2inch(14.5, 8.96), dot=False):

	if (xmin is None):
		xmin = x[0]
	if (xmax is None):
		xmax = x[-1]
	if (ymin is None):
		if (log):
			ymin = 0.9 * np.min(np.absolute(f))
		else:
			ymin = - 1.1 * np.max(np.absolute(f))
	if (ymax is None):
		if (log):
			ymax = 10.0 * np.max(np.absolute(f))
		else:
			ymax = 1.1 * np.max(np.absolute(f))
	if (figsize is None):
		figure = plt.figure(figsize=cm2inch(14.5, 8.96))
	else:
		figure = plt.figure(figsize=figsize)

	plt.scatter(x, f, c=t, s=1)
	plt.grid()
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	if (xlabel is not None):
		plt.xlabel(xlabel, fontsize=18)
	if (ylabel is not None):
		plt.ylabel(ylabel, fontsize=18)
	if (path is None):
		plt.show()
	else:
		plt.tight_layout()
		plt.savefig(path, format="PNG", bbox_inches="tight", transparent=True, dpi=FIG_DPI)
	plt.close(figure)

def polarContour (f, r, L, M, title='', rmax=20.0, path="./test.png", mode='light'):
	
	maxDX = np.argmin(np.absolute(r - 7.0))

	r = r[0 : maxDX]
	f = f[0 : maxDX]
	# if mode == 'dark':
	# 	axisColor = (1,1,1)
	# else:
	# 	axisColor = (0,0,0)

	theta = np.linspace(0.0, 1.0 * np.pi, 2 * np.size(r))

	R, T = np.meshgrid(r, theta)

	fun  = sin.interp1d(r, f)

	F    = fun(R)
	Y = sps.lpmv(M, L, np.cos(T))

	X1   = R * np.cos(T)
	X2   = R * np.sin(T)

	R_AX = r[-1] * 1.0
	ymin = (1.0 - R_AX / r[-1]) / 2.0
	ymax =  R_AX / r[-1] / 2.0 + 0.5

	yticks=[0, 1, 2, 3]

	for k in range(int(R_AX / 2.0) + 1):
		yticks[k] = 2.0 * k

	x = np.linspace(yticks[2] / np.sqrt(2.0), R_AX / np.sqrt(2.0), 1024)

	fig = plt.figure(figsize=cm2inch(14.5, 8.96))
	ax = plt.subplot(111, frameon=False)

	ax.plot(np.sqrt(np.power(R_AX, 2.0) - np.power(R_AX * np.cos(theta), 2.0)), R_AX * np.cos(theta), color=COLOR, linewidth=0.75)
	ax.axvline(0.0, ymin=ymin, ymax=ymax, color=COLOR, linewidth=1.0, zorder=4)
	ax.set_yticks(ticks=yticks)
	ax.set_xticks(ticks=[])
	for k in range(2, np.size(yticks)):
		ax.plot(np.sqrt(np.power(0.50, 2.0) - np.power(0.50 * np.cos(theta), 2.0)), 0.50 * np.cos(theta), '--', color=COLOR, linewidth=0.75, alpha=0.50)
		# ax.plot(- np.sqrt(np.power(0.50, 2.0) - np.power(0.50 * np.cos(theta), 2.0)), 0.50 * np.cos(theta), '--', color=COLOR, linewidth=2.5, alpha=0.50)
	ax.axis('equal')
	ax.yaxis.label.set_color(COLOR)
	ax.tick_params(axis='y', colors=COLOR)

	plt.axis('off')

	plt.plot(x, x, '--', linewidth=0.75, color=COLOR, alpha=0.50)
	plt.plot(x, - x, '--', linewidth=0.75, color=COLOR, alpha=0.50)
	plt.plot(np.sqrt(2.0) * x, x * 0, '--', linewidth=0.75, color=COLOR, alpha=0.50)

	plt.text(0.0, 1.05 * R_AX,r'$0$',
	     horizontalalignment='center',
	     verticalalignment='center', color=COLOR)
	plt.text(1.075 * R_AX / np.sqrt(2.0), 1.075 * R_AX / np.sqrt(2.0),r'$\pi / 4$',
	     horizontalalignment='center',
	     verticalalignment='center', color=COLOR)
	plt.text(1.075 * R_AX, 0.0, r'$\pi / 2$',
	     horizontalalignment='center',
	     verticalalignment='center', color=COLOR)
	plt.text(1.1 * R_AX / np.sqrt(2.0), - 1.1 * R_AX / np.sqrt(2.0),r'$3 \pi / 4$',
	     horizontalalignment='center',
	     verticalalignment='center', color=COLOR)
	plt.text(0.0, - 1.05 * R_AX,r'$\pi$',
	     horizontalalignment='center',
	     verticalalignment='center', color=COLOR)

	func = F * Y
	func /= np.max(np.absolute(func))
	im = ax.pcolormesh(X2, X1, func, cmap="seismic", vmin=-1.0, vmax=1.0, zorder=3, rasterized=True)

	plt.ylabel("Radius (a.u.)", color=COLOR)
	# plt.title(title, color=COLOR)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.25)
	cb = fig.colorbar(im, cax=cax, cmap=cmap)
	cb.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
	cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
	plt.setp(cbytick_obj, color=COLOR)
	cb.ax.yaxis.set_tick_params(color=COLOR)
	cb.set_label("Orbital (norm.)")
	plt.tight_layout()
	plt.savefig(path, format="PDF", bbox_inches="tight", dpi=600, transparent=True)

def linearPlot2Y (x, y1, y2, xmin=None, xmax=None, y2min=None, y2max=None, figsize=cm2inch(14.5, 8.96)):

	fig, ax1 = plt.subplots(figsize=figsize)
	color = 'tab:blue'
	ax1.set_xlabel('Energy (eV)')
	ax1.set_ylabel('Amplitude (norm.)', color=color)
	ax1.plot(x, y1, color='tab:blue', zorder=0)
	ax1.fill_between(x, y1, color=color, alpha=0.50, zorder=0)
	ax1.set_xlim(xmin, xmax)
	ax1.set_ylim(0, 1.1)
	ax1.grid(True)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:red'
	ax2.set_ylabel('Group Delay (fs)', color=color)  # we already handled the x-label with ax1
	ax2.plot(x, y2, color=color, zorder=1)
	ax2.plot(x, 0.01 * x, '--', color=color, zorder=1)
	ax2.set_ylim(y2min, y2max)
	ax2.set_ylim(y2min, y2max)
	ax2.tick_params(axis='y', labelcolor=color)
	plt.tight_layout()
	plt.savefig("./test2.png", dpi=300, format="PNG", bbox_inches="tight", transparent=False)

def plotSystem (targetID, eigenStates, density, r, drdx, P, N_r, N_g, N_kso_sph, n_sph, l_sph, occ_sph):

	densityPlot = (N_r + 1) * (N_r + 2) / 2.0 * density / drdx * P * P

	linearPlot (r, (N_r + 1) * (N_r + 2) / 2.0 * density / drdx * P * P, \
				xlabel=r'Radius (a.u.)', ylabel=r'Radial Probability Density', \
				xmin=r[0], xmax=r[-1], \
				ymin=0.0, \
				title=str(targetID),\
				path="../figures/" + internal.getTargetLabel(targetID) + "/single/static/groundState/linearPlots/density.pdf", thesisStyle=False, log=False, logx=True, figsize=cm2inch(14.5, 8.97), dot=False)

	for ksodx in range(N_kso_sph):

		f = np.sqrt((N_r + 1) * (N_r + 2) / 2.0 / drdx) * P * eigenStates[:, ksodx].real

		dip = np.zeros(N_r)

		print("       - Generating wave function " + str(n_sph[ksodx]) + internal.translateL(l_sph[ksodx]) + str(occ_sph[ksodx]) + " plot ")
		linearPlot ((r), f, \
					xlabel=r'$r$ (a.u.)', ylabel=r'$\psi_{' + str(n_sph[ksodx]) + r',' + internal.translateL(l_sph[ksodx]) + '}(r)$', \
					xmin=r[0], xmax=r[-1], \
					title=None, \
					minorTicks=False,\
					path="../figures/" + internal.getTargetLabel(targetID) + "/single/static/groundState/linearPlots/" + str(n_sph[ksodx]) + internal.translateL(l_sph[ksodx]) + str(occ_sph[ksodx]) + ".pdf", \
					log=False, logx=True, thesisStyle=False, dot=False, figsize=cm2inch(6.625, 7.5))

		n = n_sph[ksodx]
		l = l_sph[ksodx]
		for m in range(0, l_sph[ksodx] + 1):
			polarContour (f, r, l_sph[ksodx], m, \
							   rmax=10.0,\
							   title=r'$\psi_{' + str(n) + ',' + str(l) + ',' + str(m) + '}$',\
							   path="../figures/" + internal.getTargetLabel(targetID) + "/single/static/groundState/contourPlots/n" + str(n_sph[ksodx]) + 'l' + internal.translateL(l_sph[ksodx]) + 'm' + str(m) + ".pdf", mode='dark')
