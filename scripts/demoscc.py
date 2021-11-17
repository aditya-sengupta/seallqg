import time
from sealrtc import *
from sealrtc.optics.demo_scc import *

gopt_path = joinsimdata("gopt.npy")
if path.isfile(gopt_path):
	gopt = np.load(gopt_path)
	gopt_lowfs=np.load(joinsimdata("gopt_lowfs.npy"))
else:
	phatm = make_noise_pl(nmrms,imagepix,pupilpix,wav0,-2)
	phatmin = phatm
	coeffsolarr = np.zeros((2*len(freq_loop),niter))
	coeffslowfsolarr = np.zeros((zrefarr.shape[0],niter))
	print("Open-loop")
	for i in tqdm.trange(niter):
		imin, phatmin = genim(phatmin, phout_diff) 
		# open loop: images are not used to generate DM commands, 
		# just to generate coefficients that are processed off-line to determine the optimal gains
		coeffsolarr[:,i] = lsq_get_coeffs(imin)
		coeffslowfsolarr[:,i] = lowfs_get_coeffs(imin)

	optgim=np.zeros((imagepix,imagepix))
	goptarr=np.zeros(len(freq_loop))
	numoptg=100
	for j in tqdm.trange(len(freq_loop)):
		coefftseries=detrend(np.sqrt(coeffsolarr[j,:]**2.+coeffsolarr[j+len(freq_loop),:]**2.))
		freqol,psdol=welch(coefftseries,fs=1./t_int)
		freqol,psdol=freqol[1:],psdol[1:] #remove DC component (freq=0 Hz)

		optg = lambda g: np.trapz(psdol*np.abs(Hrej(freqol,t_int,t_lag,g))**2.,freqol) #function to optimize the gain for the integral of the OL PSD times the square modulus of rejection transfer function as a function of the gain

		optgarr=np.zeros(numoptg)
		garr=np.linspace(0.,1.5,numoptg)
		for gg in range(numoptg):
			g=garr[gg]
			optgarr[gg]=optg(g)
		gopt=garr[np.where(optgarr==np.min(optgarr))[0]][0]
		optgim[p3i(loopy[j]-beam_ratio/2.):p3i(loopy[j]+beam_ratio/2.),p3i(loopx[j]-beam_ratio/2.):p3i(loopx[j]+beam_ratio/2.)]=gopt
		goptarr[j]=gopt
	goptlowfsarr=np.zeros(zrefarr.shape[0])
	for j in range(zrefarr.shape[0]):
		coefftseries_lowfs=detrend(coeffslowfsolarr[j,:])
		freqol_lowfs,psdol_lowfs=welch(coefftseries_lowfs,fs=1./t_int)
		freqol_lowfs,psdol_lowfs=freqol_lowfs[1:],psdol_lowfs[1:]
		optg_lowfs = lambda g: np.trapz(psdol_lowfs*np.abs(Hrej(freqol_lowfs,t_int,t_lag,g))**2.,freqol_lowfs)
		optglowfsarr=np.zeros(numoptg)
		garr=np.linspace(0.,1.5,numoptg)
		for gg in range(numoptg):
			g=garr[gg]
			optglowfsarr[gg]=optg_lowfs(g)
		gopt=garr[np.where(optglowfsarr==np.min(optglowfsarr))[0]][0]
		goptlowfsarr[j]=gopt

	gopt=np.vstack(np.append(goptarr,goptarr)) #assumes the same gain for the sine and cosine coefficients
	np.save(joinsimdata('ol_coeffs.npy'), coeffsolarr)
	np.save(joinsimdata('ol_coeffs_lowfs.npy'), coeffslowfsolarr)
	np.save(joinsimdata('gopt.npy'), gopt)
	np.save(joinsimdata('gopt_lowfs.npy'), goptlowfsarr)
	np.save(joinsimdata('goptim.npy'), optgim)

	gopt=np.load('gopt.npy')
	gopt_lowfs=np.load('gopt_lowfs.npy')


def close_sim_loop():
	#close the loop!
	phatm = make_noise_pl(nmrms,imagepix,pupilpix,wav0,-2)
	phatmin = phatm
	phdm = phout_diff
	#phdm=no_phase_offset
	imin, phatmin = genim(phatmin, phout_diff) #first frame
	np.save(joinsimdata('phdm.npy'), phdm)
	np.save(joinsimdata('phatmin.npy'), phatmin)
	np.save(joinsimdata('imin.npy'), imin)

	#manually set DH corners equal to zero gain; not sure why there are a non-zero gain as I am setting these modes to zero in the IM...
	ind_corner=np.where(freq_loop>N_act/2.)[0]
	gopt[ind_corner,:],gopt[ind_corner+len(freq_loop),:]=0.,0.

	vpup=lambda im:(aperture*im)[p3i(imagepix/2-pupilpix/2):p3i(imagepix/2+pupilpix/2),p3i(imagepix/2-pupilpix/2):p3i(imagepix/2+pupilpix/2)]
	vim=lambda im: im[p3i(imagepix/2-N_act/2*beam_ratio):p3i(imagepix/2+N_act/2*beam_ratio),p3i(imagepix/2-N_act/2*beam_ratio):p3i(imagepix/2+N_act/2*beam_ratio)]

	size=20
	font = {'family' : 'Times New Roman',
			'size'   : size}

	mpl.rc('font', **font)
	mpl.rcParams['image.interpolation'] = 'nearest'

	from matplotlib import animation

	fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(10,10))
	[ax.axis('off') for ax in axs.flatten()]
	[axs.flatten()[i].set_title(['OL phase','CL phase','OL SCC image','CL SCC image'][i],size=size) for i in list(range(4))]
	im1=axs[0,0].imshow(vpup(phatm),vmin=-1,vmax=1)
	im2=axs[0,1].imshow(vpup(phatm),vmin=-1,vmax=1)
	im3=axs[1,0].imshow(vim(imin),vmin=0,vmax=1e-5)
	im4=axs[1,1].imshow(vim(imin),vmin=0,vmax=1e-5)
	fig.suptitle('t=0.000s, OL',y=0.1,x=0.51)

	Tint_cl=1 #number of seconds to run the closed-loop simulation
	num_time_steps=int(Tint_cl/t_int)
	time_steps = np.arange(num_time_steps)

	def animate(it):
		phdm=np.load('phdm.npy')
		phatmin=np.load('phatmin.npy')
		imin=np.load('imin.npy')
		
		# note that there is no servo lag simulated here; 
		# DM commands are applied to the atmospheric realization starting at the end of the previous exposure 
		# (so this is at least simulating half a frame lag). 
		# at 100 Hz, even a 1 ms lag is only one tenth of a frame delay, effectively negligible

		if it*t_int<=0.25: #open loop
			phlsq,imoutlsq=corr(imin,np.zeros(gopt.shape))
			phlowfs=lowfsout(imin,np.zeros(gopt_lowfs.shape))
			fig.suptitle('t='+str(round(t_int*it,4))+'s, OL',y=0.1,x=0.51)
		elif it*t_int<=0.5: #TTF loop closed
			phlsq,imoutlsq=corr(imin,np.zeros(gopt.shape))
			phlowfs=lowfsout(imin,gopt_lowfs)
			fig.suptitle('t='+str(round(t_int*it,4))+'s, CL TTF',y=0.1,x=0.51)
		else: #all loops closed
			phlsq,imoutlsq=corr(imin,gopt)
			phlowfs=lowfsout(imin,gopt_lowfs)
			fig.suptitle('t='+str(round(t_int*it,4))+'s, CL TTF+HO',y=0.1,x=0.51)
		phdm=phdm+phlsq+phlowfs

		imout_ol,phatmout=genim(phatmin,phout_diff)
		imout,phatmout=genim(phatmin,phdm)
		
		np.save(joinsimdata('phdm.npy'),phdm)
		np.save(joinsimdata('phatmin.npy'),phatmout)
		np.save(joinsimdata('imin.npy'),imout)

		im1.set_data(vpup(phatmout))
		im2.set_data(vpup(phatmout+phdm-phout_diff))
		im3.set_data(vim(imout_ol))
		im4.set_data(vim(imout))
		print(f"closed-loop: iteration {it} of {num_time_steps}")
		return [im1,im2,im3,im4]

	ani = animation.FuncAnimation(fig, animate, time_steps, interval=50, blit=True)
	ani.save(joinplot('fast_cl_demo.gif'),writer='imagemagick')
	plt.close(fig)