# split off from tt_opt.py for convenience, so that we can separately %run that from IPython.
# to be run after tt_opt.py.

#make MTF side lobe mask
xsidemaskcen, ysidemaskcen = 161.66, 252.22 #x and y location of the side lobe mask in the cropped image
sidemaskrad = 26.8 #radius of the side lobe mask
mtfgrid = np.mgrid[0:imini.shape[0], 0:imini.shape[1]]
sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask = np.zeros(imini.shape)
sidemaskind = np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind] = 1

#side lobe mask where there is no signal to measure SNR
xnoise, ynoise = 54, 252.22
sidemaskrhon = np.sqrt((mtfgrid[0]-ynoise)**2 +(mtfgrid[1]-xnoise)**2)
sidemaskn = np.zeros(imini.shape)
sidemaskindn = np.where(sidemaskrhon<sidemaskrad)
sidemaskn[sidemaskindn] = 1

def processimabs(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(np.fft.fft2(imin, norm='ortho')) #(1) FFT the image
	otf_masked = otf*mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus = np.fft.ifft2(otf_masked, norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

#grid tip/tilt search 
tsleep = 0.05 #time to sleep in between applying DM commands and grabbing images, determined in optt of genIM.py

cenmaskrho = np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
cenmask = np.zeros(imini.shape)
cenmaskradmax,cenmaskradmin = 49, 10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
cenmaskind = np.where(np.logical_and(cenmaskrho<cenmaskradmax, cenmaskrho>cenmaskradmin))
cenmask[cenmaskind] = 1

namp = 10
amparr = np.linspace(-0.3, 0.3, namp) #note the range of this grid search is can be small, assuming day to day drifts are minimal and so you don't need to search far from the previous day to find the new optimal alignment; for larger offsets the range may need to be increases (manimum search range is -1 to 1); but, without spanning the full -1 to 1 range this requires manual tuning of the limits to ensure that the minimum is not at the edge
ttoptarr = np.zeros((namp, namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(amparr[i], amparr[j], False)
		time.sleep(tsleep)
		imopt = stack(1000)
		mtfopt = mtf(imopt)
		sidefraction = np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction = np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr[i,j] = sidefraction+0.01/cenfraction #the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at 1%, this attempts to give equal weight to both terms 

#medttoptarr = median_filter(ttoptarr,3) #smooth out hot pizels, attenuating noise issues
indopttip, indopttilt = np.where(ttoptarr == np.max(ttoptarr))
indopttip, indopttilt = indopttip[0],indopttilt[0]
applytiptilt(amparr[indopttip],amparr[indopttilt])

expt(1e-3)

ampdiff = amparr[2]-amparr[0] #how many discretized points to zoom in to from the previous iteration
tipamparr = np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
tiltamparr = np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
ttoptarr1 = np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(tipamparr[i], tiltamparr[j])
		time.sleep(tsleep)
		imopt = stack(1000)
		mtfopt = mtf(imopt)
		sidefraction = np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction = np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr1[i,j] = sidefraction+0.01/cenfraction 

#medttoptarr1 = median_filter(ttoptarr1,3) #smooth out hot pizels, attenuating noise issues
indopttip1,indopttilt1 = np.where(ttoptarr1 == np.max(ttoptarr1))
applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])

bestflat = getdmc()

#next: manually fine tune bestflat by placing sine waves and adjusting by eye that the spot intensities look even
#...still having trouble with implementing this section; it seems like my eyes may be biased to evening out speckles 
# that are interfering with the sinespots, thereby degrading the quality of the alignment
'''
#functions to apply DM Fourier modes 
ycen,xcen = ydim/2-0.5,xdim/2-0.5
indrho1 = np.where(rho == 1)
gridnorm = np.max(grid[0][indrho1])
rgrid = lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa,bestflat = bestflat): #generate sine wave
	sin = amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm = sin.astype(float32)
	dmc = bestflat+sindm
	applydmc(dmc*aperture)
	return sindm

sin1 = dmsin(0.1,2.5,90,bestflat = bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
#applytilt(-0.1)
bestflat = getdmc()-sin1
sin2 = dmsin(0.1,2.5,0,bestflat = bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
bestflat = getdmc()-sin2
'''
np.save('bestflat.npy',bestflat)

#unfinished code to close the tip/tilt loop:
def vim(tsleep): #if I want to close the tip/tilt loop, take a look at what timesteps in between images are needed 
	applybestflat()
	time.sleep(tsleep)
	imref = cropim(getim())
	applytiptilt(-0.1, 0)
	time.sleep(tsleep)
	imtip = cropim(getim())
	#applytiptilt(0,0.1)
	#time.sleep(tsleep)
	#imtilt = cropim(getim())
	applybestflat()
	ds9.view(imtip-imref)
	