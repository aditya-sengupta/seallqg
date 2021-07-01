'''
optimize tip/tilt offset based on fringe visibility

so far, manual steering by eye seems better than the optimal determined value

note that at the moment, code only works with the chopper not runing and pinhole unblocked
'''
#MANUALLY USE ABOVE FUNCTIONS TO STEER THE PSF BACK ONTO THE FPM AS NEEDED, then:
from ..src import tt

bestflat = getdmc()

#apply tip/tilt starting only from the bestflat point (start here if realigning the non-coronagraphic PSF) 
def applytiptilt(amptip,amptilt,bestflat=bestflat): #amp is the P2V in DM units
	dmctip=amptip*tip
	dmctilt=amptilt*tilt
	dmctiptilt=remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat)+0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	#applydmc(aperture*dmctiptilt)
	applydmc(dmctiptilt)

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=252.01,159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(imini.shape)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#side lobe mask where there is no signal to measure SNR
xnoise, ynoise=161.66, 252.22
sidemaskrhon=np.sqrt((mtfgrid[0]-ynoise)**2+(mtfgrid[1]-xnoise)**2)
sidemaskn=np.zeros(imini.shape)
sidemaskindn=np.where(sidemaskrhon<sidemaskrad)
sidemaskn[sidemaskindn]=1

def processimabs(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
	applytiptilt(-0.1,-0.1)
	time.sleep(tsleep)
	im1=stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stack(10)
	ds9.view(im1-imf)
#tsleep=0.005 #on really good days
tsleep=0.05 #optimized from above function
#tsleep=0.4 #on bad days


cenmaskrho = np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
cenmask = np.zeros(imini.shape)
cenmaskradmax, cenmaskradmin=49, 10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
cenmaskind=np.where(np.logical_and(cenmaskrho<cenmaskradmax, cenmaskrho>cenmaskradmin))
cenmask[cenmaskind] = 1

#grid tip/tilt search 
namp = 10
amparr = np.linspace(-0.1,0.1,namp) #note the range of this grid search is can be small, assuming day to day drifts are minimal and so you don't need to search far from the previous day to find the new optimal alignment; for larger offsets the range may need to be increases (manimum search range is -1 to 1); but, without spanning the full -1 to 1 range this requires manual tuning of the limits to ensure that the minimum is not at the edge
ttoptarr=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(amparr[i],amparr[j])
		time.sleep(tsleep)
		imopt=stack(10)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr[i,j]=sidefraction+0.1/cenfraction #the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at 1%, this attempts to give equal weight to both terms 

#medttoptarr=median_filter(ttoptarr,3) #smooth out hot pizels, attenuating noise issues
indopttip,indopttilt=np.where(ttoptarr==np.max(ttoptarr))
indopttip,indopttilt=indopttip[0],indopttilt[0]
applytiptilt(amparr[indopttip],amparr[indopttilt])

#expt(1e-4)

ampdiff=amparr[2]-amparr[0] #how many discretized points to zoom in to from the previous iteration
tipamparr=np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
tiltamparr=np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
ttoptarr1=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(tipamparr[i],tiltamparr[j])
		time.sleep(tsleep)
		imopt=stack(10)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr1[i,j]=sidefraction+0.1/cenfraction 

#medttoptarr1=median_filter(ttoptarr1,3) #smooth out hot pizels, attenuating noise issues
indopttip1,indopttilt1=np.where(ttoptarr1==np.max(ttoptarr1))
applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])

bestflat = getdmc()
im_bestflat = stack(100)

#next: manually fine tune bestflat by placing sine waves and adjusting by eye that the spot intensities look even...still having trouble with implementing this section; it seems like my eyes may be biased to evening out speckles that are interfering with the sinespots, thereby degrading the quality of the alignment
'''
#functions to apply DM Fourier modes 
ycen,xcen=ydim/2-0.5,xdim/2-0.5
indrho1=np.where(rho==1)
gridnorm=np.max(grid[0][indrho1])
rgrid=lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa,bestflat=bestflat): #generate sine wave
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=bestflat+sindm
	applydmc(dmc*aperture)
	return sindm

sin1=dmsin(0.1,2.5,90,bestflat=bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
#applytilt(-0.1)
bestflat=getdmc()-sin1
sin2=dmsin(0.1,2.5,0,bestflat=bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
bestflat=getdmc()-sin2
'''

from datetime import datetime
np.save('bestflats/'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy', bestflat)
np.save('bestflats/im_'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy', im_bestflat)
np.save('bestflat.npy',bestflat)

#unfinished code to close the tip/tilt loop:
def vim(tsleep): #if I want to close the tip/tilt loop, take a look at what timesteps in between images are needed 
	applydmc(bestflat)
	time.sleep(tsleep)
	imref=cropim(getim())
	applytiptilt(-0.1, 0)
	time.sleep(tsleep)
	imtip = cropim(getim())
	#applytiptilt(0,0.1)
	#time.sleep(tsleep)
	#imtilt=cropim(getim())
	applydmc(bestflat)
	ds9.view(imtip-imref)
