import itertools

#CREATE IMAGE X,Y INDEXING THAT PLACES SINE, COSINE WAVES AT EACH LAMBDA/D REGION WITHIN THE NYQUIST REGION
#the nyquist limit from the on-axis psf is +/- N/2 lambda/D away, so the whole nyquist region is N lambda/D by N lambda/D, centered on the on-axis PSF
#to make things easier and symmetric, I want to place each PSF on a grid that is 1/2 lambda/D offset from the on-axis PSF position; this indexing should be PSF copy center placement location
allx,ally=list(zip(*itertools.product(np.linspace(cropsize[0]-maxld*beam_ratio+0.5*beam_ratio,cropsize[0]-0.5*beam_ratio,maxld),np.linspace(cropsize[0]-maxld*beam_ratio+0.5*beam_ratio,cropsize[0]+maxld*beam_ratio-0.5*beam_ratio,N_act))))
loopx,loopy=np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

freq_loop=np.sqrt((loopy-cropsize[0].)**2.+(loopx-cropsize[0])**2.)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
pa_loop=90-180/np.pi*np.arctan2(loopy-cropsize[0],loopx-cropsize[0]) #position angle of sine wave for (lambda/D)**2 region w/in DH