import numpy as np
from matplotlib import pyplot as plt
from os.path import join

tstamp = "2021_11_18_13_26_54"
zernamparr = np.load(join("..", "data", "linearity", f"lin_{tstamp}.npy"))
zernampout = np.load(join("..", "data", "linearity", f"lout_{tstamp}.npy"))

wfe_in_microns = False
if wfe_in_microns:
    conv = dmc2wf
    unit = "$\\mu$m"
else:
    conv = 1
    unit = "DM units"
fig, axs=plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
fig.suptitle("Linearity in the first five Zernike modes on FAST")

colors = mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
axarr=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
fig.delaxes(axarr[-1])
for i in range(len(nmarr)):
    ax=axarr[i]
    ax.set_title('n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
    if i==4:
        ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--',label='y=x')
        for j in range(len(nmarr)):
            if j==i:
                ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j],label='n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
            else:
                ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
    else:
        ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--')
        for j in range(len(nmarr)):
            if j==i:
                ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j])
            else:
                ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j])

plt.savefig("linearity.pdf")