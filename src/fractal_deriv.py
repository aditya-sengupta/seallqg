import numpy as np
import mpmath as mp
from fractions import Fraction
import matplotlib.pyplot as plt
from utils import genpsd

plt.ion()

def design_from_psd(y, dt):
    """
    Designs a filter for a time-series of open loop values.

    Arguments
    ---------
    y : np.ndarray
    The time-series.

    dt : float
    The time step in seconds.
    """
    f, p = genpsd()
    fF = np.hstack((-np.flip(f[1:], f)))

def design_filt(dt=1, N=1024, fc=0.1, a=1e-6, tf=None, plot=True, oplot=False):
    '''Design a filter that takes white noise as input
    and produces the f^-2/3 transitioning to f^-11/3 spectrum of atmospheric
    tip/tilt. The way to do this is to put a "1/3" pole
    near s=0 and a "3/2-pole" (11/3-2/3)/2 at the wind
    clearing frequency, s=-fc

    Inputs:
        dt - time sample in seconds, float (defaults to 1.0 sec)
        N - number of points in the resulting impulse response, integer
            note: use a large number (default is 1024) to properly sample the spectrum over a
            reasonable dynamic range.
        fc - wind clearing frequency in Hz, float (defaults to 0.1 Hz)
        a - the location of the "-1/3" pole, a small number on the real axis near the origin, float (defaults to 1e-6 Hz)
        tf - the transfer function: a function that takes in 'f' and returns the corresponding point on the FT.
             or just the PSD that we want to invert.
        plot - if true, produce a plot of the impulse response and the power spectrum (default is True)
        oplot - if true, prodece a plot, but overplot it on curves from a prior call to design_filt (default is False)

    Output:
        x - a time sequence representing the impulse response of the filter. The filter can be implemented
            as a moving-average (MA) model using the values in x as the coefficients.
    '''
    i = 1j
    if tf is not None and not callable(tf):
        # the TF is just a power spectrum that's been passed in
        xF = tf
        m = len(tf) * 2
        f = (np.arange(m)-m//2)/(m * dt)
    else:
        # we generate the TF from design parameters in the function args
        f = (np.arange(N)-N//2)/(N * dt)
        s = i*f
        xF = ((s+a)**(-1/3)*(s+fc)**(-3/2))
    # this is flipped, so that it tracks instead of controlling, and maybe 1/3 instead of 3/2 for an 1/f^2 powerlaw
    if callable(tf):
        xF = np.vectorize(tf)(f)

    xF = np.fft.fftshift(xF)
    f = np.fft.fftshift(f)
    x = np.fft.ifft(xF)
    t = np.arange(N)*dt
    if plot:
        if not oplot:
            plt.figure(figsize=(6,7))
        plt.subplot(211)
        t = t[0:N//2]
        x = x[0:N//2]
        plt.plot(t, np.real(x))
        plt.grid(True)
        plt.ylabel('Impulse Response')
        plt.xlabel('time, seconds')
        plt.subplot(212)
        f = f[1:N//2]
        xF = xF[1:N//2]
        plt.plot(f,np.abs(xF)**2)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.ylabel('Power Spectrum')
        plt.xlabel('frequency, Hz')
    globals().update(locals())
    return np.real(x) / sum(np.real(x))

def filt(a, dt=1, u=None, N=1024, plot=True, oplot=False):
    '''filter the time series u by the filter a, defined
    in terms of its impulse response

    a is normalized to have unit integral
    u defaults to white noise

    Inputs:
        a - the impulse response of the filter. This can be any length vector of floats
        dt - the sample time, in seconds, float
        u - the input sequence, a vector of floats.
            Can be specified as None, in which case a white noise is generated in the routine. (defaults is None)
        N - the length of the white noise sequence u if u is to be generated (otherwise N not used). integer (defaults to 1024)
        plot - if True generate a plot of the resulting filter output time sequence and its power spectrum (default True)
        oplot - if True, the plot will add a curve to a plot from a prior call of filt (default False)

    Output:
        y - the result of convolving u with the filter a

    '''
    a = a/np.sum(a)
    n = len(a)
    if u is None:
        u = np.random.normal(size=(N,))
    else:
        N = len(u)
    y = np.zeros(N)
    t = dt*np.arange(N)
    df = 1./(2 * N*dt) # I put in the factor of 2 here to match Ben's genpsd method
    f = df*np.arange(1, N+1) # I changed arange(N) here to arange(1, N+1) here to match Ben's genpsd method
    for k in range(N):
        L = min(n,k+1)
        y[k] = np.sum(u[range(k,k-L,-1)]*a[0:L])
    if plot:
        if not oplot:
            plt.figure(figsize=(6,7))
            lu = 'b-'
            ly = 'r-'
        else:
            lu = 'b:'
            ly = 'r:'
        plt.subplot(211)
        plt.plot(t,u,label='white noise')
        plt.plot(t,y,label=r'tracked noise')
        plt.grid(True)
        plt.xlabel('time, seconds')
        plt.legend()
        plt.subplot(212)
        uF = np.fft.fft(u)/N
        yF = np.fft.fft(y)/N
        uF = uF[1:N//2]
        yF = yF[1:N//2]
        uPSD = N*np.abs(uF)**2
        yPSD = N*np.abs(yF)**2
        f = f[1:N//2]
        plt.plot(f,uPSD,lu,label='white noise')
        plt.plot(f,yPSD,ly,label=r'filtered noise')
        rPSD = (1/n)*(f*dt)**(-2/3)
        plt.plot(f,rPSD,'k--')
        cPSD = f**0
        plt.plot(f,cPSD,'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True,which='both')
        plt.xlabel('frequency, Hz')
        plt.legend()

    globals().update(locals())
    return y
