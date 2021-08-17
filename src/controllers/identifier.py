# authored by Aditya Sengupta

# could refactor to not be a class, but it's fine for now

from operator import xor
import numpy as np
from scipy import optimize, signal, stats
from .kfilter import KFilter
from ..utils import genpsd
from ..constants import fs

def log_likelihood(func, data):
    def get_ll(pars):
        pars_model, sd = pars[:-1], pars[-1]
        data_predicted = func(pars_model)
        LL = -np.sum(stats.norm.logpdf(data, loc=data_predicted, scale=sd))
        return LL

    return get_ll

class SystemIdentifier:
    """
    Driver class to build KFilter objects from open-loop data.
    Largely built off the Meimon 2010 system identification, and simulation/experimentation on top of that.
    """
    def __init__(
        self, 
        fs=fs, 
        f1=None, f2=None, fw=None, 
        N_vib_max=10, energy_cutoff=1e-8, 
        max_ar_coef=5, 
        time_id=10
    ):

        self.fs = fs # Hz
        if f1 is None:
            self.f1 = fs / 60 # lowest possible frequency of a vibration mode
        else:
            self.f1 = f1

        if f2 is None:
            self.f2 = fs / 3 # highest possible frequency of a vibration mode
        else:
            self.f2 = f2

        if fw is None:
            self.fw = fs / 3 # frequency above which measurement noise dominates
        else:
            self.fw = fw

        self.max_ar_coef = max_ar_coef
        self.N_vib_max = N_vib_max # number of vibration modes to be detected
        self.energy_cutoff = energy_cutoff # proportion of total energy after which PSD curve fit ends
        self.time_id = time_id # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.

    @property
    def times(self):
        return np.arange(0, self.time_id, 1 / self.fs)

    def damped_harmonic(self, pars_model):
        A, f, k, p = pars_model
        return A * np.exp(-k * 2 * np.pi * f * self.times) * np.cos(2 * np.pi * f * np.sqrt(1 - k**2) * self.times - p)

    def make_psd(self, pars_model):
        s = self.damped_harmonic(pars_model)
        return genpsd(s, dt = 1 / self.fs)

    def psd_f(self, f):
        def get_psd_f(pars):
            k = pars[0]
            return self.make_psd([1, f, k, np.pi])[1]

        return get_psd_f

    def est_measurenoise(self, freqs, psd):
        return np.sqrt(np.mean(self.fs * psd[freqs > self.fw]))

    def interpolate_psd(self, freqs, psd):
        mask = np.logical_and(self.f1 < freqs, freqs < self.f2)
        return stats.linregress(np.log(freqs[mask], np.log(psd[mask])))

    def vibe_fit_freq(self, freqs, psd):
        # takes in the frequency axis for a PSD, and the PSD.
        # returns a 4xN np array with fit parameters, and a 1xN np array with variances.
        par0 = [1e-4, 1]
        PARAMS_SIZE = 2
        df = freqs[1] - freqs[0]
        width = max(1, int(np.ceil(1/df)))

        peaks = []
        unsorted_peaks = signal.find_peaks(psd, height=self.energy_cutoff)[0]
        freqs_energy = np.flip(np.argsort(psd)) # frequencies ordered by their energy
        for f in freqs_energy:
            if f in unsorted_peaks and self.f1 <= freqs[f] <= self.f2:
                peaks.append(f)

        params = np.zeros((self.N_vib_max, PARAMS_SIZE))
        variances = np.zeros(self.N_vib_max)

        i = 0
        for peak_ind in peaks:
            if i >= self.N_vib_max:
                break
            if not(np.any(np.abs(params[:,0] - freqs[peak_ind]) <= width * df)): 
                # this is to prevent peak overlapping/fitting to the same thing twice
                l, r = peak_ind - width, peak_ind + width
                windowed = psd[l:r]
                objective = lambda pars: self.psd_f(freqs[peak_ind])(pars)[l:r]
                psd_ll = log_likelihood(objective, windowed)
                k, sd = optimize.minimize(psd_ll, par0, method='Nelder-Mead').x
                params[i] = [freqs[peak_ind], k]
                variances[i] = sd ** 2
                i += 1

        return params[:i], variances[:i], psd

    def make_state_transition_vibe(self, params):
        STATE_SIZE = 2 * params.shape[0]
        A = np.zeros((STATE_SIZE, STATE_SIZE))
        for i in range(STATE_SIZE // 2):
            f, k = params[i]
            w0 = 2 * np.pi * f / np.sqrt(1 - k**2)
            A[2 * i][2 * i] = 2 *  np.exp(-k * w0 / self.fs) * np.cos(w0 * np.sqrt(1 - k**2) / self.fs)
            A[2 * i][2 * i + 1] = -np.exp(-2 * k * w0 / self.fs)
            A[2 * i + 1][2 * i] = 1
        return A

    def make_kfilter_vibe(self, params, variances, sigma):
        # takes in parameters and variances from which to make a physics simulation
        # and measurements to match it against.
        # returns a KFilter object.
        A = self.make_state_transition_vibe(params)
        STATE_SIZE = 2 * params.shape[0]
        C = np.array([[1, 0] * (STATE_SIZE // 2)])
        W = np.zeros((STATE_SIZE, STATE_SIZE))
        for i in range(variances.size):
            Q[2 * i][2 * i] = variances[i]
        V = sigma**2 * np.identity(1)
        return KFilter(A, C, W, V)

    def make_kfilter_turb(self, impulse, sigma):
        # takes in an impulse response as generated by make_impulse
        # and returns a KFilter object.
        n = impulse.size
        A = np.zeros((n, n))
        for i in range(1, n):
            A[i][i-1] = 1
        A[0] = (np.real(impulse)/sum(np.real(impulse)))
        W = np.zeros((n,n))
        W[0][0] = 1 # arbitrary: I have no idea how to set this yet.
        C = np.zeros((1,n))
        C[:,0] = 1
        V = np.array([sigma ** 2])
        return KFilter(A, C, W, V)

    def make_kfilter_ar(self, openloops, ar_len, sigma=0.06):
        n = len(openloops)
        TTs_mat = np.empty((n - ar_len, ar_len))
        for i in range(ar_len):
            TTs_mat[:, i] = openloops[ar_len - i : n - i] 

        ar_coef, _, _, _ = np.linalg.lstsq(TTs_mat, openloops[ar_len:], rcond=None)
        ar_residual = openloops[ar_len:] - (TTs_mat @ ar_coef)
        A = np.zeros((ar_len, ar_len))
        A[0,:] = ar_coef
        for i in range(1, ar_len):
            A[i,i-1] += 1.0

        C = np.zeros((1, ar_len))
        C[0] += 1

        W = np.zeros((ar_len, ar_len))
        W[1,1] = np.mean(ar_residual ** 2)

        V = np.array([[sigma ** 2]])

        return KFilter(A, C, W, V)

    def make_kfilter_from_openloop(self, ol, model_atm=False, model_vib=True):
        """
        Designs a KFilter object based on open-loop data. 
        (Essentially stitches together a bunch of KFilter objects.)

        Arguments
        ---------
        ol : np.ndarray, (N,2)
        The open-loop data to use to build the filter.

        model_atm : bool
        Makes a filter that models atmospheric aberrations.

        model_vib : bool
        Makes a filter that models mechanical vibrations.

        Returns
        -------
        kf : KFilter
        A Kalman filter object that controls either tip or tilt.
        """
        kfs = []
        self.time_id = ol.shape[0] / self.fs
        self.N_vib_max = 1 # just for now
        for i in range(2): # tip, tilt
            kf = KFilter(np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0)), verbose=False)
            f, p = genpsd(ol[:,i], dt=1 / self.fs)
            self.energy_cutoff = np.mean(p[f > self.fw])
            sigma = self.est_measurenoise(f, p)

            if model_vib:
                params, variances, p = self.vibe_fit_freq(f, p)
                kf += self.make_kfilter_vibe(params, variances, sigma)
            if model_atm:
                kf += self.make_kfilter_turb(np.fft.ifft(np.fft.fftshift(p))) # ?? is this it? 
                # dinosaur double check fractal_deriv
            kfs.append(kf)

        return kfs[0].concat(kfs[1])
