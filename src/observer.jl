"""
Make a bunch of Kalman filters.
"""

using LinearAlgebra
import Statistics: mean
using PyCall
using DSP

signal = pyimport("scipy.signal")

include("kfilter.jl")

# global parameter definitions
f_sampling = 1000  # Hz
f_1 = f_sampling / 60  # lowest possible frequency of a vibration mode
f_2 = f_sampling / 3  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
N_vib_max = 10  # number of vibration modes to be detected
energy_cutoff = 1e-8  # proportion of total energy after which PSD curve fit ends
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
time_id = 1 # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = 0:time_id:1/f_sampling # array of times to operate on
a = 1e-6 # the pole location for the f^(-2/3) powerlaw

function get_pgram(pos)
    return welch_pgram(pos, nfft=max(1000, length(pos)÷2), fs=f_sampling)
end

function find_peaks(s, height=1e-4)
    locs, peaks = signal.find_peaks(s, height=height)
    return locs, peaks["peak_heights"]
end

function damped_harmonic(pars_model)
    A, f, k, p = pars_model
    return A * exp.(-k * 2π * f * times) .* cos(2π * f * sqrt(1 - k^2) * times - p)
end

function find_freq_peaks(psd, N=N_vib_max)
    
end

function vibe_fit_freq(psd, N=N_vib_max)
    # takes in a PSD
    # returns an length-N array of length-4 arrays with fit parameters, and a length-N array with variances.
    par0 = [1e-4, 1]
    PARAMS_SIZE = 2
    width = 1

    peaks = []
    
end

function make_kfilter_vib(params, variances)
    
end

"""
Make the Kalman filter for an autoregressive model.
"""
function make_kfilter_ar(ar_len::Int64, openloops::Vector; σ::Float64=0.06)
    n = length(openloops)
    TTs_mat = Matrix{Float64}(undef, n - ar_len, ar_len)
    for i = 1:ar_len
        TTs_mat[:, i] = openloops[ar_len - i + 1 : n - i] 
    end

    ar_coef = TTs_mat \ openloops[ar_len + 1: end]
    ar_residual = openloops[ar_len:end-1] .- (TTs_mat * ar_coef)
    A = zeros(ar_len, ar_len)
    A[1,:] = ar_coef
    for i in 2:ar_len
        A[i,i-1] += 1.0
    end

    B = zeros(ar_len, 1) # need to change this for control delay
    B[1] = 1.0 # input just hits the current x
    C = zeros(1, ar_len)
    C[1] += 1

    Q = zeros(ar_len, ar_len)
    Q[1,1] = mean(ar_residual .^ 2)

    R = σ^2 * Matrix{Float64}(I, 1, 1)

    KFilter(A, B, C, Q, R)
end

