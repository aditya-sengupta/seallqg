"""
Make a bunch of Kalman filters.
"""

using LinearAlgebra
import Statistics: mean

include("kfilter.jl")

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