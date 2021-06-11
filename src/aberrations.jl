# 2021-06-08, Aditya Sengupta
# Functions to generate aberration profiles, to be called by scripts/make_aberrations.jl.

using PyCall
using Unitful
using Unitful: Hz, m
using UnitfulAstro: arcsecond
using LinearAlgebra
using Distributions

hcipy = pyimport("hcipy")

"""
Interface to hcipy functions to define optics and atmospheric layers.
"""
function get_atmosphere_and_optics(params::Dict{Symbol,Number})
    λ = params[:λ]
    ps = params[:pupil_size]
    fs, fw = params[:focal_samples], params[:focal_width]
    pupil_grid = hcipy.make_pupil_grid(ps, 1)
    focal_grid = hcipy.make_focal_grid_from_pupil_grid(pupil_grid, fs, fw, wavelength=ustrip(λ |> m))
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    aperture = hcipy.circular_aperture(1)(pupil_grid)
    layers = hcipy.make_standard_atmospheric_layers(pupil_grid)
    return layers, prop, aperture, pupil_grid
end

"""
Takes in an hcipy Field and returns its center of mass.
"""
function center_of_mass(f::PyObject, focal_samples::Int64)
    s = f.grid.shape[1] / 2
    normalize = s / (focal_samples * maximum(f.grid.x))
    cm = normalize .* [(f.grid.x * f).sum(), (f.grid.y * f).sum()] ./ f.sum()
    return cm
end

"""
Makes vibration parameters (amplitude, frequency, phase, damping) for N vibrational modes.
"""
function make_vibe_params(N::Int64=10; ranges::Array{Array{<: Number}})
    return map(r -> rand(Uniform(r[1], r[2]), N), ranges)
end

function make_vibe_data(nsteps::Int64, vibe_params=nothing, N::Int64=10)
    if isnothing(vibe_params)
        vibe_params = make_vibe_params(N)
    else
        N = length(vibe_params[1])
    end

    if length(vibe_params) == 5
        # we are in 2D mode
        return hcat(
            make_vibe_data(nsteps, vibe_params[1:4]), 
            make_vibe_data(nsteps, vibe_params[[5, 2, 3, 4]])
        ) |> eachcol |> collect
    end

    times = 0:(1 / f_sampling):((nsteps - 1) / f_sampling)

    vibrations = sum([
        vibe_params[1,i] * cos(2π * vibe_params[2,i] .* times - vibe_params[4,i]) 
        * exp(-(vibe_params[3,i]/(1 - vibe_params[3,i]^2)) * 2π * vibe_params[2,i] .* times)
        for i in 1:N
    ])
    return vibrations
end

"""
weights = number of desired lambda-over-Ds the center of the PSF is to be moved. Contains [tip_wt, tilt_wt].
"""
function make_fixed_tt(pupil_grid::PyObject, weights::Array{<: Number})
     tt = [hcipy.zernike(hcipy.ansi_to_zernike(i)..., 1)(pupil_grid) for i in 1:2]
     phase = π * (weights ⋅ tt)
     return hcipy.Wavefront(aperture * exp(im * phase))
end

"""
Makes atmospheric sim data for "nsteps" steps, starting from a wavefront "wf" propagating through layers "layers".
"""
function make_atm_sim(nsteps::Int64, wf::PyObject, layers::PyObject, params::Dict{Symbol,Number}; zerotime=0.0)
    λ = params[:λ]
    f_sampling = ustrip(params[:f_sampling] |> Hz)
    px_to_mas = (λ/params[:D]) |> arcsecond * 1000 / params[:focal_samples]
    tt_cms = zeros(nsteps, 2)
    for i in 1:nsteps
        for layer in layers
            startwf = copy(wf)
            layer.evolve_until(i / f_sampling + zerotime)
            startwf = layer(startwf)
        end
        tt_cms[i,:] = center_of_mass(prop(wf).intensity, params[:focal_samples])
    end
    tt_cms .*= px_to_mas
    return tt_cms
end
