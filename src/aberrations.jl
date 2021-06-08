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
function get_atmosphere(params::Dict{Symbol,Number})
    λ = params[:λ]
    ps = params[:pupil_size]
    fs, fw = params[:focal_samples], params[:focal_width]
    pupil_grid = hcipy.make_pupil_grid(ps, 1)
    focal_grid = hcipy.make_focal_grid_from_pupil_grid(pupil_grid, fs, fw, wavelength=ustrip(λ |> m))
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    aperture = hcipy.circular_aperture(1)(pupil_grid)
    layers = hcipy.make_standard_atmospheric_layers(pupil_grid)
    return layers, prop, aperture
end

function make_vibe_params(N::Int64=10; ranges::Array{Array{<: Number}})
    return map(r -> rand(Uniform(r[1], r[2]), N), ranges)
end

function make_1D_vibe_data(nsteps::Int64, vibe_params=nothing, N::Int64=10)

end

function make_fixed_tt(weights::Array{<: Number})
     tt = [hcipy.zernike(hcipy.ansi_to_zernike(i)..., 1)(pupil_grid) for i in 1:2]
     phase = π * (weights ⋅ tt)
     return hcipy.Wavefront(aperture * exp(im * phase))
end

"""
Makes atmospheric sim data for "nsteps" steps, starting from a wavefront "wf" propagating through layers "layers".
"""
function make_atm_sim(nsteps::Int64, wf::PyObject, layers::PyObject, params::Dict{Symbol,Number}; zerotime=0.0, f_sampling = 1000*Hz)
    λ = params[:λ]
    f_sampling = ustrip(f_sampling |> Hz)
    px_to_mas = (λ/params[:D]) |> arcsecond * 1000 / params[:focal_samples]
    tt_cms = zeros(nsteps, 2)
    for i in 1:nsteps
        for layer in layers
            startwf = copy(wf)
            layer.evolve_until(i / f_sampling + zerotime)
            startwf = layer(startwf)
        end
        tt_cms[i,:] = center_of_mass(prop(wf).intensity)
    end
    tt_cms .*= px_to_mas
    return tt_cms
end
