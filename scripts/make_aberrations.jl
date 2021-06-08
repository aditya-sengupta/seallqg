# 2021-06-08, Aditya Sengupta

include("../src/aberrations.jl")

using Unitful: Hz, m
using UnitfulAstro: arcsecond

mas = arcsecond / 1000 # I can't pipe to mas but that's okay for now

# defining constants

aberration_params = Dict{Symbol,Number}(
    :N_vib_app          => 10,
    :f_sampling         => 1000.0 * Hz,
    :f_low              => 1000 / 60 * Hz, # lowest possible frequency of a vibration mode
    :f_high             => 1000 / 3 * Hz, # highest possible frequency of a vibration mode
    :f_w                => 1000 / 3 * Hz, # frequency above which measurement noise dominates
    :measurement_noise  => 0.06 * mas, # milliarcseconds; pulled from previous notebook
    :D                  => 10.95 * m,
    :r0                 => 16.5e-2 * m,
    :pupil_size         => 16, # pixels
    :focal_samples      => 8, # samples per λ/D
    :focal_width        => 8, # half the number of lambda over Ds
    :λ                  => 500e-9 * m 
)

layers, prop, aperture, pupil_grid = get_atmosphere_and_optics(aberration_params)