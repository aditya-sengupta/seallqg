# 2021-06-08, Aditya Sengupta

include("../src/aberrations.jl")

using Unitful: Hz, m, rad
using UnitfulAstro: arcsecond

mas = arcsecond / 1000 # I can't pipe to mas but that's okay for now
vibe_dimension = 2

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


vibe_ranges = [
    [0.1 * mas, 1 * mas], # amplitude, or amplitude_x
    [aberration_params[:f_low], aberration_params[:f_high]], # frequency
    [1e-5, 1e-4], # damping,
    [0, 2π * rad]
]
if vibe_dimension == 2
    vibe_ranges = vcat(vibe_ranges, [vibe_ranges[1]]) # add an amplitude_y
end

layers, prop, aperture, pupil_grid = get_atmosphere_and_optics(aberration_params)
get_wf_clean = () -> hcipy.Wavefront(aperture, )