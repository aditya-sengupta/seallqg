using ControlSystems



"""
Makes the impulse response for a first-order system (with TF = w / (s + w) for input w)
"""
function make_impulse_response(w::Number)
    transferfn = tf(w, [1, w])
    y, t, _ = impulse(transferfn)
    return t, y ./ sum(y)
end

"""
Makes the impulse response for a second-order system with a specified overshoot and rise time.
"""
function make_impulse_response(overshoot::Number, rise_time::Number)
    # damping ratio from overshoot
    ζ = -log(overshoot) / sqrt(π^2 + log(overshoot)^2)
    # find approximation to rise_time * natural frequency: weights from Nise chapter 7 ± 2 (I'll look this up later)
    ω = (1/rise_time) * (1.76 * ζ^3 - 0.417 * ζ^2 + 1.039 * ζ + 1)
    transferfn = tf(ω^2, [1, 2 * ω * ζ, ω^2])
    y, t, _ = impulse(transferfn)
    return t, y ./ sum(y)
end
