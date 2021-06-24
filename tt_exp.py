# my tip-tilt experiments
# to be run after %run tt_align.py

def find_limits():
    """
    Applies tips and tilts from the best flat to find the feasible range of DM TT commands.
    """
    min_amp = 1e-3
    max_amp = 1e-1

    limits = []
    for dmfn in [applytip, applytilt]:
        for sgn in [-1, +1]:
            applybestflat()
            applied_cmd = 0.0
            for step_size in 10 ** np.arange(np.log10(max_amp), np.log10(min_amp)-1, -1):
                in_range = True
                while in_range:
                    applied_cmd += sgn * step_size
                    in_range = all(dmfn(sgn * step_size))
                applied_cmd -= sgn * step_size
                dmfn(-sgn * step_size, False) # move it back within range, and retry with the smaller step size
            limits.append(applied_cmd)
                
    return limits[:2], limits[2:] # the first is min_tip, max_tip; the second is min_tilt, max_tilt

def command_linearity():
    """
    
    """
    pass

def uniformity():
    pass

def amplitude_linearity():
    pass

def unit_steps(min_amp, max_amp, steps_amp, steps_ang=12, tsleep=tsleep):
    angles = np.linspace(0.0, 2 * np.pi, steps_ang)
    amplitudes = np.linspace(min_amp, max_amp, steps_amp)
    for amp in amplitudes:
        for (i, ang) in enumerate(angles):
            applybestflat()
            time.sleep(tsleep) # vary this later: for now I'm after steady-state error
            imflat = stack(10)
            applytiptilt(amp * np.cos(ang), amp * np.sin(ang))
            time.sleep(tsleep)
            imtt = stack(10)
            np.save("unitstep_amp_%d_ang_%d.npy".format(round(amp, 3), round(ang, 3)), imtt-imflat)

def sinusoids():
    pass