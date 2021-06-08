# ttsysid

System identification for tip-tilt adaptive optics. This is a partial port of the tip-tilt control repository [here](github.com/aditya-sengupta/tip-tilt-control). Commits are from aditya-sengupta, I hope.

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> ttsysid

It is authored by aditya-sengupta.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.
