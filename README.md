[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13323359.svg)](https://doi.org/10.5281/zenodo.13323359)
# Code for Koch _et al._ 2024 

This repository contains the data and code for reproducing the figures in the following publication:

Koch, N. A., Corrigan, B. W., Feyerabend, M., Gulli, R. A., Jimenez-Sosa, M. S., Abbass, M., … Martinez-Trujillo, J. C. (2025). Spike frequency adaptation in primate lateral prefrontal cortex neurons results from interplay between intrinsic properties and circuit dynamics. Cell Reports, 44(1), 115159. doi:[10.1016/j.celrep.2024.115159](https://doi.org/10.1016/j.celrep.2024.115159)


## Figures
First install the required packages in ```./requirements.txt```
```
pip install -r requirements.txt
```
Then run the python (.py) files in the ```./figures/src``` directory to reproduce the figures.



## Model Simulation
This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named ```LPFC_adaptation```

To reproduce the simulations, do the following:
1. Download this code base. 
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```
   This will install all necessary packages for you to be able to run the scripts andeverything should work out of the box, including correctly finding local paths.
3. Run the Julia (.jl) scripts in the ```./scripts``` directory

## License
<!-- [![License: CC BY 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png)](https://creativecommons.org/licenses/by/4.0/) -->

[<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png" style="width: 150px" >](https://creativecommons.org/licenses/by/4.0/)

This repository is licensed under a Creative Commons Attribution 4.0 International License.
