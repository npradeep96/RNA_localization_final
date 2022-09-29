# README


This code is associated with the following [preprint](https://doi.org/10.1101/2022.09.19.508534) titled "Organization and regulation of nuclear condensates by gene activity".

Here we investigate the effects of localized RNA production on condensate properties.

## Requirements & Installation for Phase-field package

### TLDR Recipe (details below)

`conda create --name <MYFIPYENV> --channel conda-forge python=2.7 numpy scipy matplotlib pysparse mayavi weave`

`activate <MYFIPYENV>`

`pip install fipy`

`pip install moviepy` (This is for generating mp4 files)

`sudo apt-get install gifsicle` (This generates optimized GIF files)

`pip install imageio` (this is for unoptimized GIF files)

For simulations with circular meshes, one must install Gmsh (http://gmsh.info/). wget the appropriate OS and put the executable found in /bin/gmsh into the bin of the environment that you are using. This code has been tested on linux flavored OS and should install in a few minutes.

### Documentation

Navigate to **docs/build/html/index.html** and launch the file in browser to get user-friendly overview of the detailed documentation.

### Running the code

`python phase_field --i $input_file --o $output_folder --p $param_file`

$input_file contains the text-file with the input parameters (see demo/Input/input_params.txt for reference) in the following format

`param,value \n`

$output_folder contains the path to output_folder 

$param_file is the *only* optional input file & it contains a list of parameters to iterate over in the following format (see demo/Input/param_list.txt for reference):

`param`

`value1`

`value2`

### Demo

Example outputs upon running the phase field code using the parameter files in the directory `demo/Input/` are available in `demo/Output/`. The demo simulations can be run using the command

`python phase_field --i demo/Input/input_parameters.txt --o demo/Output/ --p demo/Input/param_list.txt`

The demo simulations are done to progressively by increasing the parameter $k_T$ across three values - 0.0001, 1, 100. As a comparison, plot 2A plots the same data with a much finer sampling of $k_T$. 

The simulation code also generates files called `spatial_variables.hdf5`, within each directory under `demo/Output/`, which contains the time trajectories of the concentration fields and other spatial variables. While your simulation code should output these files, we have not shown them in this github repo as they are quite large > 50 MB. 


### Long version
This code is written to employ the latest version of fipy (3.3.1) interfacing with py 2.7.15.

The [FiPy webpage](https://www.ctcms.nist.gov/fipy/INSTALLATION.html) has instructions on setting up a specific conda environment with important packages from conda-forge, and installing latest fipy through pip.

The visualization outputs are currently all over the place requiring 3 different packages - gifsicle, imagio, & moviepy. Future plans are to integrate all by only employing moviepy!

_P.S. It should likely work with python3 as well, though rigorous testing with other packages yet to be completed._
