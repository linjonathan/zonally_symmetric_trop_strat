# Zonally Symmetric Troposphere-Stratosphere

Solves for the coupled troposphere-stratosphere response to a SST perturbation, under QE dynamics. Modify the input parameters as you see fit, which will change the non-dimensional controlling parameters of the model. You will also need to set the meridional SST function, in sT.

Citation: [Lin et al. (2023)](https://arxiv.org/abs/2305.01110) (in review)

## Quick Start: TLDR
If you know what you are doing and familiar with Python packages, here is a quick start command list to get the model running.

    conda create -n trop_strat numpy matplotlib python scipy
    conda activate trop_strat
    pip install findiff
    
## Environment Set Up
We strongly recommend creating a virtual environment to install the packages required to run this model. A very popular virtual environment/package manager is [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#). 
<code>conda create -n trop_strat numpy matplotlib python scipy</code>
This requires conda to be installed, and should create a virtual environment with the name **trop_strat**. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details. Then, you need to install the package [findiff](https://findiff.readthedocs.io/en/latest/) via pip.
<code>pip install findiff</code>

Author: Jonathan Lin (jlin@ldeo.columbia.edu)
