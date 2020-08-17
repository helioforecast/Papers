# ICME rates for solar cycle 25 and Parker Solar Probe observations

Figures and results for the paper Möstl et al. (2020, ApJ).

## Installation instructions

Install python 3.7.6 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

      git clone https://github.com/helioforecast/Papers
      
      cd Papers/Moestl2020_PSP_rate

Create conda environment and install all packages:

	  conda env create -f environment.yml
	  
	  conda activate helio_paper
	  
	  pip install -r requirements.txt
	  
	  

You need to download the data files from this figshare repository: https://doi.org/10.6084/m9.figshare.11973693.v7
and place them in the 'data' folder. The version number is important as future updates might change these data sets.
  
  
## Running the code
  
Make sure you have activated the helio_paper environment, and run the jupyter notebook with 

    jupyter lab
or

    jupyter notebook  

and select the script cme_rate.ipynb and run it from within jupyter notebook or jupyter lab. The same code is available as the script cme_rate.py 

Folder results/plots_rate contains the produced figures.
	  
