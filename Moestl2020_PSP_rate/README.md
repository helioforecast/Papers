# ICME rates for solar cycle 25 and Parker Solar Probe observations

This code creates results, figures, and animations for the paper Möstl et al. (2020, ApJ).

There are 2 jupyter notebooks, 

(1)  cme_rate.ipynb  calculates the ICME rate for solar cycle 25


(2)  psp_3dcore.pynb simulates PSP double crossings with 3DCORE


  
  
## Running the code

You need to install a conda environment like demonstrated below in the installation instructions, and download the data files from this figshare repository: https://doi.org/10.6084/m9.figshare.11973693.v7
and place them in the 'data' folder. The version number is important as future updates might change these data sets.
  
Make sure you have activated the helio_paper environment, and run the jupyter notebook by opening

    jupyter lab
or

    jupyter notebook  

and select the script cme_rate.ipynb or psp_3dcore.ipynb and run it from within jupyter notebook or jupyter lab. The same code is available in the scripts cme_rate.py and psp_3dcore.ipynb.

Folder results/plots_rate contains the produced figures and animations (as mp4).


---
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
	  
