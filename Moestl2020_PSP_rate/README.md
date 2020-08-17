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
      
      cd Papers/Moestl_2020_

Create conda environment:

	  conda env create -f environment.yml
	  
	  conda activate helio_paper
	  
	  pip install -r requirements.txt
	  
	  

You need to download the data files from this figshare repository: https://doi.org/10.6084/m9.figshare.11973693.v7
(The version number is important as future updates might change these data sets.)

data/wind_2018_2019_heeq.p
data/wind_2007_2018_heeq_helcats.p
data/psp_2018_2019_sceq.p
data/stereoa_2007_2019_sceq.p
data/stereoa_2019_2020_sceq_beacon.p
data/stereob_2007_2014_sceq.p
data/maven_2014_2018_removed_smoothed.p
data/ulysses_1990_2009_rtn.p
data/vex_2007_2014_sceq_removed.p
data/messenger_2007_2015_sceq_removed.p
  
and place them in the 'data' folder.

  
Run with

    jupyter lab
or	  
	jupyter notebook  
	
and select the script cme_rate.ipynb and run it from within jupyter notebook or jupyter lab. 
	  
