# L5 Dst Prediction

This Jupyter Notebook presents the methodology used to test and validate an approach for providing a 4.5-day 
forecast of the geomagnetic Dst index at Earth using data measured at the L5 point. In this study, the STEREO-B 
satellite was used as a proxy for a true L5 mission.

## Installation instructions

Install python 3.7.6 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

Create conda environment:

	  conda env create -f environment.yml
	  
	  source activate jupydst

Run

	  jupyter notebook
