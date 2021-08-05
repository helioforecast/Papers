## Multipoint ICME lineups in the Solar Orbiter era

Code repository for paper Möstl et al. 2021 ApJ (in prep.).

Authors: C. Möstl, IWF Graz, Austria; twitter @chrisoutofspace; https://github.com/cmoestl

**work in progress, last update August 2021**

To install a conda environment, dependencies are listed under environment.yml, and pip in requirements.txt. Plots are saved in folder "plots" as png and pdf. 

See lineups.ipynb 

---


## License

Licensed under the MIT License.

---

## Installation 

Install python 3.7.6 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

	  git clone https://github.com/helioforecast/papers
	  cd Moestl2021_multipoint

Create a conda environment using the environment.yml and requirements.txt file in the heliocats root directory, and activate the environment in between:

	  conda env create -f environment.yml

	  conda activate multi

	  pip install -r requirements.txt
	  
