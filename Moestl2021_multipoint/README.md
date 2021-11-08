## Multipoint ICME lineups in the Solar Orbiter era

Code repository for paper Möstl et al. 2021 ApJ.

Authors: C. Möstl, IWF Graz, Austria; twitter @chrisoutofspace; https://github.com/cmoestl

**last update November 8 2021**

To install a conda environment, dependencies are listed under environment.yml, and pip in requirements.txt. 

The multipoint ICME catalog is created with the jupyter notebook "lineups.ipynb".

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

	  conda activate helio_multi

	  pip install -r requirements.txt
	  
