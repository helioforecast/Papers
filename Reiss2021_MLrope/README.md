# HELIO4CAST - MFRpred

This python package includes a predictive tool based on machine learning algorithms to estimate the Bz magnetic field component from upstream in situ observations of solar coronal mass ejections. A preprint of the manuscript accepted for publication in AGU Space Weather can be found on [arXiv](https://arxiv.org/abs/2108.04067).

#### Contributors
[Martin A. Reiss](https://www.oeaw.ac.at/iwf/staff/martin-august-reiss), [Rachel L. Bailey](https://github.com/bairaelyn), Ute V. Amerstorfer, Hannah Rüdisser, and [Christian Möstl](https://www.oeaw.ac.at/iwf/staff/christian-moestl). This research was funded by the Austrian Science Fund (FWF) and conducted at the Space Research Institute (IWF) in Graz, Austria.

#### Contacts
If you want to use the source code for your own research, please get in touch with us by email (christian.moestl@oeaw.ac.at) or by [chrisoutofspace](https://twitter.com/chrisoutofspace) on Twitter.


## 1. Machine learning  

    conda activate mfrpred
    jupyter lab
    
Run the mfrpred_mreiss_bz.ipynb and mfrpred_mreiss_btot notebooks in jupyter lab.    

---

## 2. Installation 

Install python 3.7.6 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

	  git clone https://github.com/helioforecast/Papers
      cd Reiss2021_MLrope

Create a conda environment for the mfrpred_... notebooks:

	  conda env create -f environment.yml

	  conda activate mfrpred

	  pip install -r requirements.txt
	  

Create a conda environment for the detection_... notebooks:

	  conda env create -f environment_detect.yml

	  conda activate detect

	  pip install -r requirements_detect.txt
	  

Before running the scripts, you need to download three data files (in total 1.8 GB) from this figshare repository, 

    https://doi.org/10.6084/m9.figshare.12058065.v8

and place them in the data/ folder.

    data/stereoa_2007_2021_sceq_ndarray.p
    data/stereob_2007_2014_sceq_ndarray.p
    data/wind_2007_2021_heeq_ndarray.p
        
A catalog for interplanetary coronal mass ejections (HELCATS ICMECAT v2.0) is included in this repo, for updates see: https://helioforecast.space/icmecat