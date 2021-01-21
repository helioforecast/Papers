# Ambient Solar Wind Prediction using Machine Learning

The Jupyter Notebook in this package details the training and implementation of a machine learning method, specifically a Gradient Boosting Regressor, to predict the ambient solar wind flows observed at Earth. The input data are variables from solar coronal magnetic models - the flux tube expansion factor and the distance to the coronal hole boundary along with the solar wind from one Carrington rotation before.

A manuscript based on this work has been submitted to a peer-reviewed journal for publication.

## Files

### ambsowi_forecast_ML.ipynb

This Jupyter Notebook details the work carried out. Exact steps are described there.

### optim_ambsowi_model.pickle

The output model from the machine learning algorithm. Unpack the pickle to access two variables: optim_model and optim_labels. Optim_labels defines the features you need to extract from a DataFrame (see notebook for making the DataFrame) to provide as featues to optim_model (xgboost.XGBRegressor object).

### data/...

Data needed to train the model and produce the results are saved here. Files that can be easily downloaded are listed by link in the notebook. The exception is the full training data set, which is based on output from the WSA model, which can not be shared openly. Output from other coronal magnetic models can be subbed as input if provided in the same format (see example file cmm_test.txt).

### plots/...

Collection of plots produced by notebook and used in the manuscript.

### results/...

Results produced by the various machine learning models that were considered and compared. Results for the [ISWAT ambient solar wind forecasting team](https://www.iswat-cospar.org/s2) have also been included here. The statistical analysis of the results was carried out on this data using the [OSEA matlab package](https://github.com/starsarestrange/solar-wind-forecast-verification/tree/v1.0).

## Usage

Parts 1-5 in the notebook detail the training and testing of the model. Statistical analysis is done with an external package. See part 6 in the notebook for exact implementation of the final model to new data.

## Authors

Author: R. L. Bailey (IWF Graz / ZAMG Vienna), 2019-2020.

## License

Licensed under the [MIT license](https://github.com/bairaelyn/ambsowi-ml/blob/master/LICENSE).
