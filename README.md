# FindingGhostParticles-RegressionSubgroup
Capstone B10-01

Contributors:
- Ammie Xie
- Haotian Zhu

This repository is made for the checkpoint submission, further work will be merged and pushed to the main repository found [here](https://github.com/matthewsegovia/MajoranaNeutrinoHunt/tree/main).

## Description
The goal of this project utilize the parameters extracted from the raw waveform time series data to train regression based models in order to predict the energy label variable. 

Models selected in this project include: Linear Regression Model, Ridge Regression Model, SVM Regression Model, Random Forest Regressor, and Neural Networks.

## Installation Instructions
How to clone the repository:
``` bash
git clone https://github.com/axie0927/FindingGhostParticles-RegressionSubgroup.git
``` 

## Exploratory Data Analysis

We performed an EDA on the processed data to figure out which features would be the best the use in our model training. This EDA utilizes a subset of the larger training data set, this file can be found named 'results.csv'. Note that it will not be used in the model building and we will be using the full data set for training.

## Models

Below is a list of models we built that are trained on our processed data (MJD_TRAIN_PROCESSED).

- **Linear Regression Model (Baseline Model)**

- **Ridge Regression Model**

- **SVM Regression Model**

- **Random Forest Regressor Model**

- **Neural Network Model**

## Parameters
Below is a list of all the parameters extracted from the raw data as well as a brief description of them. 

- **Drift Time** (tdrift.py): The time taken from the initiation of charge generation to the collection at the detector's point contact at increments of 10%, 50% and 99.9%.

- **Late Charge** (lq80.py): The amount of energy being collected after 80% of the peak. 

- **Late Charge Slope** (Area Growth Rate (agr.py)): The integrated drift time of the charge collected after 80% of the waveform. 

- **Second derivative Inflection Points** (inflection.py): The amount of inflection points from 80% of our charge to the peak. 

- **Rising Edge Slope** (rising_edge.py): The slope of the charge that was recorded.

- **Rising Edge Asymmetry** (rea.py): This function measures how tilted in a direction the rising edge of the signal is.

- **Current Amplitude** (current_amplitude.py): The peak rate of charge collection, defined as I = dq/dt which means current amplitude is the derivative of charge.

- **Energy Peak** (peakandtailslope.py): The maximum analog-to-digital (ADC) count. The height of this peak correlates with the energy deposited by the particle in the detector.

- **Tail Slope** (peakandtailslope.py): The rate of charge collection over the length of the waveform’s tail. It indicates how quickly charge dissipates in the detector after the initial interaction.

- **Delayed Charge Recovery** (dcr.py): The rate of area growth in the tail slope region. This is measured by the area above the tail slope to the peak of the rise. 

- **Fourier Transform and Low Frequency Power Ratio** (fourier_lfpr.py): The Fourier Transform is a mathematical operation that transforms a time-domain signal into its frequency-domain representation. Low Frequency Power Ratio (LFPR) is used, quantifying how much of the signal’s energy is concentrated in the low-frequency threshold by the total power spectrum of the Fourier transformed waveform.  

The Master.py file combines all these parameters into one file. Remove_Duplicates.py removes all duplicate rows in the processed files. 

## Reproducing The Code
The data can be downloaded at this [link](https://zenodo.org/records/8257027). There are 25 different data files, and this data is not processed. In order to extract parameters from the data, download the raw data and run the Master.py script located in the src folder of the repository. The src folder also contains a parameter-functions folder with each parameter extraction function separately defined. Due to the large size of the data files, the processed data will not be kept in this repository. The processed data can be found in this [Google Drive](https://drive.google.com/drive/folders/1SnmQemcXWPvKvJBmGkd0hSqTQ8gbs0C4).

## How To Use The Notebook?
Download the data at this [link](https://drive.google.com/drive/folders/1SnmQemcXWPvKvJBmGkd0hSqTQ8gbs0C4), put the MJD_TRAIN_PROCESSED and MJD_TEST_PROCESSED in the 'Data' folder under src/Models before running the notebook. 

## Further Reading
[Majorana Demonstrator Data Release Notes](https://arxiv.org/pdf/2308.10856)