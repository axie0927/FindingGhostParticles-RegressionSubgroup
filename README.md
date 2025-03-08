# FindingGhostParticles-RegressionSubgroup
Capstone B10-01

Contributors:
- Ammie Xie
- Haotian Zhu

Our Website can be found here: https://zhtdbb1.github.io/FindingGhostParticles-Website/

## Description
The goal of this project utilize the parameters extracted from the raw waveform time series data to train regression based models in order to predict the energy label variable. 

Models selected in this project include: Linear Regression Model, Ridge Regression Model, SVM Regression Model, Random Forest Regressor, and Neural Networks.

## Getting Started

### Step 1: Installation Instructions
How to clone the repository:
``` bash
git clone https://github.com/axie0927/FindingGhostParticles-RegressionSubgroup.git
```
In order to clone the dependencies needed for our project, please follow the steps below. Make sure you have Anaconda installed.<br><br>
### Step 2: Anaconda Environemnt Instructions
#### 1. Replace `name_of_environment` with a name you like:
``` bash
conda env create -f environment.yml --name name_of_environment
```
#### 2. Activate the environment:
``` bash
conda activate name_of_environment
```
### Step 3: Download the Proprocessed Dataset or Preprocess your own raw Data:
#### Option 1: Download the preprocessed dataset(Recommended):
1. Download the preprocessed data from this [link](https://drive.google.com/drive/folders/1SnmQemcXWPvKvJBmGkd0hSqTQ8gbs0C4), place all the csv files in the 'Data' folder under `src/Models` before running the .py files.
2. (Optional): If you would like to use the notebooks located at `src/Models/Notebooks`, also place the files from the link above in the 'Data' folder under `src/Models/Notebooks`.


#### Option 2: Proprocess your own data:
There are 25 different data files, and this data is not processed. In order to extract parameters from the data, download the raw data and run the Master.py script located in the src folder of the repository. The src folder also contains a parameter-functions folder with each parameter extraction function separately defined. Due to the large size of the data files, the processed data will not be kept in this repository.
1. Download the raw data at this [link](https://zenodo.org/records/8257027).
2. Create a directory at `src/Parameter Extraction` and name it 'data', place all the raw data in it.
3. Run the code below in your terminal:
``` bash
cd src/Parameter\ Extraction
```
``` bash
python3 Master.py
```
4. Place all the generated csv files in the 'Data' folder under `src/Models` before running the .py files.
5. (Optional): If you would like to use the notebooks located at `src/Models/Notebooks`, also place the files from the link above in the 'Data' folder under `src/Models/Notebooks`.

### Step 4: Apply the models on the processed dataset:
> **⚠️ Warning:** Make sure there are 4 files in the data folder -- `MJD_NPML_PCOCESSED.csv`,`MJD_TEST_PCOCESSED.csv`, `MJD_Train_PCOCESSED.csv`,`npml_cut.csv`. Where `npml_cut.csv` is the predictions of classification group.
> 
> **❗ Important:** `DeepLearning_NN.py` is our best final Model,not only it is applied on test set but also generates the predictions on NPML dataset which is the real world data without known true value, the others are applied only on test set for reference.
#### 1. Move to the Models directory:
``` bash
cd src/Models
```
#### 2. Replace `the_model_you_like.py` with the true model name:
``` bash
python3 the_model_you_like.py
```
After finishing the step above, please see your terminal for the results and guides 😉


## Models

Below is a list of models we built that are trained on our processed data (MJD_TRAIN_PROCESSED).

- **Linear Regression Model (Baseline Model)**

- **Ridge Regression Model**

- **SVM Regression Model**

- **Random Forest Regressor Model**

- **Neural Network Model (Final Model)**

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

## Exploratory Data Analysis

We performed an EDA on the processed data to figure out which features would be the best the use in our model training. This EDA utilizes a subset of the larger training data set, this file can be found named 'results.csv'. Note that it will not be used in the model building and we will be using the full data set for training.

## File Explanation
root/
- src/
  - Models/
    - Baseline_Model.py: Linear Regression
    - Ridge_Regression_Model.py: Ridge Regression
    - RF_Regressor_Model.py: Random Forest Regression 
    - SVM_Model.py: Support Vector Machine Regression
    - DeepLearning_NN.py: Neural Network(**Best**)
    - Data/
      - npml_cut.csv: Classification result from B10-2 (We are B10-1)
    - Notebooks/
      - Baseline_Model.ipynb: Linear Regression
      - Ridge_Regression_Model.ipynb: Ridge Regression
      - RF_Regressor_Model.ipynb: Random Forest Regression 
      - SVM_Model.ipynb: Support Vector Machine Regression
      - DeepLearning_NN.ipynb: Neural Network(**Best**)
      - Final Test.ipynb: For testing NN
      - Final Test_overfitting.ipynb: Test that aims on mitigating overfitting
      - Plots.ipynb: Generating combined or NPML dataset only Full Energy Spectrum and Energy spectrum after cut
      - Data/
        - npml_cut.csv: Classification result from B10-2 (We are B10-1)
  - EDA/
    - EDA.ipynb: Notebook for exploratory Analysis
    - results.csv: Necessary dataset for our EDA
  - Parameter Extraction/
    - Master.py: Main python file to extract feature from raw dataset
    - Remove_Duplicates.py: As its name suggests, ignore this file since we've already done this
    - Parameter-functions/: Parameter extraction files needed for Mater.py
  - assets/: Ignore this folder, needed for storing other files
- README.md
- Analysis_Unidoc.pdf: Copy of our report
- environment.yml: Anaconda Environment file
- requirements.yml: Substitute of environment.yml(Deprecated)

 

## Further Reading
[Majorana Demonstrator Data Release Notes](https://arxiv.org/pdf/2308.10856)
