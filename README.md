# All work for Functional Hormones project

# Data

Data for this project was generated in this study: [https://www.science.org/doi/10.1126/scitranslmed.adg8464](https://www.science.org/doi/10.1126/scitranslmed.adg8464)

And the original dataset can be found here: [https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/5TW8YF](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/5TW8YF)

Data is stored in the `Data` folder.

- `md_data_controls.csv`  - Contains the main data, that is the time series of the recorded hormones for each participant. In total there is 213 participants (i.e., 213 unique MasterIDs).
- `Hormone_time_series_interpolated.csv` - This is the main file used for all the analysis in our paper. It is created from `md_data_controls.csv` by applying the preprocessing detailed in `Preprocess_data.qmd`. It contains all the participants with measurements interpolated to be in a common time frame for ease of analysis.
- `redcap_controls.csv` - The majority of the time series metadata is contained in redcap, eg: wake time, lunch time, physical activity etc..., for a full list of the metadat recorded see the file `Dictionary_redcap.csv`. Metadata is recoded by participants over 3 days. Hormone sampling typically started in the morning of day 2 and therefore ends 24hrs later in the morning of day 3. This metadata is self recoded by participants.
- `form_914_controls.csv` - Main physiological metadata, columns of interest are: GenderId, WAIST, HIP, WAIST_HIP_RATIO, SYSBP, DIABP, BMI, Age.
- `form_912_controls.csv` - More physiological metadata. Addition data includes smoker status and HEIGHT.
- `Metadata_merge.csv` - combination of the 914 and 912 metadata files into a single file.

Some of the column names are appreciations these stand for and take the following options:

* GenderId, male = 1 and female = 2
* WAIST = Waist circumference (cm).
* HIP = Hip circumference (cm).
* WAIST_HIP_RATIO = WAIST/HIP
* SYSBP = Systolic blood pressure (mm Hg). Note: recordings were taken on a different day to the sampling session and as a single measurement.
* DIABP = Diastolic blood pressure (mm Hg). Note: recordings were taken on a different day to the sampling session and as a single measurement.
* WEIGHT = Body weight (Kg).
* BMI = Body mass index (Kg/m^2)
* ULT_Smoker = Smoker status, yes = 1, no = 2.
* HEIGHT = (cm)

Note the metadata file contain an extra MasterID for which we do not have hormone data for (MasterID = 5071) so we remove them in the preprocessing stage.

# Code

## Data pre-processing

## Fitting Von Meises model

In the file `fitting.qmd` we fit our relatively simple functional data analysis model of cortisol. This comprises of the sum of 3 basis functions and weights. Since each von Meisis distribution has 3 parameters (Mean, amplitude, standard deviation / dispersion) this model has 9 parameters in total (reducing the amount of data needed to fully describe the model from 72 time points to 9 parameters).

The output of this notebook is a csv file contianing all the fitted parameters for each participant `Param_values.csv`. The parameters are save in a list or given individually in both cases:

- param0 = mean (basis1)
- param1 = amplitude (basis1)
- param2 = dispersion (basis1)
- param3 = mean (basis2)
- param4 = amplitude (basis2)
- param5 = dispersion (basis2)
- param6 = mean (basis3)
- param7 = amplitude (basis3)
- param8 = dispersion (basis3)

## PCA of hormone time series data

## Fitting Von Meises model
