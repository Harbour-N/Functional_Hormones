# All work for Functional Hormones project

# Data

Data for this project was generated in this study: [https://www.science.org/doi/10.1126/scitranslmed.adg8464](https://www.science.org/doi/10.1126/scitranslmed.adg8464)

And the original dataset can be found here: [https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/5TW8YF](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/5TW8YF)

Data is stored in the `Data` folder.

- `md_data_controls.csv`  - Contains the main data, that is the time series of the recorded hormones for each participant. In total there is 214 participants (i.e., 214 unique MasterIDs).
- `Hormone_time_series_interpolated.csv` - This is the main file used for all the analysis in our paper. It is created from `md_data_controls.csv` by applying the preprocessing detailed in `Preprocess_data.qmd`. It contains all the participants with measuremnts interpolated to be in a common time frame for ease of analysis.
- `redcap_controls.csv` - The majority of the time series metadata is contained in redcap, eg: wake time, lunch time, physical activity etc..., for a full list of the metadat recorded see the file `Dictionary_redcap.csv`. Metadata is recored by particicpants over 3 days. Hormone sampling typically started in the morining of day 2 and therefore ends 24hrs later in the morning of day 3. This metadata is self recored by participants.
- `form_914_controls.csv` - Main phyisological metadata, columns of interest are: SamplingId, MasterId, GenderId, WAIST, HIP, WAIST_HIP_RATIO, SYSBP, DIABP, BMI, Age.
- `form_912_controls.csv` - More physiological metadata. Addition data includes smoker status and HEIGHT.
- `Metadata_all.csv` - combination of the 914 and 912 metadata files into a sinlge file.

Some of the column names are appreviations these stand for and take the following options:

* GenderId, male = 1 and female = 2
* WAIST = Waist circumfracne (cm).
* HIP = Hip circumfrrance (cm).
* WAIST_HIP_RATIO = WAIST/HIP
* SYSBP = Systolic blood pressure (mm Hg). Note: recordings were taken on a different day to the sampling session and as a single measurement.
* DIABP = Diastolic blood pressure (mm Hg). Note: recordings were taken on a different day to the sampling session and as a single measurement.
* WEIGHT = Body weight (Kg).
* BMI = Body mass index (Kg/m^2)
* ULT_Smoker = Smoker status, yes = 1, no = 2.
* HEIGHT = (cm)

# Code

## Data pre-processing

## PCA of hormone time series data

## Fiting Von Meises model
