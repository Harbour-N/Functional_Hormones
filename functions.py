import numpy as np
import pandas as pd
import datetime as dt
import skfda


#################################
# Special parameters
############


MasterID_2_days = 5081 # Has measuremtns from 2 days and unclear time stamps
MasterID_below_llq = 761 # Has one value of around 5 but all others are close to or below LLQ



################################## 
# Functions to preprocess the data
##################################


def to_integer(df):
    '''
    Convert SampleTime to number of minutes from midnight for easier manipulation,
    with reset on change of participant.
    '''
    df = df.copy()  # avoid modifying original df
    new_time = np.zeros(len(df))

    for i in range(len(df)):
        # Convert time to minutes since midnight
        current_time = df["SampleTime"].iloc[i]
        minutes = current_time.hour * 60 + current_time.minute

        if i == 0:
            new_time[i] = minutes
        elif df["MasterID"].iloc[i] == df["MasterID"].iloc[i - 1]:
            # assuming uniform 20 min intervals if same participant
            new_time[i] = new_time[i - 1] + 20
        else:
            # reset time for a new participant
            new_time[i] = minutes

    return new_time




def change_wake_time(input_pd):
    output_num_array = np.zeros(input_pd.shape)
    

    for i in range(len(input_pd)):

        # Use .iloc[] for positional indexing
        value = input_pd.iloc[i]

        # Check if it is NaN
        if pd.isna(value):
            output_num_array[i] = 0
        else:
            output_num_array[i] = int(input_pd[i][0:2])*60 + int(input_pd[i][3:])

    return output_num_array




# Function to load in the data and clean it up
def load_data(file_path):
    
    df = pd.read_csv(file_path)
    df['SampleTime'] =  pd.to_datetime(df['SampleTime'],dayfirst=True)
    # Sort the dataframe first by ID then by time
    df = df.sort_values(by=['MasterID', 'SampleTime'])
    hormones = ["18oxoF", "18OHF", "Aldo", "Cortisol", "Cortisone", "18OHCCS", "DHEAS", "21DOC", "11DOC", "CCS", "aTHF", "THF", "aTHE", "THE", "DXM", "THAldo", "Andro", "11deoxyCCS", "Testo", "DHEA", "17OHP", "EpiTesto", "DHT", "Prog"]
    keep_hormones = []
    l = len(df)
    threshold = 1 # set to 0.1 if want only 18OHF, Cortisol and Cortisone which are present with all measurements with only interpolation limit of 5. otherwise need large interpolation limit to get full dataset (no nans)
    for hormone in hormones:
        #print(hormone)
        nan_count = df[hormone].isna().sum()
        #print(nan_count/l)
        # Filter out hormones that have more than threshold % of their values as NA
        if nan_count / l > threshold: 
            df = df.drop(hormone, axis=1)
        else:
            keep_hormones.append(hormone)
            # interpolate NA values, the limit set the maximum number of consecutive NAs to interpolate over, which in the paper they say is 3
            df[hormone] = df[hormone].interpolate(limit = 5, limit_direction='both')
        
        
    # There is one participant for which we have 2 days of data
    # we will only consider a single day of data (first 72 measuremrnts)
    
    # MasterIDs = df['MasterID'].unique()
    # for MasterID in MasterIDs:
    #     if len(df[df['MasterID'] == MasterID]) > 72:
    #         # remove all but the first 72 measurements
    #         temp = df[df['MasterID'] != MasterID]
    #         df = pd.concat([temp, df[df['MasterID'] == MasterID].iloc[:72]], ignore_index=True)
    
    # Remove these two individuals from the dataset
    df = df[df['MasterID'] != MasterID_2_days]
    df = df[df['MasterID'] != MasterID_below_llq]

    # add newtime to dataframe
    df["NewTime"] = to_integer(df)

    # there are a handful of participants for which we have more than the 72 time points
    # We will filter these out so we only consider the first 72 time points
    df = df[df["SampleNo"] <= 72]

    # remove columns not interested in
    #df = df.drop('SampleTime', axis=1)
    #df = df.drop('SampleNo', axis=1)
    #df = df.drop('Unnamed: 0', axis=1)

    return df




def circle_shift(df):
    '''
    Ensures all data is in the range of 720–2160 minutes.
    Shifts early time points (<720) forward by 24 hours,
    and late ones (>=2160) back by 24 hours.
    '''

    new_df = pd.DataFrame()

    # Get a list of unique PIDs
    MasterIDs = df['MasterID'].unique()

    for MasterID in MasterIDs:
        # Work on a copy of each patient's data to avoid inplace issues
        patient_df = df[df['MasterID'] == MasterID].copy()

        # Shift early time points
        patient_df.loc[patient_df['NewTime'] < 720, 'NewTime'] += 1440
        # Shift late time points
        patient_df.loc[patient_df['NewTime'] >= 2160, 'NewTime'] -= 1440

        # Sort by new time
        patient_df = patient_df.sort_values(by="NewTime")

        # Concatenate
        new_df = pd.concat([new_df, patient_df], ignore_index=True)

    return new_df


               






# function that returns a df with a series of common time points for all patients for comparison across time
def common_time(df, interp_limit=5):



    # Shift the time to be in the range of 12-36
    

    # make a common set of time points from 12am(NewTime 12*60 = 720) ,till 12 am the next day (36*60 = 2160)
    common_times = np.arange(720,2160,20)

    common_df = pd.DataFrame()

    # Get a list of unique PIDs
    MasterIDs = df['MasterID'].unique()

    # First consider patients which have exactly 72 time points
    for MasterID in MasterIDs:

        # for each patient create a new df
        patient_df = df[df['MasterID'] == MasterID]

        # add the common times to the patient df
        # Iterate over common times and add them to the DataFrame if they don't exist
        for time in common_times:
            if time not in patient_df['NewTime'].values:
                # Create a new row as a DataFrame
                new_row = pd.DataFrame({'NewTime': [time], 'MasterID': [MasterID]})
                # Append the new row to the existing DataFrame
                patient_df = pd.concat([patient_df, new_row], ignore_index=True)

        
        patient_df = patient_df.sort_values(by="NewTime")

        # Interpolate to fill in the missing values at the common time points
        patient_df = patient_df.interpolate(limit=interp_limit, limit_direction='both')


        # remove rows that are not in the common time points
        patient_df = patient_df[patient_df['NewTime'].isin(common_times)]

        # append the patient_df to the common_df
        common_df = pd.concat([common_df, patient_df], ignore_index=True)

            


    #common_df = df.drop('NewTime', axis=1)
    
    

    return common_df


def df_to_numpy(df,hormone):
    '''
    Function to convert the dataframe to a numpy array, where each row is a patient with 72 measurements of a single hormone
    '''
    data = np.zeros((len(df['MasterID'].unique()), 72))
    for i, MasterID in enumerate(df['MasterID'].unique()):
        data[i] = df[df['MasterID'] == MasterID][hormone].values  

    return data


def df_to_fda_class(df_common):
    # Time grid common to all patients
    t = np.array(df_common["NewTime"])
    t = t[0:72]

    PIDs  = df_common['PID'].unique()

    data_matrix = []
    for PID in PIDs:
        test = df_common[df_common["PID"] == PID]
        test = np.array(test["Cortisol"])
        data_matrix.append(test)
        

    # define a new functional data class from hormone data
    fd = skfda.FDataGrid(
        data_matrix= data_matrix,
        grid_points=t,
    )

    return fd


# Map -pi to pi onto a range of 12-36
def map_pi_to_range(x):
    y = 12/np.pi * x + 24
    return y

def map_range_to_pi(y):
    x = np.pi/12*(y-24)
    return x



########################## 
# Registration of the data
##########################

def shift_register(num_array):
    '''
    Function to shift register of a numpy array of time series of hormone levels 
    so that there peaks are alligned at 7am
    '''
    peak  = np.argmax(num_array, axis=1)
    shift = 72 - peak - 21
    for i in range(len(shift)):
        num_array[i] = np.roll(num_array[i], shift[i])

    return num_array



def shift_reg_wake_time(num_array, wake_times):
    '''
    Function to shift register of a numpy array of time series of hormone levels 
    so that they are aligned to a common wake time
    '''

    output = np.zeros(num_array.shape)


    for i in range(len(wake_times)):

        # Find the closes index to the wake time
        wake_time_ind = np.argmin(np.abs(np.arange(720,2140+20,20) - wake_times[i]))
        shift = 72 - wake_time_ind -21
        output[i] = np.roll(num_array[i], shift)

    return output




#############################
# Fitting the basis functions
#############################



