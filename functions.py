import numpy as np
import pandas as pd
import datetime as dt
import skfda

################################## 
# Functions to preprocess the data
##################################

def to_integer(df):
    '''
    Function to convert the time (from a datatime object) to an integer (of mins) for easier manipulation and add it to a column in the dataframe
    '''
    
    mins = np.zeros(len(df))
    hrs = np.zeros(len(df))
    new_time = np.zeros(len(df))
    
    #start_time = int(df["SampleTime"][0][-8:-6]) + int(df["SampleTime"][0][-5:-3])/60
    start_time = int(df["SampleTime"][0][-8:-6])*60 + int(df["SampleTime"][0][-5:-3])

    new_time[0] = start_time

    
    for i in range (1,len(df)):
        
        if df["PID"][i] == df["PID"][i-1]:
            # each successive time is 20 mins after the previous one
            #new_time[i] = new_time[i-1] + 20/60
            new_time[i] = new_time[i-1] + 20
        else:
            # restart the timer
            #new_time[i] = int(df["SampleTime"][i][-8:-6]) + int(df["SampleTime"][i][-5:-3])/60
            new_time[i] = int(df["SampleTime"][i][-8:-6])*60 + int(df["SampleTime"][i][-5:-3])
        
    return   new_time

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
    df['SampleTime'] = pd.to_datetime(df['SampleTime']).apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S')) # make sure to format the dates like this!!!

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
        
        
        
    
    # add newtime to dataframe
    df["NewTime"] = to_integer(df)

    # remove columns not interested in
    #df = df.drop('SampleTime', axis=1)
    #df = df.drop('SampleNo', axis=1)
    df = df.drop('Unnamed: 0', axis=1)

    return df

# function that returns a df with a series of common time points for all patients for comparison across time
def common_time(df, interp_limit):

    # Remove some columns that are not needed
    df = df.drop('SampleTime', axis=1)
    df = df.drop('SampleNo', axis=1)
    df = df.drop('SID', axis=1)

    # make a common set of time points from 12am(NewTime 12*60 = 720) ,till 12 am the next day (36*60 = 2160)
    common_times = np.arange(720,2160,20)

    common_df = pd.DataFrame()

    # Get a list of unique PIDs
    PIDs = df['PID'].unique()

    # First consider patients which have exactly 72 time points
    for PID in PIDs:

        # for each patient create a new df
        patient_df = df[df['PID'] == PID]

        # add the common times to the patient df
        # Iterate over common times and add them to the DataFrame if they don't exist
        for time in common_times:
            if time not in patient_df['NewTime'].values:
                # Create a new row as a DataFrame
                new_row = pd.DataFrame({'NewTime': [time], 'PID': [PID]})
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
    data = np.zeros((len(df['PID'].unique()), 72))
    for i, PID in enumerate(df['PID'].unique()):
        data[i] = df[df['PID'] == PID][hormone].values  

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



