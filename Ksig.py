import numpy as np
import matplotlib.pyplot as plt
import ksig
from functions import load_data

df, hormones, df_com, num_com = load_data("Hackathon/MDdatahackathontraining_1342743918.csv")

# Time grid common to all patients
t = np.array(df_com["NewTime"])
t = t[0:72]


num_patients_in_each_time = df_com.groupby("NewTime").size()



PIDs = df['PID'].unique()
PIDs_com  = df_com['PID'].unique()


def comparison_to_cortisol(other_hormone: str):
    cortisol_matrix = []
    other_matrix = []
    for PID in PIDs_com:
        test = df_com[df_com["PID"] == PID]
        cortisol = np.array(test["Cortisol"])
        cortisol = cortisol/np.nanstd(cortisol)
        cortisol_matrix.append(cortisol)
        other = np.array(test[other_hormone])
        other = other/np.nanstd(other)
        other_matrix.append(other)

    cortisol_matrix = np.float32(np.array(cortisol_matrix))
    cortisol_matrix=cortisol_matrix[:75,:]
    other_matrix = np.float32(np.array(other_matrix))
    other_matrix=other_matrix[:75,:]

    # Number of signature levels to use.
    n_levels = 5 

    # Use the RBF kernel for vector-valued data as static (base) kernel.
    static_kernel = ksig.static.kernels.RBFKernel() 

    # Instantiate the signature kernel, which takes as input the static kernel.
    sig_kernel = ksig.kernels.SignatureKernel(n_levels, static_kernel=static_kernel)


    K_XY = sig_kernel(cortisol_matrix, other_matrix)
    return K_XY


def comparison_figs():
    aldo = comparison_to_cortisol("Aldo")
    OHF = comparison_to_cortisol("18OHF")
    cortisone = comparison_to_cortisol("Cortisone")

    plt.subplot(1,3,1)
    plt.imshow(aldo)
    plt.title('Aldo')

    plt.subplot(1,3,2)
    plt.imshow(OHF)
    plt.title('18OHF')

    plt.subplot(1,3,3)
    plt.imshow(cortisone)
    plt.title('Cortisone')

    plt.savefig('comparisons.png')