## KORUS-AQ Data form

#Platform
platform = ""

#Location
location = ""

#Instrument
instrument =""

#Uncertainty
uloid_flog = 0
ulod_value = 0
llod_flag = 0
llod_value = 0

#DATA
#file_dataframe

############################################
data_infos = ["PLATFORM","LOCATION","INSTRUMENT_INFO","ULOD_FLAG","ULOD_VALUE","LLOD_FLAG","LLOD_VALUE"]
data = []
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def korus_analyze(path):
    file_path = path
    file = open(file_path, 'r')

    data_ord =  0
    target_num = 0
    while data_ord < len(data_infos):
        line = file.readline()
        info = data_infos[data_ord]

        if line.startswith(info) :
            rline = line.replace(":","").strip()
            data.append(rline)
            data_ord += 1

        elif line.startswith("DOY_UTC") :
            while True :
                line = file.readline()
                if line.startswith('0'):
                    break

                slice = line.replace(",","").split(" ")
                data_infos.insert(data_ord, slice[0])
                data.append(slice[1])
                data_ord += 1
                target_num += 1

    label = []
    while True:
        line = file.readline()
        if line.startswith("Start"):
            label.append(line)
            break

    leftLines = file.readlines()

    targets = label + leftLines

    for i in range(0, len(targets)):
        targets[i] = targets[i].replace(" ", "").strip().split(',')

    df_target = pd.DataFrame(targets[1:-1])
    df_target.columns = targets[0]
    df_target = df_target.drop(['Start_UTC', 'Stop_UTC', 'Mid_Time'], axis=1)
    df_target['DOY_UTC'] = pd.to_datetime(df_target['DOY_UTC'])
    df_target.set_index('DOY_UTC', inplace=True)
    df_target = df_target.apply(pd.to_numeric)
    df_target = df_target.replace(-9999, np.NaN)
    print(df_target.head())
    print(df_target.info())
    ########################################################################
    # plotting

    fig, ax = plt.subplots(figsize=(16, 10))

    for i in range(0,target_num):
        ax.plot(df_target[data_infos[i]],label=data_infos[i])
    plt.legend()
    plt.show()
    ###########################################################################
    # correlation

    f, ax = plt.subplots(figsize=(16, 20))
    correlation = df_target.corr()
    print(correlation)

    sns.heatmap(correlation, mask=np.zeros_like(correlation, dtype=np.bool), cmap='RdYlGn_r', ax=ax, annot=True)
    plt.show()


##########################################################################
korus_analyze('/home/hmk/Korus_AQ/ict/O3.ict')
