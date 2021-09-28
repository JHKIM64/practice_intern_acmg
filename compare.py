#D-4
import sys

import numpy as np
import pandas as pd
import datetime
from dateutil.parser import parse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geoschemenv
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
## KORUS-AQ Data form
#DATA
#file_dataframe

############################################
data_infos = ["PLATFORM","LOCATION","INSTRUMENT_INFO","ULOD_FLAG","ULOD_VALUE","LLOD_FLAG","LLOD_VALUE"]
data = []
###########################################3
#관측자료 가져오기
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
    df_target['DOY_UTC'] = pd.to_datetime(df_target['DOY_UTC'])+datetime.timedelta(minutes=30)
    df_target.set_index('DOY_UTC', inplace=True)
    df_target = df_target.apply(pd.to_numeric)
    df_target = df_target.replace(-9999, np.NaN)

    return df_target, target_num

    ########################################################################
    # plotting
font1 = {'family': 'serif',
      'color':  'darkred',
      'weight': 'normal',
      'size': 16}

def plot(target_obs,target,acc,nmb) :
    # 모델자료 가져오기
        pd.plotting.register_matplotlib_converters()
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(target_obs.index,target_obs["model_"+target],label="model_"+target)
        ax.plot(target_obs.index,target_obs[target], label=target)

        plt.ylabel("ppm")
        plt.legend()
        print("###################")
        print(target + " compare with model value")
        print('RMSE : '+str(acc))
        print("NMB : "+str(nmb) )
        print("###################")
        # plt.savefig("./compare"+target+".png")
        plt.show()

#########################################################################
##모델정확도를 뭐로 측정하지..
def rmse(data, target) :
    data = data.dropna(axis=0)
    return np.sqrt(mean_squared_error(data["model_"+target], data[target]))

def nmb(data, target) :
    data = data.dropna(axis=0)
    data_model = np.array(data["model_"+target]).reshape(len(data["model_"+target]),1)
    data_obs = np.array(data[target]).reshape(len(data[target]),1)
    diff = data_obs-data_model
    return diff.mean()
##########################################################################
def run(obs_target,target) :

    #모델 계산값
    model_target = geoschemenv.run("20160429-20160610", target, "37.519662,127.122418")

    obs_target = obs_target.loc[:, [col for col in obs_target.columns if col == target]]
    data_merge = obs_target.join(model_target,how='outer')
    acc=rmse(data_merge, target)
    nm = nmb(data_merge,target)
    plot(data_merge,target,acc,nm)
###########################################################################
def chooseChemicals() :
    # 관측값
    obs_target, _ = korus_analyze('/home/hmk/Korus_AQ/ict/O3.ict')

    while True :
        print("chemical : ")
        chemical = sys.stdin.readline().strip()
        if chemical == "exit" :
            break;

        run(obs_target, chemical)

chooseChemicals()