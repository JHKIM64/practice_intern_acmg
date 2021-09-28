import datetime
from dateutil.parser import parse
import numpy as np
import xarray as xr
import pandas as pd

dataset = []

def saveModelData(period) :
    s_period = period.split("-")
    begin = parse(s_period[0])
    end = parse(s_period[1])
    while begin <= end :
        strDate = begin.strftime("%Y%m%d")
        path = '/veloce/jij/scherzo/GEOS_Chem_v12/yjo/RunN/Base/OutputDir/' \
                + 'GEOSChem.SpeciesConc.' + strDate+ '_0000z.nc4'
        ds = xr.open_dataset(path)
        # print("=============="+strDate+"===============")
        # print(ds.info())
        # print("==================================================")
        dataset.append(ds)
        begin += datetime.timedelta(days=1)

def chooseChemicals(chem,model_lat, model_lon) :
    day = 0
    loc_data = pd.DataFrame()
    while day < dataset.__len__() :
        data_1d = dataset[day]
        data_1d_mod = data_1d['SpeciesConc_'+ chem][:,0,model_lat,model_lon]
        time_1d=np.array(data_1d_mod['time'])
        conc_1d=np.array(data_1d_mod)*1000000

        if day==0 :
            loc_data = pd.DataFrame(conc_1d,time_1d)
        else :
            loc_data = loc_data.append(pd.DataFrame(conc_1d,time_1d))

        day += 1

    loc_data.columns = ['model_'+chem]
    return loc_data

def chooseLonLat(location) :
    longitude = np.array(dataset[0]["lon"])
    latitude = np.array(dataset[0]["lat"])
    # 관측지점을 감싸는 모델 지역 확인
    s_location = location.split(",")
    (lon,lat) = (float(s_location[1]),float(s_location[0]))
    lon_point = (np.abs(longitude-lon)).argmin()
    lat_point = (np.abs(latitude-lat)).argmin()
    return lat_point, lon_point

def run(period,chem,location) :
    saveModelData(period)
    model_Lat   ,model_Lon = chooseLonLat(location)
    concentration = chooseChemicals(chem,model_Lat,model_Lon)
    # plot 부분
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(concentration)
    # plt.savefig("./"+chem+"_"+period +".png" )
    return concentration

run("20160429-20160610","NO2","37.519662,127.122418")