import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import calendar
import json
import pprint

#Config can be as well loaded from json file via load_config function

config =  {
  "params": {
    'time'                                  : 'Date',
    'max_temp'                              : 'max_temp',
    'min_temp'                              : 'min_temp',
    'temp'                                  : 'Temperature (℃)',
    'max_dew_temp'                          : 'max_dew_temp',
    'min_dew_temp'                          : 'min_dew_temp',
    'dew_temp'                              : 'Dew point temperature (℃)',
    'max_wind_speed'                        : 'max_wind_speed',
    'wind_speed'                            : 'Wind speed (m/s)',
    "global_horiz_radiation"                : "OBS_GHI",
    "direct_normal_radiation"               : "dni_dirindex",
    
  },
  "weights": {
    "total": 20,
    'max_temp'                : 1,
    'min_temp'                : 1,
    'temp'                    : 2,
    'max_dew_temp'            : 1,
    'min_dew_temp'            : 1,
    'dew_temp'                : 2,
    'max_wind_speed'          : 1,
    'wind_speed'              : 2,
    "global_horiz_radiation"  : 5,
    "direct_normal_radiation" : 5,
  },

  "min_years_required": 9,

}

    
def load_config(path, filename):
    with open(f'{path}/{filename}.json', 'r') as f:
        config = json.load(f)
    return config


def load_csv(path, filename, config):
    params = config["params"]
    col_names = dict( zip(params.values(), params.keys()) )
    df =  pd.read_csv(f"{path}/{filename}.csv", 
                      usecols=col_names,
                      parse_dates=[params['time']])
    df.rename(columns=col_names,inplace=True)
    df.set_index('time', inplace=True)
   
    #check if all years in dataset have 12 monhts
    number_m = []
    year = np.unique(df.index.year).tolist()
    
    for m in range(1,13):
        number_m .append(len(np.unique(df[df.index.month==m].index.year)))

    zip_iterator = zip(year, number_m )
    years_dict = dict(zip_iterator)
     
    low_years = dict((k, v) for k, v in years_dict.items() if v < 12)
    
    
    if len(low_years) == 0:
        print("All years have 12 months")
        return df
    else:
        print("Following years doesn't have 12 months")
        print ("Year : Number of monhts") 
        for year in low_years:
            print (year,':',low_years[year])        
        
def ecdf(df, prop):
    a = df[prop].values
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return cusum / cusum[-1], x

def countlist(random_list):
    retlist = []
    count=1
    # Avoid IndexError for  random_list[i+1]
    for i in range(len(random_list) - 1):
        # Check if the next number is consecutive
        if abs(random_list[i] - random_list[i+1]) == 1:
            count += 1
        else:
            # If it is not append the count and restart counting
            retlist.append(count)
            count = 1
    # Since we stopped the loop one early append the last count
    retlist.append(count)
    if max(retlist) ==1:
        lenght = 0
    else:    
        lenght = max(retlist) 
    runs = sum(i > 1 for i in retlist)
    sumdict = {"runs" : runs, "max_lenght" : lenght}
    return sumdict 

def bestyear(df, years):
    """
    df is original dataframe
    years is a list of years for the best"""
   
    var = ["global_horiz_radiation", "temp"] #variables list

    bestyears = dict.fromkeys(years)
    
    for yr in years:
        df_year = df[df.index.year==yr].copy()
        
        df_year["day"] = df_year.index.day
        
        #for run lenths for GHI
        df_year = df_year.sort_values(var[0])
        Q1 = df_year[var[0]].quantile(0.33)
        Q3 = df_year[var[0]].quantile(0.67)
        
        Q3_Ghi = df_year[(df_year[[var[0]]] > (Q3) ).any(axis=1)]
        Q1_Ghi = df_year[(df_year[[var[0]]] < (Q1) ).any(axis=1)]
    
        #for run lenths for temperature
        df_year = df_year.sort_values(var[1])
        Q1 = df_year[var[1]].quantile(0.33)
        Q3 = df_year[var[1]].quantile(0.67)
        
        Q3_temp = df_year[(df_year[[var[1]]] > (Q3) ).any(axis=1)]
        Q1_temp = df_year[(df_year[[var[1]]] < (Q1) ).any(axis=1)]
        
        dic1 = countlist(Q3_temp.day.values)
        dic2 = countlist(Q1_temp.day.values)
        dic3 = countlist(Q1_Ghi.day.values)
        
        tot_dic = {"runs" : sum([dic1['runs'], dic2['runs'], dic3['runs']]) , 
                   "max_lenght" : max(dic1['max_lenght'], dic2['max_lenght'], dic3['max_lenght'])
                                                               }
        bestyears[yr] = tot_dic
    return bestyears

def selectYear(df, m, config):
    
    def keys_from_val(dataset_numpy, dict_tot):
        ar1 = dataset_numpy
        ar2 = []
        with np.nditer(dataset_numpy, op_flags=['readonly']) as it:
            for x in it:
                x1 = dict_tot[x.item()]
                ar2.append(x1)
        return ar1, ar2

    """
    Use the Sandia method, to select the most typical year of data
    for the given month
    """
    df = df.resample('d').mean()

    df = df[df.index.month==m]

    

    weights = dict(config['weights'])
    total = weights.pop('total')

    score = dict.fromkeys(df.index.year,0)
    fs = dict.fromkeys(weights)
    cdfs = dict.fromkeys(weights)
    
    
    for w in weights:
        cdfs[w] = dict([])
        fs[w] = dict([])

        # Calculate the long term CDF for this weight 
        # and create LUT for this function
        
        cdf_y, cdf_x  = ecdf(df,w)
        zip_iterator = zip(cdf_x,cdf_y)
        dict_tot =  dict(zip_iterator)
        
        #This is CDF for tot and safe to same dict as others
        cdfs[w]["all"] = cdf_y, cdf_x 

        for yr in set(df.index.year):
            df_year = df[df.index.year==yr]

            # calculate the CDF for this weight for specific year
            cdfs[w][yr] = ecdf(df_year,w)

            orig_val, CDF_tot_val_for_year = keys_from_val(cdfs[w][yr][1], dict_tot)
            CDF_tot_val_for_year = np.array(CDF_tot_val_for_year)
            cdfs[w][yr] = cdfs[w][yr]+ tuple(np.array([CDF_tot_val_for_year]))
            # Finkelstein-Schafer statistic (difference between long term
            # CDF and year CDF
            fs[w][yr] =  np.mean( abs(cdfs[w][yr][0] - cdfs[w][yr][2]) ) 

            # Add weighted FS value to score for this year
            score[yr] += fs[w][yr] * weights[w]/total


    # select the top 5 years ordered by their weighted scores

    top5 = sorted(score,key=score.get)[:5]

    pp = pprint.PrettyPrinter(depth=2, sort_dicts=False)
    print("These are runs and max_lenght of top5 years")
    pp.pprint(bestyear(df, top5))
    
    # TODO: select best year based on further statistics
    best_year = input("Please select best year:  ")

    return best_year, top5



   