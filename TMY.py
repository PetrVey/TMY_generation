import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import calendar
import json
import pprint
from sklearn.metrics import mean_squared_error

#This is config for TMY3 based on NREL (US)
#Config can be as well load from json file
config =  {
    #Rename parametrs
  "params": {
     #New name in DF: Original DF name
    'time'                                  :      'Date',
    'max_temp'                              :     'max_temp',
    'min_temp'                              :'min_temp',
    'temp'                                  :  'Temperature (℃)',
    'max_dew_temp'                          : 'max_dew_temp',
    'min_dew_temp'                          :'min_dew_temp',
    'dew_temp'                              :'Dew point temperature (℃)',
    'max_wind_speed'                        : 'max_wind_speed',
    'wind_speed'                            :'Wind speed (m/s)',
    "global_horiz_radiation"                : "OBS_GHI",
    "direct_normal_radiation"               : "dni_dirindex",
    
  },
   #Importance weight for FS-statistic method 
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

    #if all years have 12 monhts it returns df that is used for TMY method
    if len(low_years) == 0:
        print("All years have 12 months")
        return df
    else: 
        #else, it prints the years that dont have 12 monhts for further cleaning
        print("Following years doesn't have 12 months")
        print ("Year : Number of monhts") 
        for year in low_years:
            print (year,':',low_years[year])        
        
def ecdf(df, prop):
    """
    Parameters
    ----------
    df : dataframe
    prop : property from wchi cdf should be created 

    """
    a = df[prop].values
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return cusum / cusum[-1], x

def countlist(random_list):
    #####
    #Just a simple count list function for consecutive runs
    #It works only with integers, as it was intended for date.day
    #####
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
    Parameters
    ----------
    df : original dataframe
    years: list of the best-fit years (eg. top5)
    
    Returns
    -------
    bestyears: statistics dictionary needed for selection of the best-fit years 
                including runs, lenght, std, RMSE
    """
   
    #variables  for selection
    var = ["global_horiz_radiation", "temp"] #add temp later

    #create dictionary
    bestyears = dict.fromkeys(years)
    
    #longterm daily average, eg. 1st of January average from multiple years
    df_longterm_daily_avg = df.groupby('{:%m-%d}'.format).mean()
    
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
        
        #Standard deviation for year for GHI
        deviation = np.round(np.std(df_year[var[0]]),decimals=1)
        
        #RMSE is calculated to long-term daily for GHI
        y_long = df_longterm_daily_avg[var[0]]
        y_year = df[df.index.year==yr].copy()[var[0]]
        rms = np.round(mean_squared_error(y_long, y_year, squared=False),decimals=1)
        
        #return dictionary with runs, lenght, std and RMSE
        tot_dic = {"runs" : sum([dic1['runs'], dic2['runs'], dic3['runs']]) , 
                   "max_lenght" : max(dic1['max_lenght'], dic2['max_lenght'], dic3['max_lenght']),
                   "GHI_std": deviation,
                   "GHI_RMSE": rms }
        
        bestyears[yr] = tot_dic
    return bestyears

def keys_from_val(dataset_numpy, dict_tot):
    ar1 = dataset_numpy
    ar2 = []
    with np.nditer(dataset_numpy, op_flags=['readonly']) as it:
        for x in it:
            x1 = dict_tot[x.item()]
            ar2.append(x1)
    return ar1, ar2


def selectYear(df, m, config):
    """
    Use the Sandia method, to select the most typical year of data
    for the given month
    """
    df = df.resample('d').mean()
    
    df = df[df.index.month==m]
    
    #delete leap years
    df = df[~((df.index.month == 2) & (df.index.day == 29))]

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
    top5_statistics = bestyear(df, top5)
    pp.pprint(top5_statistics)
    
    #select best year based on further statistics
    #best_year = input("Please select best year:  ")
    best_year = Sandia_selection(top5_statistics)

    return best_year, top5


def plotCdfs(df, m, config, selectedyear):
    """
    basically repeat the process of the FS statitiscs
    and then plot the CDFs of TMY selected year, best-fit year
    and long-term average
    """
    df = df.resample('d').mean()
    df = df[df.index.month==m]
    #delete leap years
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
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
    
    for w in weights:
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(cdfs[w]["all"][1],cdfs[w]["all"][0], color="black", linewidth=2, label="Long-term")
        #for yr in set(df.index.year):
        years_l = list(set(df.index.year)) 
        years_l.sort()
        
        best_w = sorted(fs[w],key=fs[w].get)[0]
        worst_w = sorted(fs[w],key=fs[w].get)[len(years_l)-1]
        
        print(f"{best_w}  {w}")
        for yr in years_l:
            if yr == selectedyear:
                plt.plot(cdfs[w][yr][1],cdfs[w][yr][0], label = f'TMY Selected ({yr})', 
                         color="green", linewidth=1.5, linestyle="solid", 
                         marker="x", markersize=4)
            elif yr ==  best_w:
                plt.plot(cdfs[w][yr][1],cdfs[w][yr][0], label =  f'Best ({yr})', 
                         color="blue", linewidth=1, linestyle="-.", 
                         marker="o", markersize=4, markerfacecolor="none")
                
            elif yr == worst_w:
                plt.plot(cdfs[w][yr][1],cdfs[w][yr][0], label =f'Worst ({yr})', 
                         color="orange", linewidth=1, linestyle="--", 
                         marker="s", markersize=4, markerfacecolor="none")
                
        plt.ylim(0,1)
        plt.title(f"{w} ({calendar.month_name[m]})")
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        plt.show()

def mergeMonths(df_orig, tmy_m):
    """

    Parameters
    ----------
    df_orig : original dataframe that contains all years
    tmy_m : dataframe with TMY months in pd df with rows year and month_name

    Returns
    -------
    newTMY : TMY datafile 

    """
    tmy_list = []
    for index, row in tmy_m.iterrows():
        y = row["Year"]
        m = row["Month_name"]
        df_tmy_m = df_orig[f"{y}-{m}"]
        tmy_list.append(df_tmy_m)
        
    newTMY = pd.concat(tmy_list)
        
    return newTMY

def Sandia_selection(top5):
    """
    Parameters
    ----------
    top5 : List of Top 5 years with runs and lenght counts

    Returns
    -------
    Selectedyear: Selected year for TMY for specific month
    
    """
    
    # List of Top5 years to pandas df
    # where the index is first rename to year and then reset
    # to maintain integer style index
    top5_df = pd.DataFrame.from_dict(top5 ,orient='index').rename_axis(index='year').reset_index()
    
    #First, check if there is a year with 0 runs, then drop it
    zero_runs_idx = [i for i, j in enumerate(top5_df["runs"] == 0) if j == True]
    top5_df = top5_df.drop(top5_df.index[zero_runs_idx])  
    #need to restart index after every drop, otherwise it will drop incorect one in next iteration
    top5_df = top5_df.reset_index().iloc[:, 1:]
    
    #Second, drop the year with max runs
    #if there are equal number of runs, then drop the last in the serie
    max_runs = max(top5_df["runs"])
    max_runs_idx = [i for i, j in enumerate(top5_df["runs"]) if j == max_runs]
    top5_df = top5_df.drop(top5_df.index[max(max_runs_idx)])
    #need to restart index after every drop, otherwise it will drop incorect one in next iteration
    top5_df = top5_df.reset_index().iloc[:, 1:]
    
    #Third, drio the yer with max run lenght
    #if there are more years with max run lenght, drop the last in the serie
    max_lenght = max(top5_df["max_lenght"])
    max_lenght_idx = [i for i, j in enumerate(top5_df["max_lenght"]) if j == max_lenght]
    top5_df = top5_df.drop(top5_df.index[max(max_lenght_idx)])
    #need to restart index after every drop, otherwise it will drop incorect one in next iteration
    top5_df = top5_df.reset_index().iloc[:, 1:]
    
    #Finaly, there is df with 3 or 2 years
    #here we pick the one with lowest index, 
    # i.e. the one with previously lowest FS index

    Selected_year = top5_df["year"][0]
    
    return Selected_year 




