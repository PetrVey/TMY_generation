# TMY_generation
Typical Meteorological Year (TMY) generation  <br />

Python script for calculation of TMY based on Sandia methodology <br />

"Dataset_for_tmy.csv" is a test dataset. <br />
Necessary columns within this dataset based on Sandia-NREL TMY3 method are :  <br />
{'Date', 'max_temp', 'min_temp', 'Temperature (℃)', max_dew_temp', 'min_dew_temp',  <br />
'Dew point temperature (℃)',  'max_wind_speed', 'Wind speed (m/s)',  "OBS_GHI",  "dni_erbs"} <br />

Data sources & references:<br />
[1] Meteorological variables: Central Weather Bureau (TAIWAN)  <br />
[2] GHI is satellite-derived from MTSAT-1, MTSAT-2 and Himawari-8 based on https://doi.org/10.1016/j.renene.2022.01.027 <br />
[3] Erbs separation model : https://doi.org/10.1016/0038-092X(82)90302-4 <br />
[4] NREL TMY3 Sandia methodlogy: Wilcox, S., Marion, W., 2008. Technical Report NREL/TP-581-43156: Users Manual for TMY3 Data Sets. Colorado


Example of use

path = "D:/"
filename = "Dataset_for_tmy"
df = load_csv(path, filename, config)


TMY_months = {}
for m in range(1,13,1):
    print("getting " + calendar.month_name[m])
    tmyyear, top5months, top5_st = selectYear(df, m, config, model="Sandia")
    TMY_months[calendar.month_name[m]] = tmyyear
    
tmy_m = pd.DataFrame({'Month_name':list(TMY_months.keys()), 'Year': list(TMY_months.values())} )
tmy_m.to_csv("D:/Petr/Papers/2022_TMY/TMYmonths.csv")


df_orig = pd.read_csv(f"{path}/{filename}.csv", parse_dates= ["Date"]).iloc[:,1:].set_index('Date')
df_orig  = df_orig [~((df_orig.index.month == 2) & (df_orig.index.day == 29))] 

df_TMY = mergeMonths(df_orig, tmy_m)    
    
testpath = "D:/Figures/TMY"  
plotCdfs(df, 1, config, 2010, daily=True, save=True, pathfig=testpath)

smoothdf = smooth_discontinuities(df_TMY)

