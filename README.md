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
