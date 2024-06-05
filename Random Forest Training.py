#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


# In[2]:


df = pd.read_csv("SDOH_Filtered_Data_Per_Capita.csv")
df = df.drop('STATE', axis = 1)
df = df.drop('COUNTY', axis = 1)


# In[3]:


df = df.sort_values(['COUNTYFIPS','YEAR'])


# In[4]:


# unneeded_prev_variables = ['YEAR', 'COUNTYFIPS', 'STATEFIPS', 'CDCA_HEART_DTH_RATE_ABOVE35']
# for col in df.columns:
#     if (col in unneeded_prev_variables):
#         continue
#     df[col + ' (t-1)'] = df.groupby('COUNTYFIPS')[col].shift(fill_value = -1000)


# In[5]:


# df = df[df.YEAR != 2009]


# In[6]:


df_county_codes = df["COUNTYFIPS"]
df_county_codes.drop_duplicates(inplace = True)


# In[7]:


features = df
labels = np.array(features['CDCA_HEART_DTH_RATE_ABOVE35'])
features= features.drop('CDCA_HEART_DTH_RATE_ABOVE35', axis = 1)
feature_list = list(features.columns)
# train_features = features[features.YEAR < 2016]
# test_features = features[features.YEAR >= 2016]
features = np.array(features)


# In[8]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[9]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf.score(test_features, test_labels))


# In[10]:


import joblib
# joblib.dump(rf, 'rf_model_4.pkl')
rf = joblib.load('rf_model_4.pkl')


# In[10]:


df = pd.read_csv("SDOH_Filtered_Data_Per_Capita.csv")
df = df.drop('STATE', axis = 1)
df = df.drop('COUNTY', axis = 1)
df = df.sort_values(['COUNTYFIPS','YEAR'])
unneeded_prev_variables = ['YEAR', 'COUNTYFIPS', 'STATEFIPS']
for col in df.columns:
    if (col in unneeded_prev_variables):
        continue
    df[col + ' (t-1)'] = df.groupby('COUNTYFIPS')[col].shift(fill_value = -1000)
df = df[df.YEAR != 2009]
features = df
labels = np.array(features['CDCA_HEART_DTH_RATE_ABOVE35'])
features= features.drop('CDCA_HEART_DTH_RATE_ABOVE35', axis = 1)
feature_list = list(features.columns)
features = np.array(features)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
from sklearn.ensemble import RandomForestRegressor
rf_past = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_past.fit(train_features, train_labels)
predictions = rf_past.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_past.score(test_features, test_labels))


# In[14]:


import joblib
joblib.dump(rf_past, 'rf_model_past.pkl')


# In[11]:


importances = list(rf_past.feature_importances_)


# In[12]:


feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]


# In[13]:


[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[15]:


import joblib
rf = joblib.load('rf_model_past.pkl')


# In[11]:


df_counties = {}
for county in df["COUNTYFIPS"].unique():
    df_counties[str(county)] = df[df["COUNTYFIPS"] == county]


# In[12]:


import scipy
from scipy.interpolate import interp1d
import pylab as P
from collections import defaultdict


# In[13]:


f = defaultdict(list)
for df_county in df_counties:
    for variable in df_counties[df_county].columns:
        if (variable == "CDCA_HEART_DTH_RATE_ABOVE35" or variable == "YEAR" or variable == "COUNTYFIPS" or variable == "STATEFIPS"):
            continue
        f[df_county].append(scipy.interpolate.interp1d(df_counties[df_county].YEAR, df_counties[df_county][variable], kind = "linear", fill_value='extrapolate'))


# In[14]:


df.columns


# In[ ]:





# In[15]:


def get_extrapolated_data (countyFIPS, stateFIPS, year, changeValue = False, columnToChange = "", changeBy = 0):
    variable_sign = [1, 1,-1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1]
    if (type(year) == int):
        countyFIPS = str(int(countyFIPS))
        stateFIPS = str(int(stateFIPS))
        dfx = pd.DataFrame()
        values = [year, countyFIPS, stateFIPS]
        ind = 0
        find = 0
        while (ind < len(df.columns)):
            variable = df.columns[ind]
#             print (variable)
            if (variable == "CDCA_HEART_DTH_RATE_ABOVE35" or variable == "YEAR" or variable == "COUNTYFIPS" or variable == "STATEFIPS"):
                ind += 1
                continue
            offset = 0
            if (changeValue and variable == columnToChange):
                offset = float(f[countyFIPS][find](year)) *  changeBy * variable_sign[find] / 100
            values.append(float(f[countyFIPS][find](year)) + offset)
            ind += 1
            find += 1
        dfx = dfx.append(pd.Series(values), ignore_index = True)
        return dfx
    elif (type(year) == list):
        dfx = pd.DataFrame()
        countyFIPS = str(int(countyFIPS))
        stateFIPS = str(int(stateFIPS))
        for y in range (year[0], year[1] + 1):
            values = [y, countyFIPS, stateFIPS]
            ind = 0
            find = 0
            while (ind < len(df.columns)):
                variable = df.columns[ind]
                if (variable == "CDCA_HEART_DTH_RATE_ABOVE35" or variable == "YEAR" or variable == "COUNTYFIPS" or variable == "STATEFIPS"):
                    ind += 1
                    continue
                offset = 0
                if (changeValue and variable == columnToChange):
                    offset = float(f[countyFIPS][find](y)) *  changeBy  * variable_sign[find] / 100
                values.append(float(f[countyFIPS][find](y)) + offset)
                ind += 1
                find += 1
            dfx[y] = pd.Series(values)
        return dfx.transpose()


# In[66]:


variables = df.columns
print (len(variables))
best_variables = {}
chosen_year = 2024
top_variables = defaultdict(list)
chosen_year_forecast = {}
for code in df_county_codes:
    code = str(code)
    forecast_df = get_extrapolated_data (code, code[0:-3], 2024, False)
    chosen_year_forecast[code] = rf.predict(forecast_df)[0]
    for variable in variables:
        extrap_df = get_extrapolated_data (code, code[0:-3], chosen_year, True, variable, 10)
        forecast_pred = rf.predict(extrap_df)
        year = 2024
        for pred in forecast_pred:
            if year == chosen_year:
                best_variables[variable] = float(pred)
    #         print ("Year: " + str(year) + " Death Rate: " + str(float(pred)))
            year += 1
    keys = list(best_variables.keys())
    values = list(best_variables.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    print (code, end = ' ')
    print (chosen_year_forecast[code], end= ' ')
    print(list(sorted_dict.keys())[0:3], list(sorted_dict.values())[0:3])
    top_variables[code].append(list(sorted_dict.keys())[0:3])
    top_variables[code].append(list(sorted_dict.values())[0:3])


# In[67]:


np.save('2024_forecast_county_2', np.array(dict(top_variables)))


# In[26]:


variables = df.columns
best_variables = {}
chosen_year = 2024
top_variables = defaultdict(list)
chosen_year_forecast = {}
for code in df_county_codes:
    code = str(code)
    forecast_df = get_extrapolated_data (code, code[0:-3], 2024, False)
    chosen_year_forecast[code] = rf.predict(forecast_df)[0]


# In[45]:


top_variables["21033"][0][0] + " " + str(top_variables["21033"][1][0])


# In[46]:


chosen_year_forecast["21033"]


# In[56]:


count = {}
for county in top_variables:
    if (top_variables[county][0][0] not in count):
        count[top_variables[county][0][0]] = 0
    count[top_variables[county][0][0]] += 1
count = dict(sorted(count.items(), key=lambda item: item[1], reverse = True))
print (count)


# In[249]:


from sklearn.ensemble import RandomForestRegressor
rf_oob = RandomForestRegressor(n_estimators = 1000, random_state = 42, bootstrap = True, oob_score = True)
rf_oob.fit(train_features, train_labels)
predictions = rf_oob.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_oob.score(test_features, test_labels))


# In[70]:


features2 = df
train_labels2 = np.array(features2[features2.YEAR < 2017]['CDCA_HEART_DTH_RATE_ABOVE35'])
test_labels2 = np.array(features2[features2.YEAR >= 2017]['CDCA_HEART_DTH_RATE_ABOVE35'])
features2 = features2.drop('CDCA_HEART_DTH_RATE_ABOVE35', axis = 1)
feature_list2 = list(features2.columns)
train_features2 = features2[features2.YEAR < 2017]
test_features2 = features2[features2.YEAR >= 2017]
# features = np.array(features)


# In[71]:


print('Training Features Shape:', train_features2.shape)
print('Training Labels Shape:', train_labels2.shape)
print('Testing Features Shape:', test_features2.shape)
print('Testing Labels Shape:', test_labels2.shape)


# In[65]:


rf_timed_split = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_timed_split.fit(train_features2, train_labels2)
predictions = rf_timed_split.predict(test_features2)
errors = abs(predictions - test_labels2)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_timed_split.score(test_features2, test_labels2))


# In[72]:


rf_timed_split2 = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_timed_split2.fit(train_features2, train_labels2)
predictions = rf_timed_split2.predict(test_features2)
errors = abs(predictions - test_labels2)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_timed_split2.score(test_features2, test_labels2))


# In[73]:


import csv


# In[75]:


np.save('2024_forecast_county', np.array(dict(top_variables)))


# In[79]:


top_variables = np.load('2024_forecast_county_2.npy', allow_pickle = True)


# In[80]:


top_variables


# In[81]:


top_variables = dict(enumerate(top_variables.flatten(), 1))[1]


# In[82]:


top_variables


# In[83]:


best_variable = []
for i in dict(top_variables).values():
    best_variable.append(i[0][0])


# In[115]:


best_vars, best_vars_count = np.unique(np.array(best_variable), return_counts=True)


# In[116]:


best_vars_count_inds = best_vars_count.argsort()
sorted_best_vars_count = best_vars_count[best_vars_count_inds[::-1]]
sorted_best_vars = best_vars[best_vars_count_inds[::-1]]


# In[117]:


sorted_best_vars_count


# In[119]:


sorted_best_vars


# In[124]:


for i in dict(top_variables):
    if (top_variables.get(i)[0][0] == 'ACS_MEDIAN_HH_INC'):
        print (i)


# In[85]:


improved_rate = []
for i in dict(top_variables).values():
    improved_rate.append(i[1][0])


# In[86]:


improved_rate


# In[87]:


chosen_year_forecast_df = pd.DataFrame()
chosen_year_forecast_df['COUNTYFIPS'] = pd.Series(list(chosen_year_forecast.keys())).apply(lambda x: str(x).zfill(5))
chosen_year_forecast_df['pred_heart_death_rate'] = pd.Series(list(chosen_year_forecast.values()))
chosen_year_forecast_df['best_variable'] = pd.Series(best_variable)
chosen_year_forecast_df['improved_heart_death_rate'] = pd.Series(improved_rate)


# In[88]:


chosen_year_forecast_df.loc[chosen_year_forecast_df['COUNTYFIPS'] == '06013']


# In[89]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[90]:


list(chosen_year_forecast.values())


# In[ ]:





# In[91]:


import plotly.express as px
fig = px.choropleth(chosen_year_forecast_df, geojson=counties, locations='COUNTYFIPS', color='pred_heart_death_rate',
                           color_continuous_scale="ylorbr",
                           range_color=(0, 1000),
                           scope="usa",
                           hover_name = "best_variable",
                           labels={'pred_heart_death_rate':'2024 predicted cardiovascular death rate', 'best_variable':'most impactful variable', 'improved_heart_death_rate':'predicted death rate after most impactful improved by 10%'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_traces(marker_line_width=0)
fig.show()


# In[ ]:





# In[92]:


best_variable1 = []
for i in dict(top_variables).values():
    best_variable1.append(i[0][0])
best_variable2 = []
for i in dict(top_variables).values():
    best_variable2.append(i[0][1])
best_variable3 = []
for i in dict(top_variables).values():
    best_variable3.append(i[0][2])
improved_rate1 = []
for i in dict(top_variables).values():
    improved_rate1.append(i[1][0])
improved_rate2 = []
for i in dict(top_variables).values():
    improved_rate2.append(i[1][1])
improved_rate3 = []
for i in dict(top_variables).values():
    improved_rate3.append(i[1][2])


# In[93]:


chosen_year_forecast_df = pd.DataFrame()
chosen_year_forecast_df['COUNTYFIPS'] = pd.Series(list(chosen_year_forecast.keys())).apply(lambda x: str(x).zfill(5))
chosen_year_forecast_df['pred_heart_death_rate'] = pd.Series(list(chosen_year_forecast.values()))
chosen_year_forecast_df['best_variable1'] = pd.Series(best_variable1)
chosen_year_forecast_df['improved_heart_death_rate1'] = pd.Series(improved_rate1)
chosen_year_forecast_df['best_variable2'] = pd.Series(best_variable2)
chosen_year_forecast_df['improved_heart_death_rate2'] = pd.Series(improved_rate2)
chosen_year_forecast_df['best_variable3'] = pd.Series(best_variable3)
chosen_year_forecast_df['improved_heart_death_rate3'] = pd.Series(improved_rate3)
chosen_year_forecast_df['best_variable1'].value_counts()['ACS_MEDIAN_HH_INC']


# In[94]:


fig2 = px.choropleth(chosen_year_forecast_df, geojson=counties, locations='COUNTYFIPS', color='improved_heart_death_rate1',
                           color_continuous_scale="ylorbr",
                           range_color=(0, 1000),
                           scope="usa",
                           hover_name = "best_variable1",
                           labels={'best_variable1':'most impactful variable', 'improved_heart_death_rate1':'improved death rate'}
                          )
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig2.update_traces(marker_line_width=0)
fig2.show()


# In[95]:


chosen_year_forecast_df[chosen_year_forecast_df.COUNTYFIPS == '06085']


# In[96]:


df_SC = df[df.COUNTYFIPS == 6085]
df_SC = df_SC[['YEAR', 'CDCA_HEART_DTH_RATE_ABOVE35']]


# In[97]:


df.head()


# In[98]:


import matplotlib.pyplot as plt


# In[ ]:





# In[99]:


years = [2020, 2024]
SC_forecast = rf.predict(get_extrapolated_data("06085", "06", years, False))
years = list(range(years[0], years[1] + 1))
print (SC_forecast)
print (years)


# In[100]:


df_SC


# In[101]:


SC_changed1 = rf.predict(get_extrapolated_data("06085", "06", [2020, 2024], True, top_variables['6085'][0][0], 10))
SC_changed2 = rf.predict(get_extrapolated_data("06085", "06", [2020, 2024], True, top_variables['6085'][0][1], 10))
SC_changed3 = rf.predict(get_extrapolated_data("06085", "06", [2020, 2024], True, top_variables['6085'][0][2], 10))


# In[102]:


top_variables['6085'][0]


# In[103]:


plt.plot(df_SC[df_SC.YEAR <= 2019].YEAR, df_SC[df_SC.YEAR <= 2019].CDCA_HEART_DTH_RATE_ABOVE35, color = "blue")
plt.plot(years, SC_forecast, color = "red")
plt.plot(years, SC_changed1, color = "orange")
plt.plot(years, SC_changed4, color = "purple")
plt.plot(years, SC_changed3, color = "green")
ax = plt.gca()
ax.set_ylim([250, 350])


# In[104]:


top_variables['6085'][0][0]


# In[105]:


years


# In[ ]:





# In[106]:


SC_changed1


# In[107]:


df_2019 = df[df.YEAR == 2019]


# In[108]:


df_2019


# In[109]:


df_2019[df_2019.CDCA_HEART_DTH_RATE_ABOVE35 == df_2019.CDCA_HEART_DTH_RATE_ABOVE35.max()]


# In[110]:


def generate_plot (years, countyfips):
    statefips = countyfips[0:2]
    forecast = rf.predict(get_extrapolated_data(countyfips, statefips, years, False))
    changed1 = rf.predict(get_extrapolated_data(countyfips, statefips, years, True, top_variables[str(int(countyfips))][0][0], 10))
    changed2 = rf.predict(get_extrapolated_data(countyfips, statefips, years, True, top_variables[str(int(countyfips))][0][1], 10))
    changed3 = rf.predict(get_extrapolated_data(countyfips, statefips, years, True, top_variables[str(int(countyfips))][0][2], 10))
#     print(top_variables[str(int(countyfips))][0])
    plt.figure(dpi=400)
    plt.plot(df[df.COUNTYFIPS == int(countyfips)].YEAR, df[df.COUNTYFIPS == int(countyfips)].CDCA_HEART_DTH_RATE_ABOVE35, color = "blue")
    plt.plot(list(range(years[0], years[1] + 1)), changed1, color = "orange", linestyle='dashed')
    plt.plot(list(range(years[0], years[1] + 1)), changed2, color = "purple", linestyle='dashed')
    plt.plot(list(range(years[0], years[1] + 1)), changed3, color = "green", linestyle='dashed')
    plt.plot(list(range(years[0], years[1] + 1)), forecast, color = "red", linestyle='dashed')
    plt.title(countyfips + " Cardiovascular Disease Death Rate Over Time")
    
    ax = plt.gca()
    improved = " improved"
    ax.legend(('actual', top_variables[str(int(countyfips))][0][0] + improved, top_variables[str(int(countyfips))][0][1] + improved, top_variables[str(int(countyfips))][0][2] + improved, 'forecast'))
    ax.set_ylim([min(forecast) - 50, max(forecast) + 150])


# In[ ]:





# In[112]:


generate_plot ([2009, 2025], "06085")


# In[302]:


from sklearn.ensemble import RandomForestRegressor
rf_small = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth = 20)
rf_small.fit(train_features, train_labels)
predictions = rf_small.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_small.score(test_features, test_labels))


# In[303]:


print('Score: ', rf_small.score(train_features, train_labels))


# In[312]:


from sklearn.ensemble import RandomForestRegressor
rf_small2 = RandomForestRegressor(n_estimators = 20, random_state = 42, max_depth = 40)
rf_small2.fit(train_features, train_labels)
predictions = rf_small2.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Score: ', rf_small2.score(test_features, test_labels))


# In[333]:


top_variables["48113"][0][0]


# In[357]:


df_2019[df_2019.COUNTYFIPS == "12011"]


# In[ ]:




